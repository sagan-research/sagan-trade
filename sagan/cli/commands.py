import json
import typer
from typing import List, Optional
from pathlib import Path

from sagan.config import config
from sagan.explain.gemma import run_explanation
from sagan.explain.gemma import run_explanation
from sagan.portfolio.csv_import import import_portfolio
from sagan.portfolio.snaptrade import get_snaptrade_holdings
from sagan.registry import list_models
from sagan.ensemble import train as train_ens
from sagan.predict import predict as predict_ens
from sagan.database import get_logs
from sagan.metrics import run_novelty_battery
from sagan.parallel import train_parallel_from_fetch
import subprocess
import sys
from sagan.metrics import run_novelty_battery
import subprocess
import sys

app = typer.Typer(help="Sagan XAI – Quantitative Trading Signal Library")


@app.command()
def train(
    tickers: List[str] = typer.Argument(..., help="Tickers to train on"),
    parallel: bool = typer.Option(False, "--parallel", help="Use parallel training (one model per ticker)"),
    epochs: int = typer.Option(30, "--epochs", help="Training epochs"),
    years: int = typer.Option(5, "--years", help="Years of data to fetch"),
):
    """Train a new signal ensemble."""
    
    typer.echo(f"Training ensemble for {tickers}...")
    
    if parallel:
        results = train_parallel_from_fetch(tickers, epochs=epochs, years=years)
        typer.echo(f"OK Parallel training complete: {list(results.values())}")
    else:
        model_id = train_ens(tickers, epochs=epochs, years=years)
        typer.secho(f"OK Training complete. Model ID: {model_id}", fg=typer.colors.GREEN)

@app.command()
def predict(
    model_id: Optional[str] = typer.Option(None, "--model-id", help="Model ID to use"),
    compliance: bool = typer.Option(False, "--compliance", help="Generate SEBI-compliant reports"),
):
    """Generate predictive signals and save to last_predict.json."""
    
    result = predict_ens(model_id=model_id, compliance=compliance)
    
    # Save for explain command
    last_predict_path = config.home_dir / "last_predict.json"
    with open(last_predict_path, "w") as f:
        json.dump(result, f, indent=2)
    
    color = typer.colors.GREEN if "LONG" in result["signal"] else (
        typer.colors.RED if "SHORT" in result["signal"] else typer.colors.YELLOW
    )
    
    typer.secho(f"\nSignal: {result['signal']}", fg=color, bold=True)
    typer.echo(f"Confidence: {result['confidence']:.2%}")
    typer.echo(f"Timestamp: {result['timestamp']}")
    
    if result["xai_justification"].get("conflict"):
        typer.secho("\n(!) CONFLICT DETECTED: ML Signal and Rule-based Thresholds disagree.", fg=typer.colors.BRIGHT_RED)
    
    typer.echo(f"\nJustification: {result['xai_justification']['reason']}")
    
    if compliance:
        typer.secho("\nOK Compliance reports generated in ~/.sagan/compliance/", fg=typer.colors.CYAN)

@app.command()
def userlogs(limit: int = typer.Option(20, help="Number of logs to show")):
    """View the local audit trail of trading actions."""
    logs = get_logs(limit=limit)
    if not logs:
        print("No logs found.")
        return
        
    print(f"\n{'Timestamp':25} | {'Action':18} | {'Model ID':15} | {'Conflict':8}")
    print("-" * 75)
    for log in logs:
        print(f"{log['timestamp']:25} | {log['action']:18} | {log['model_id'][:15]} | {log['conflict']}")

@app.command()
def explain():
    """Generate an LLM-powered explanation of the last prediction."""
    run_explanation()


@app.command("import")
def import_csv(file: Path = typer.Argument(..., help="Path to portfolio CSV")):
    """Import a portfolio from a CSV file."""
    df = import_portfolio(str(file))
    print(df.to_string(index=False))

@app.command()
def connect():
    """Connect to a brokerage via SnapTrade."""
    df = get_snaptrade_holdings()
    print(df.to_string(index=False))

@app.command("list")
def list_models_cmd():
    """List all trained models."""
    df = list_models()
    if df.empty:
        print("No models trained yet.")
    else:
        print(df.to_string(index=False))

@app.command()
def metrics():
    """Run the institutional-grade novelty battery benchmark."""
    run_novelty_battery()

@app.command()
def dash():
    """Launch the Sagan Quant Studio (Streamlit) dashboard."""
    app_path = Path(__file__).parent.parent / "app.py"
    if not app_path.exists():
        typer.secho(f"❌ Error: Dashboard file not found at {app_path}", fg=typer.colors.RED)
        return
    
    typer.secho(f"🚀 Starting Sagan Quant Studio dashboard...", fg=typer.colors.GREEN)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)
    except KeyboardInterrupt:
        typer.echo("\n👋 Dashboard stopped.")

if __name__ == "__main__":
    app()
