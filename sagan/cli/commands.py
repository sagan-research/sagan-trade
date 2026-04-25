import json
import typer
from typing import List, Optional
from pathlib import Path

from sagan.config import config
from sagan.explain.gemma import run_explanation
from sagan.portfolio.csv_import import import_portfolio
from sagan.portfolio.snaptrade import get_snaptrade_holdings
from sagan.registry import list_models
from sagan.ensemble import train as train_ens
from sagan.predict import predict as predict_ens
from sagan.database import get_logs
from sagan.parallel import train_parallel
import subprocess
import sys

app = typer.Typer(help="Sagan XAI – Quantitative Trading Signal Library")


@app.command()
def train(
    tickers: List[str] = typer.Argument(..., help="Tickers to train on"),
    signals: Optional[List[str]] = typer.Option(None, "--signals", "-s", help="Signals to include (e.g. Open, High, Volume)"),
    years: int = typer.Option(1, "--years", help="Years of data to fetch"),
    profile: str = typer.Option("balanced", "--profile", help="Performance profile (eco, balanced, turbo)"),
):
    """Train a new symbolic mathematical ensemble."""
    
    typer.echo(f"Training symbolic ensemble for {tickers}...")
    typer.echo(f"Performance Profile: {profile.upper()}")
    
    # Handle comma-separated signals in a single string if provided via CLI
    final_signals = []
    if signals:
        for s in signals:
            final_signals.extend([x.strip() for x in s.split(",")])
    else:
        final_signals = None

    model_id = train_ens(tickers, signals=final_signals, period=f"{years}y", profile=profile)
    typer.secho(f"OK Training complete. Model ID: {model_id}", fg=typer.colors.GREEN)

@app.command()
def train_portfolio(
    tickers: str = typer.Argument(..., help="Comma-separated tickers"),
    profile: str = typer.Option("balanced", "--profile", help="Performance profile"),
):
    """Develop independent mathematical foundations for a portfolio."""
    ticker_list = [t.strip() for t in tickers.split(",")]
    typer.echo(f"Developing Portfolio Foundations for {ticker_list}...")
    mids = train_parallel(ticker_list, profile=profile)
    typer.secho(f"OK Portfolio ready. Registered IDs: {list(mids.values())}", fg=typer.colors.GREEN)

@app.command()
def vars(ticker: str = typer.Argument(..., help="Ticker to explore")):
    """List all available signals for a ticker (from yfinance)."""
    from sagan.signals import get_available_signals
    available = get_available_signals(ticker)
    typer.echo(f"Available signals for {ticker}:")
    for s in available:
        typer.echo(f" - {s}")

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
def research(
    ticker: str = typer.Argument(..., help="Ticker to research"),
    formula: str = typer.Option("(Close / RSI) * Volume", "--formula", "-f", help="Symbolic formula to backtest"),
    period: str = typer.Option("2y", "--period", help="Historical period (1y, 2y, 5y)"),
):
    """Run a symbolic backtest research on a custom formula."""
    from sagan.research import BacktestEngine
    
    typer.echo(f"Running Symbolic Research for {ticker}...")
    typer.echo(f"Formula: {formula}")
    
    engine = BacktestEngine(ticker, formula, period=period)
    results = engine.run()
    
    if results["status"] == "success":
        typer.secho("\n--- Research Results ---", fg=typer.colors.CYAN, bold=True)
        typer.echo(f"Total Return: {results['total_return']:.2%}")
        typer.echo(f"B&H Return:   {results['bh_return']:.2%}")
        typer.echo(f"Sharpe Ratio: {results['sharpe']:.2f}")
        typer.echo(f"Max Drawdown: {results['max_drawdown']:.2%}")
        typer.echo(f"Win Rate:     {results['win_rate']:.2%}")
    else:
        typer.secho(f"\nError: {results['message']}", fg=typer.colors.RED)

@app.command()
def auto(
    ticker: str = typer.Argument(..., help="Ticker for autonomous research"),
    period: str = typer.Option("2y", "--period", help="Historical period"),
):
    """Run the end-to-end autonomous research pipeline."""
    from sagan.autonomous import AutonomousResearcher
    
    typer.secho(f"Launching Autonomous Alpha Discovery for {ticker}...", fg=typer.colors.CYAN, bold=True)
    
    researcher = AutonomousResearcher()
    results = researcher.run_full_pipeline(ticker, period=period)
    
    if results["status"] == "success":
        typer.secho("\n--- Discovery Complete ---", fg=typer.colors.GREEN, bold=True)
        typer.echo(f"Formula: {results['formula']}")
        typer.echo(f"Backtest Return: {results['backtest']['total_return']:.2%}")
        typer.echo(f"Sharpe Ratio:    {results['backtest']['sharpe']:.2f}")
        typer.echo("\n--- Positioning Advice ---")
        typer.secho(results["advice"], fg=typer.colors.YELLOW)
    else:
        typer.secho(f"\nError: {results.get('message', 'Unknown error')}", fg=typer.colors.RED)

@app.command()
def metrics():
    """Run the institutional-grade novelty battery benchmark."""
    run_novelty_battery()

@app.command()
def dash():
    """Launch the Sagan Quant Studio (Streamlit) dashboard."""
    app_path = Path(__file__).parent.parent / "app.py"
    if not app_path.exists():
        typer.secho(f"Error: Dashboard file not found at {app_path}", fg=typer.colors.RED)
        return
    
    typer.secho(f"Starting Sagan Quant Studio dashboard...", fg=typer.colors.GREEN)
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=False)
    except KeyboardInterrupt:
        typer.echo("\n👋 Dashboard stopped.")

if __name__ == "__main__":
    app()
