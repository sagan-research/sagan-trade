"""Command-line interface for Sagan XAI"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from sagan.ensemble import train
from sagan.parallel import train_parallel_from_fetch
from sagan.predict import predict
from sagan.registry import list_models
from sagan.metrics import run_novelty_battery


def main():
    parser = argparse.ArgumentParser(
        prog="sagan",
        description="Sagan XAI – Explainable Probabilistic Ensemble for Trading",
    )
    parser.add_argument("--train", nargs="+", metavar="TICKER",
                        help="Train a new ensemble on the given tickers")
    parser.add_argument("--predict", action="store_true",
                        help="Predict using the latest saved model")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Model ID to use for prediction (default: latest)")
    parser.add_argument("--parallel", action="store_true",
                        help="Use parallel training (one model per ticker)")
    parser.add_argument("--num-processes", type=int, default=12,
                        help="Worker processes for parallel training (default: 12)")
    parser.add_argument("--list", action="store_true",
                        help="List all trained models")
    parser.add_argument("--metrics", action="store_true",
                        help="Run the novelty battery benchmark")
    parser.add_argument("--dash", action="store_true",
                        help="Alias for 'open': launch the Streamlit dashboard")
    parser.add_argument("--func", action="store_true",
                        help="List all available CLI functions with detailed descriptions")
    # Hyper-parameter overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--years", type=int, default=5)

    parser.add_argument("command", nargs="?", choices=["train", "predict", "list", "open", "metrics"],
                        help="Command to run (optional if using flags)")
    
    args, unknown = parser.parse_known_args()

    if args.func:
        print("\n" + "="*60)
        print("SAGAN CLI: DETAILED FUNCTION LIST")
        print("="*60)
        funcs = {
            "train": "Train a new Sagan Ensemble using Physics-Informed Neural Networks. Requires tickers.",
            "predict": "Generate real-time trade signals and XAI justifications using the latest model.",
            "list": "List all trained models currently stored in the local registry.",
            "open": "Launch the Sagan Quant Studio dashboard for live visual analysis. (Alias: --dash)",
            "metrics": "Execute the institutional-grade novelty battery benchmark (DM Test, JSD, etc.).",
        }
        for cmd, desc in funcs.items():
            print(f"- {cmd:10} | {desc}")
        print("="*60 + "\n")
        return

    if args.command == "open" or args.dash or (not args.command and len(sys.argv) > 1 and sys.argv[1] == "open"):
        # Launch Streamlit dashboard
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
            print(f"❌ Error: Dashboard file not found at {app_path}")
            return
        
        print(f"🚀 Starting Sagan Quant Studio dashboard...")
        try:
            # We use sys.executable to ensure we use the same environment
            result = subprocess.run([
                sys.executable, "-m", "streamlit", "run", str(app_path)
            ], check=False)
        except KeyboardInterrupt:
            print("\n👋 Dashboard stopped.")
        return

    if args.command == "metrics" or args.metrics:
        # Run novelty battery
        run_novelty_battery()
        return

    # Fallback to old flag-based parsing if no command positional is used
    if not args.command:
        # Re-parse with flags if no positional command was recognized
        args = parser.parse_args()

    if args.list or args.command == "list":
        df = list_models()
        if df.empty:
            print("No models trained yet.")
        else:
            print(df.to_string(index=False))

    elif args.train or args.command == "train":
        # If positional 'train' is used, the tickers might be in unknown or we need more args
        tickers = args.train if args.train else unknown
        if not tickers:
            print("❌ Error: No tickers provided for training.")
            return
            
        kwargs = {}
        if args.epochs:
            kwargs["epochs"] = args.epochs
        if args.window:
            kwargs["window"] = args.window
        if args.horizon:
            kwargs["horizon"] = args.horizon
        kwargs["years"] = args.years

        if args.parallel:
            results = train_parallel_from_fetch(
                tickers, num_processes=args.num_processes, **kwargs
            )
            print(json.dumps(results, indent=2))
        else:
            mid = train(tickers, **kwargs)
            print(f"✅ Model saved: {mid}")

    elif args.predict or args.command == "predict":
        result = predict(model_id=args.model_id)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
