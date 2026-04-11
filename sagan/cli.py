"""Command-line interface for Sagan XAI"""

import argparse
import json

from sagan.ensemble import train
from sagan.parallel import train_parallel_from_fetch
from sagan.predict import predict
from sagan.registry import list_models


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
    # Hyper-parameter overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--years", type=int, default=5)

    args = parser.parse_args()

    if args.list:
        df = list_models()
        if df.empty:
            print("No models trained yet.")
        else:
            print(df.to_string(index=False))

    elif args.train:
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
                args.train, num_processes=args.num_processes, **kwargs
            )
            print(json.dumps(results, indent=2))
        else:
            mid = train(args.train, **kwargs)
            print(f"✅ Model saved: {mid}")

    elif args.predict:
        result = predict(model_id=args.model_id)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
