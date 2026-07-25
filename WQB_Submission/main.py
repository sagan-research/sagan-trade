import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from core.orchestrator import Orchestrator

def main():
    parser = argparse.ArgumentParser(description="WQB Submission System")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode (mock API calls)")
    args = parser.parse_args()

    print("Initializing WQB Submission System...")
    if args.dry_run:
        print("[DRY RUN MODE ENABLED] - API calls will be mocked.")
    
    if not args.dry_run and not os.path.exists("credentials.json"):
        print("ERROR: 'credentials.json' not found.")
        print("Please copy 'credentials.json.example' to 'credentials.json' and add your credentials.")
        sys.exit(1)
        
    try:
        orchestrator = Orchestrator(dry_run=args.dry_run)
        
        # If dry-run, we probably only want to run one loop to test the logic
        if args.dry_run:
            import asyncio
            print("\n--- Performing a single dry-run batch ---")
            asyncio.run(orchestrator.run_batch())
            print("Dry run completed successfully.")
        else:
            # 3 hours = 10800 seconds
            orchestrator.start_loop(timeout_seconds=10800)
    except Exception as e:
        print(f"Fatal error encountered: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
