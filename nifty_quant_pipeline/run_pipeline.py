import os
import json
from datetime import datetime
from research_engine import run_quantitative_research

# Using the specified AIN publication directory
OUTPUT_DIR = "C:/Users/91891/.gemini/antigravity/scratch/personal-intel/vault/wiki/05_Publications/auto_generated"

def main():
    print("Initializing Autonomous Quantitative Pipeline...")
    
    # Run the core research engine
    report = run_quantitative_research()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate unique filename for the Sunday cron job
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"NIFTY50_HRP_CVaR_Report_{date_str}.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Save the JSON report
    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"\n[SUCCESS] Pipeline complete. Quantitative research paper published to:")
    print(f"[FILE] {filepath}")

if __name__ == "__main__":
    main()
