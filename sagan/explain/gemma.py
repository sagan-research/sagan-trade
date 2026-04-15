import json
import subprocess
import shutil
import sys
from pathlib import Path
from sagan.config import config
from sagan.exceptions import SaganError
from sagan.explain.formatter import format_explain_prompt

LAST_PREDICT = config.home_dir / "last_predict.json"

def run_explanation():
    """Run Ollama/Gemma explanation for the latest prediction."""
    

    # 3. Dependency checks
    if not shutil.which("ollama"):
        print("Ollama is required for explanations. Install at https://ollama.ai then run: ollama pull gemma3n:e2b")
        return

    # Check model
    try:
        models = subprocess.check_output(["ollama", "list"], text=True)
        if "gemma3n:e2b" not in models:
            print("Model gemma3n:e2b missing. Pulling now...")
            subprocess.run(["ollama", "pull", "gemma3n:e2b"], check=True)
    except subprocess.CalledProcessError:
        print("Error checking Ollama models.")
        return

    # 4. Load prediction
    if not LAST_PREDICT.exists():
        print("No recent prediction found. Run `sagan --predict` first.")
        return
    
    with open(LAST_PREDICT, "r") as f:
        prediction = json.load(f)

    # 5. Format prompt
    prompt = format_explain_prompt(prediction)

    # 6. Run Ollama and stream
    print("\n--- Generating Explanation ---\n")
    try:
        
        process = subprocess.Popen(
            ["ollama", "run", "gemma3n:e2b", prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8' # Ensure proper encoding for windows
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
            
        process.wait()
        if process.returncode != 0:
            print(f"\nOllama exited with error code {process.returncode}")

    except Exception as e:
        print(f"Failed to run explanation: {e}")
