import os
import subprocess
import requests
from pathlib import Path

# URLs for the raw model files (Fallback)
CENTERED_MODEL_URL = "https://raw.githubusercontent.com/That-Tech-Geek/model-store/main/centered_model.pkl"
CONTROLLER_MODEL_URL = "https://raw.githubusercontent.com/That-Tech-Geek/model-store/main/pretrained_controller_expanded.pth"

MODELS_DIR = Path(__file__).resolve().parent / "models"
CENTERED_MODEL_PATH = MODELS_DIR / "centered_model.pkl"
CONTROLLER_MODEL_PATH = MODELS_DIR / "pretrained_controller_expanded.pth"

KAGGLE_KERNEL = "toocool69/notebookcbd7d72330"

def _download_from_kaggle(dest_dir: Path) -> bool:
    """
    Attempts to download models using the Kaggle CLI.
    """
    try:
        print(f"Attempting to download models from Kaggle kernel: {KAGGLE_KERNEL}...")
        cmd = ["kaggle", "kernels", "output", KAGGLE_KERNEL, "-p", str(dest_dir)]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        return False

def _download(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest*.
    Raises an exception if the request fails.
    """
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def download_if_missing() -> None:
    """Download the required model files if they are not already present.
    Prioritizes Kaggle, falls back to GitHub URLs.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not CENTERED_MODEL_PATH.is_file() or not CONTROLLER_MODEL_PATH.is_file():
        # Try Kaggle first
        success = _download_from_kaggle(MODELS_DIR)
        
        # Fallback to direct downloads if Kaggle fails or files are still missing
        if not CENTERED_MODEL_PATH.is_file():
            print(f"Downloading centered model from fallback… ({CENTERED_MODEL_URL})")
            _download(CENTERED_MODEL_URL, CENTERED_MODEL_PATH)
            
        if not CONTROLLER_MODEL_PATH.is_file():
            print(f"Downloading controller model from fallback… ({CONTROLLER_MODEL_URL})")
            _download(CONTROLLER_MODEL_URL, CONTROLLER_MODEL_PATH)
    else:
        print(f"Models already present in {MODELS_DIR}")

if __name__ == "__main__":
    download_if_missing()
