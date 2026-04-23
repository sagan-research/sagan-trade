import os
import requests
from pathlib import Path

# URLs for the raw model files
CENTERED_MODEL_URL = "https://raw.githubusercontent.com/That-Tech-Geek/model-store/main/centered_model.pkl"
CONTROLLER_MODEL_URL = "https://raw.githubusercontent.com/That-Tech-Geek/model-store/main/pretrained_controller_expanded.pth"

MODELS_DIR = Path(__file__).resolve().parent / "models"
CENTERED_MODEL_PATH = MODELS_DIR / "centered_model.pkl"
CONTROLLER_MODEL_PATH = MODELS_DIR / "pretrained_controller_expanded.pth"

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
    """Download the two model files if they are not already present.
    This function is safe to call multiple times; it will only fetch missing files.
    """
    if not CENTERED_MODEL_PATH.is_file():
        print(f"Downloading centered model… ({CENTERED_MODEL_URL})")
        _download(CENTERED_MODEL_URL, CENTERED_MODEL_PATH)
        print(f"Saved centered model to {CENTERED_MODEL_PATH}")
    else:
        print(f"Centered model already present at {CENTERED_MODEL_PATH}")

    if not CONTROLLER_MODEL_PATH.is_file():
        print(f"Downloading controller model… ({CONTROLLER_MODEL_URL})")
        _download(CONTROLLER_MODEL_URL, CONTROLLER_MODEL_PATH)
        print(f"Saved controller model to {CONTROLLER_MODEL_PATH}")
    else:
        print(f"Controller model already present at {CONTROLLER_MODEL_PATH}")

if __name__ == "__main__":
    # Allow manual execution
    download_if_missing()
