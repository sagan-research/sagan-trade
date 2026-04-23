import os
from .download_models import download_if_missing

# Ensure models are present when the package is imported
_models_dir = os.path.join(os.path.dirname(__file__), "models")
if not os.path.isdir(_models_dir):
    os.makedirs(_models_dir, exist_ok=True)
# Download the required model files if they are not already present
download_if_missing()

__all__ = [
    "loader",
    "utils",
    "trainer",
    "predictor",
    "download_models",
]

