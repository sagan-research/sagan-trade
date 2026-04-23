import subprocess
import sys

def main():
    """Build source and wheel distributions using the `build` package."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "build"])
        subprocess.check_call([sys.executable, "-m", "build"])
        print("Build completed. Distributions are in the ./dist directory.")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
