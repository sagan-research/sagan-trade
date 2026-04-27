import sys
import subprocess

def check_package(package_name):
    try:
        __import__(package_name)
        print(f"[OK] {package_name} is installed.")
    except ImportError:
        print(f"[FAIL] {package_name} is NOT installed.")

def check_cli(command):
    try:
        result = subprocess.run([command, "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {command} CLI is available. Version: {result.stdout.strip()}")
        else:
            print(f"[FAIL] {command} CLI returned an error.")
    except FileNotFoundError:
        print(f"[FAIL] {command} CLI is NOT found in PATH.")

if __name__ == "__main__":
    print("--- Hugging Face Environment Check ---")
    check_package("huggingface_hub")
    check_package("transformers")
    check_package("datasets")
    print("\n--- CLI Check ---")
    check_cli("huggingface-cli")
