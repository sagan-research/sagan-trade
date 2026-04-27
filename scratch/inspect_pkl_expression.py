import torch
import pickle
import os

def inspect_pkl(path):
    print(f"\n--- Detailed Inspection of {path} ---")
    if not os.path.exists(path):
        print("File not found")
        return
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            print("Expression:", data.get('expression'))
            print("Mean:", data.get('y_mean_c'))
            print("Std:", data.get('y_std_c'))
            print("Window:", data.get('window'))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_pkl(r"F:\Downloads\centered_model.pkl")
