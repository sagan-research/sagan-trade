import torch
import pickle
import joblib
import os

def inspect_pth(path):
    print(f"\n--- Detailed Inspection of {path} ---")
    if not os.path.exists(path):
        print("File not found")
        return
    try:
        sd = torch.load(path, map_location='cpu')
        for k, v in sd.items():
            print(f"{k}: {v.shape}")
    except Exception as e:
        print(f"Error: {e}")

def inspect_pkl(path):
    print(f"\n--- Detailed Inspection of {path} ---")
    if not os.path.exists(path):
        print("File not found")
        return
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print("Type:", type(data))
        if isinstance(data, dict):
            print("Keys:", data.keys())
            for k, v in data.items():
                print(f"Key '{k}' type: {type(v)}")
                if hasattr(v, 'shape'):
                    print(f"  Shape: {v.shape}")
                elif isinstance(v, (list, dict)):
                    print(f"  Size/Len: {len(v)}")
        else:
            print("Value:", data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_pth(r"F:\Downloads\pretrained_controller_expanded.pth")
    inspect_pkl(r"F:\Downloads\centered_model.pkl")
