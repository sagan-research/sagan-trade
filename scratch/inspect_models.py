import torch
import pickle
import joblib
import os

def inspect_pth(path):
    print(f"\n--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found")
        return
    try:
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict):
            print("Keys in checkpoint:", checkpoint.keys())
            if 'model_state_dict' in checkpoint:
                print("Found model_state_dict")
            if 'state_dict' in checkpoint:
                print("Found state_dict")
        else:
            print("Loaded object type:", type(checkpoint))
    except Exception as e:
        print(f"Error loading .pth: {e}")

def inspect_pkl(path):
    print(f"\n--- Inspecting {path} ---")
    if not os.path.exists(path):
        print("File not found")
        return
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        print("Loaded object type:", type(model))
        if hasattr(model, 'get_params'):
            print("Params:", model.get_params())
    except Exception as e:
        print(f"Error loading .pkl with pickle: {e}")
        try:
            model = joblib.load(path)
            print("Loaded object type (joblib):", type(model))
        except Exception as e2:
            print(f"Error loading .pkl with joblib: {e2}")

if __name__ == "__main__":
    inspect_pth(r"F:\Downloads\pretrained_controller_expanded.pth")
    inspect_pkl(r"F:\Downloads\centered_model.pkl")
