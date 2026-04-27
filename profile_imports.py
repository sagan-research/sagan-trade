import time
import os

def profile_import(name):
    start = time.time()
    print(f"Importing {name}...", end="", flush=True)
    exec(f"import {name}")
    print(f" done in {time.time() - start:.2f}s")

profile_import("numpy")
profile_import("pandas")
profile_import("ollama")
profile_import("sagan.models.math_engine")
profile_import("sagan.models.llm_bridge")
print("All imports finished.")
