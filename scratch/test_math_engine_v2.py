import numpy as np
import pandas as pd
from sagan.models.math_engine import MathematicalEngine
from sagan.symbolic_lib.download_models import CENTERED_MODEL_PATH

def test_math_engine():
    engine = MathematicalEngine()
    
    # Create some dummy data that follows the centered model roughly
    # -2*t_sym - 19.25
    t = np.arange(100)
    y = -2 * t - 19.25 + np.random.normal(0, 0.1, 100)
    
    print("Fitting variable with specialized models...")
    func, params, r2, std_err = engine.fit_variable(y)
    
    print(f"Best Func: {func}")
    print(f"R2: {r2:.4f}")
    print(f"Params: {params}")

if __name__ == "__main__":
    test_math_engine()
