
import sys
import os
import numpy as np

# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Importing CuPy directly...")
try:
    import cupy as cp
    print("CuPy imported.")
    
    x = cp.array([1., 2., 3.], dtype=cp.float32)
    print("Array created.")
    
    y = x * 2.0
    print("Multiplication succeeded:", y)
    
except Exception as e:
    print(f"FAILED: {e}")
