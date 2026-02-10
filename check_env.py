
import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
try:
    import cupy
    print(f"CuPy version: {cupy.__version__}")
    print(f"CuPy file: {cupy.__file__}")
except ImportError:
    print("CuPy not installed")
