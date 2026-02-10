
import sys
import os
import ctypes
import numpy as np

# Print python executable to be sure
print(f"Python: {sys.executable}")

# Calculate site-packages (logic from backend.py)
_site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Lib', 'site-packages')
print(f"Site-packages: {_site_packages}")

_nvidia_libs = [
    'nvidia/cuda_nvrtc/bin', 'nvidia/cuda_runtime/bin', 'nvidia/cublas/bin',
    'nvidia/curand/bin', 'nvidia/cusolver/bin', 'nvidia/cusparse/bin', 
    'nvidia/cufft/bin', 'nvidia/nvjitlink/bin'
]

dll_paths = []
for lib in _nvidia_libs:
    path = os.path.join(_site_packages, lib)
    if os.path.isdir(path):
        dll_paths.append(path)
        print(f"Found DLL path: {path}")
        try:
            os.add_dll_directory(path)
            os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
        except Exception as e:
            print(f"Failed to add dll directory {path}: {e}")

# Try to find nvrtc explicitly
nvrtc_path = os.path.join(_site_packages, 'nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll')
if os.path.exists(nvrtc_path):
    print(f"NVRTC found at: {nvrtc_path}")
    try:
        print("Attempting manual load with ctypes...")
        ctypes.CDLL(nvrtc_path)
        print("Manual load SUCCESS")
    except Exception as e:
        print(f"Manual load FAILED: {e}")
else:
    print(f"NVRTC NOT found at: {nvrtc_path}")

try:
    import cupy as cp
    xp = cp
    print("\nCuPy imported.")
except ImportError as e:
    print(f"\nCuPy import failed: {e}")
    sys.exit(1)

# Now try cupy ops
try:
    print("\nCreating array...")
    x = cp.array([1, 2, 3])
    print("Array created.")
    y = x * 2
    print(f"Result: {y}")
except Exception as e:
    print(f"CuPy Op FAILED: {e}")

# Try kernel
try:
    print("\nCompiling kernel...")
    k = cp.ElementwiseKernel('T x', 'T y', 'y = x * 2', 'k')
    z = k(x)
    print(f"Kernel Result: {z}")
except Exception as e:
    print(f"Kernel FAILED: {e}")
