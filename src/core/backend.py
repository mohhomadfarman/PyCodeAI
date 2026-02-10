"""
Backend abstraction for CPU (NumPy) and GPU (CuPy) computation.

Provides a unified interface that automatically uses GPU when available.
All compute modules import `xp` from here instead of numpy directly.
"""

import numpy as np
import os
import sys

# Add NVIDIA DLL directories to search path (Windows needs this for CuPy)
if sys.platform == 'win32':
    import ctypes
    _site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'Lib', 'site-packages')
    _nvidia_libs = [
        'nvidia/cuda_nvrtc/bin', 'nvidia/cuda_runtime/bin', 'nvidia/cublas/bin',
        'nvidia/curand/bin', 'nvidia/cusolver/bin', 'nvidia/cusparse/bin',
        'nvidia/cufft/bin', 'nvidia/nvjitlink/bin',
    ]
    for _nvidia_lib in _nvidia_libs:
        _dll_path = os.path.join(_site_packages, _nvidia_lib)
        if os.path.isdir(_dll_path):
            try:
                os.add_dll_directory(_dll_path)
                os.environ['PATH'] = _dll_path + os.pathsep + os.environ.get('PATH', '')
            except Exception:
                pass
            
            # Preload DLLs to avoid load errors
            try:
                for file in os.listdir(_dll_path):
                    if file.endswith('.dll'):
                        try:
                            ctypes.CDLL(os.path.join(_dll_path, file))
                        except Exception:
                            pass
            except Exception:
                pass

# Try to import CuPy for GPU acceleration
try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='CUDA path could not be detected')
        import cupy as cp
    GPU_AVAILABLE = True
    # Set memory pool limit to reduce fragmentation (leave 1GB for system)
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=3 * 1024**3)  # 3GB for 4GB GPU

    # Enable TF32 for Ampere+ GPUs (significant speedup for float32)
    try:
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
    except Exception:
        pass

except (ImportError, Exception):
    cp = None
    GPU_AVAILABLE = False

# Active array module - defaults to GPU if available
_use_gpu = GPU_AVAILABLE
xp = cp if _use_gpu else np


def use_gpu():
    """Switch to GPU backend."""
    global xp, _use_gpu
    if not GPU_AVAILABLE:
        print("WARNING: CuPy not available. Staying on CPU.")
        return
    xp = cp
    _use_gpu = True
    dev = cp.cuda.Device(0)
    name = dev.attributes.get("DeviceName", "Unknown GPU") if hasattr(dev, 'attributes') else "NVIDIA GPU"
    mem = dev.mem_info
    print(f"Backend: GPU ({name}, {mem[1] // (1024**2)}MB VRAM)")


def use_cpu():
    """Switch to CPU backend."""
    global xp, _use_gpu
    xp = np
    _use_gpu = False
    print("Backend: CPU (NumPy)")


def is_gpu():
    """Check if GPU backend is active."""
    return _use_gpu


def to_numpy(array):
    """Convert any array to numpy (for file I/O, printing, etc.)."""
    if GPU_AVAILABLE and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def to_device(array):
    """Convert numpy array to the active backend's array."""
    if _use_gpu:
        return cp.asarray(array)
    return np.asarray(array)
