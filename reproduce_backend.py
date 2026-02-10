
import sys
import os
import numpy as np

try:
    import cupy as cp
    print("CuPy imported")
    
    try:
        print("Setting allocator...")
        # This is the line I added
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
        print("Allocator set successfully")
    except Exception as e:
        print(f"Allocator setting failed: {e}")
        
except ImportError:
    print("CuPy not available")
