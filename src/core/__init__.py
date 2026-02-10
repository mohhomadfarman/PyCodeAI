# Core math utilities
from .tensor import Tensor
from .activations import relu, gelu, softmax, tanh
from .backend import xp, to_numpy, to_device, use_gpu, use_cpu, is_gpu, GPU_AVAILABLE
