"""
Tensor - The fundamental building block of neural networks.

A Tensor is a multi-dimensional array with automatic gradient computation.
This is similar to PyTorch's Tensor but built from scratch!

Learning Goals:
- Understand how neural networks store and manipulate data
- Learn how gradients flow backward through operations
- See how autograd (automatic differentiation) works
"""

import numpy as np
from . import backend as _backend
from .backend import to_numpy
from typing import Optional, List, Tuple, Union

# Global flag: when True, skip all gradient tracking (for inference)
_no_grad = False


class no_grad:
    """
    Context manager to disable gradient computation (for inference).

    Usage:
        with no_grad():
            output = model(input)  # No backward graph built, much faster
    """

    def __enter__(self):
        global _no_grad
        self._prev = _no_grad
        _no_grad = True
        return self

    def __exit__(self, *args):
        global _no_grad
        _no_grad = self._prev


class Tensor:
    """
    A multi-dimensional array with automatic gradient computation.

    Example:
        >>> x = Tensor([1, 2, 3], requires_grad=True)
        >>> y = x * 2
        >>> z = y.sum()
        >>> z.backward()
        >>> print(x.grad)  # Gradient of z with respect to x
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float],
        requires_grad: bool = False,
        _children: Tuple['Tensor', ...] = (),
        _op: str = ''
    ):
        """
        Initialize a Tensor.

        Args:
            data: The actual numbers (can be list, numpy array, or scalar)
            requires_grad: If True, gradients will be computed for this tensor
            _children: Internal - tensors that created this tensor
            _op: Internal - the operation that created this tensor
        """
        # Get current backend dynamically (supports CPU/GPU switching)
        xp = _backend.xp

        # Convert to array on the active device for efficient computation
        if isinstance(data, xp.ndarray):
            # Zero-copy if already float32, otherwise cast
            self.data = data if data.dtype == xp.float32 else data.astype(xp.float32)
        elif isinstance(data, np.ndarray):
            self.data = xp.array(data, dtype=xp.float32)
        else:
            self.data = xp.array(data, dtype=xp.float32)

        # In no_grad mode, never track gradients
        self.requires_grad = False if _no_grad else requires_grad
        self.grad = None

        # For autograd - tracking computation graph
        self._backward = lambda: None  # Function to compute gradients
        self._prev = set() if _no_grad else set(_children)
        self._op = _op                 # Operation name (for debugging)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the tensor."""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    # ==================== Basic Operations ====================

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Add two tensors: z = x + y"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='+'
        )

        if not _no_grad:
            def _backward():
                if self.requires_grad:
                    # d(x+y)/dx = 1, so gradient flows unchanged
                    self.grad = self.grad + out.grad if self.grad is not None else out.grad.copy()
                if other.requires_grad:
                    other.grad = other.grad + out.grad if other.grad is not None else out.grad.copy()

            out._backward = _backward
        return out

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Multiply two tensors element-wise: z = x * y"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='*'
        )

        def _backward():
            if self.requires_grad:
                # d(x*y)/dx = y
                grad = other.data * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                # d(x*y)/dy = x
                grad = self.data * out.grad
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        return out

    def __neg__(self) -> 'Tensor':
        """Negate: -x"""
        return self * -1

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Subtract: x - y"""
        return self + (-other)

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right subtract: y - x"""
        return other + (-self)

    def __radd__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right add: y + x"""
        return self + other

    def __rmul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Right multiply: y * x"""
        return self * other

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Divide: x / y"""
        return self * (other ** -1)

    def __pow__(self, power: float) -> 'Tensor':
        """Power: x^n"""
        assert isinstance(power, (int, float)), "Power must be a number"
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f'^{power}'
        )

        def _backward():
            if self.requires_grad:
                # d(x^n)/dx = n * x^(n-1)
                grad = power * (self.data ** (power - 1)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    # ==================== Matrix Operations ====================

    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Matrix multiplication: z = x @ y

        This is the KEY operation in neural networks!
        Every layer is essentially: output = input @ weights + bias
        """
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op='@'
        )

        def _backward():
            if self.requires_grad:
                # d(x@y)/dx = grad @ y.T
                grad = out.grad @ other.data.T
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                # d(x@y)/dy = x.T @ grad
                grad = self.data.T @ out.grad
                other.grad = other.grad + grad if other.grad is not None else grad

        out._backward = _backward
        return out

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Allow using @ operator for matmul."""
        return self.matmul(other)

    def transpose(self, dim0: int = -2, dim1: int = -1) -> 'Tensor':
        """Transpose the tensor."""
        xp = _backend.xp
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]

        out = Tensor(
            xp.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='T'
        )

        def _backward():
            if self.requires_grad:
                grad = xp.transpose(out.grad, axes)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    @property
    def T(self) -> 'Tensor':
        """Shorthand for transpose."""
        return self.transpose()

    # ==================== Reduction Operations ====================

    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Sum elements along an axis."""
        xp = _backend.xp
        out = Tensor(
            xp.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )

        def _backward():
            if self.requires_grad:
                # Gradient of sum is 1 for all elements
                grad = xp.ones_like(self.data) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean of elements along an axis."""
        xp = _backend.xp
        out = Tensor(
            xp.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='mean'
        )

        def _backward():
            if self.requires_grad:
                # Gradient of mean is 1/n for all elements
                n = self.data.size if axis is None else self.data.shape[axis]
                grad = xp.ones_like(self.data) * out.grad / n
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    # ==================== Shape Operations ====================

    def reshape(self, *shape) -> 'Tensor':
        """Reshape the tensor."""
        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='reshape'
        )

        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.shape)
                self.grad = self.grad + grad if self.grad is not None else grad

        out._backward = _backward
        return out

    def view(self, *shape) -> 'Tensor':
        """Alias for reshape (PyTorch compatibility)."""
        return self.reshape(*shape)

    # ==================== Backpropagation ====================

    def backward(self):
        """
        Compute gradients using backpropagation.

        This is THE algorithm that makes neural networks learn!
        It computes d(loss)/d(parameter) for every parameter.

        How it works:
        1. Build a topological order of all tensors in the graph
        2. Starting from the output, propagate gradients backward
        3. Each operation knows how to compute its local gradients
        """
        xp = _backend.xp

        # Build topological order
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Initialize gradient of output to 1
        self.grad = xp.ones_like(self.data)

        # Backpropagate in reverse topological order
        for v in reversed(topo):
            v._backward()

        # Free computation graph to prevent GPU memory leak.
        # _backward closures capture large GPU arrays (Q,K,V heads, attention
        # weights, etc). _prev references form cycles. Without clearing,
        # these stay alive until Python's cyclic GC runs, causing progressive
        # VRAM exhaustion and slowdown.
        for v in topo:
            v._backward = lambda: None
            v._prev = set()

    def zero_grad(self):
        """Reset gradients to None."""
        self.grad = None

    # ==================== Utility Methods ====================

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of zeros."""
        return Tensor(_backend.xp.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones."""
        return Tensor(_backend.xp.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor with random normal values."""
        return Tensor(_backend.xp.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def xavier_init(shape: Tuple[int, ...], requires_grad: bool = True) -> 'Tensor':
        """
        Xavier initialization - good for neural network weights.

        This keeps the variance of activations roughly the same
        across layers, which helps with training stability.
        """
        xp = _backend.xp
        fan_in = shape[0] if len(shape) > 1 else shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        std = float(xp.sqrt(2.0 / (fan_in + fan_out)))
        return Tensor(xp.random.randn(*shape) * std, requires_grad=requires_grad)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Tensor Autograd Demo")
    print("=" * 50)

    # Create tensors
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    w = Tensor([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)

    print(f"x = {x.data}")
    print(f"w = {w.data}")

    # Forward pass: y = x @ w, then sum
    y = x @ w
    loss = y.sum()

    print(f"\ny = x @ w = {y.data}")
    print(f"loss = sum(y) = {loss.data}")

    # Backward pass
    loss.backward()

    print(f"\nGradients:")
    print(f"x.grad = {x.grad}")
    print(f"w.grad = {w.grad}")

    print("\n[OK] Autograd is working!")
