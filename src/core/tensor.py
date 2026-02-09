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
from typing import Optional, List, Tuple, Union


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
        # Convert to numpy array for efficient computation
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None
        
        # For autograd - tracking computation graph
        self._backward = lambda: None  # Function to compute gradients
        self._prev = set(_children)    # Parent tensors
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
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='T'
        )
        
        def _backward():
            if self.requires_grad:
                grad = np.transpose(out.grad, axes)
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
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='sum'
        )
        
        def _backward():
            if self.requires_grad:
                # Gradient of sum is 1 for all elements
                grad = np.ones_like(self.data) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean of elements along an axis."""
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op='mean'
        )
        
        def _backward():
            if self.requires_grad:
                # Gradient of mean is 1/n for all elements
                n = self.data.size if axis is None else self.data.shape[axis]
                grad = np.ones_like(self.data) * out.grad / n
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
        self.grad = np.ones_like(self.data)
        
        # Backpropagate in reverse topological order
        for v in reversed(topo):
            v._backward()
    
    def zero_grad(self):
        """Reset gradients to None."""
        self.grad = None
    
    # ==================== Utility Methods ====================
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of zeros."""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor of ones."""
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        """Create a tensor with random normal values."""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
    
    @staticmethod
    def xavier_init(shape: Tuple[int, ...], requires_grad: bool = True) -> 'Tensor':
        """
        Xavier initialization - good for neural network weights.
        
        This keeps the variance of activations roughly the same
        across layers, which helps with training stability.
        """
        fan_in = shape[0] if len(shape) > 1 else shape[0]
        fan_out = shape[1] if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return Tensor(np.random.randn(*shape) * std, requires_grad=requires_grad)


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
    
    print("\n✓ Autograd is working!")
