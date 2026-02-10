"""
Linear (Dense) Layer - The fundamental neural network layer.

A linear layer performs: y = x @ W + b
Where:
- x is the input
- W is the weight matrix (learned)
- b is the bias vector (learned)

This is how information flows through a neural network!

Learning Goals:
- Understand the most basic neural network operation
- Learn about weights and biases
- See how gradients are computed for weight updates
"""

import numpy as np
from ..core import backend as _backend
from ..core.tensor import Tensor
from typing import Optional


class Linear:
    """
    Linear (Fully Connected) Layer.
    
    Transforms input of size (batch, in_features) to (batch, out_features).
    
    Example:
        layer = Linear(256, 128)  # 256 inputs → 128 outputs
        x = Tensor(np.randn(32, 256))  # Batch of 32
        y = layer(x)  # Shape: (32, 128)
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize the linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            bias: If True, include a bias term
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Xavier initialization for weights
        # This keeps variance stable across layers
        scale = _backend.xp.sqrt(2.0 / (in_features + out_features))
        self.weight = Tensor(
            _backend.xp.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        
        # Initialize bias to zeros
        if bias:
            self.bias = Tensor(
                _backend.xp.zeros(out_features),
                requires_grad=True
            )
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: y = x @ W + b
        
        Args:
            x: Input tensor of shape (..., in_features)
        
        Returns:
            Output tensor of shape (..., out_features)
        """
        # Matrix multiplication with weight
        # x: (..., in_features)
        # weight: (in_features, out_features)
        # output: (..., out_features)
        
        # Handle batched inputs
        original_shape = x.shape[:-1]
        x_flat = x.data.reshape(-1, self.in_features)
        
        # Forward: y = x @ W
        output = x_flat @ self.weight.data
        
        # Add bias if present
        if self.use_bias:
            output = output + self.bias.data
        
        # Reshape to original batch dimensions
        output = output.reshape(*original_shape, self.out_features)
        
        out = Tensor(
            output,
            requires_grad=x.requires_grad or self.weight.requires_grad,
            _children=(x, self.weight) if not self.use_bias else (x, self.weight, self.bias),
            _op='linear'
        )
        
        # Store for backward
        self._x_flat = x_flat
        self._original_shape = original_shape
        
        def _backward():
            out_grad_flat = out.grad.reshape(-1, self.out_features)
            
            if x.requires_grad:
                # Gradient w.r.t. input: dL/dx = dL/dy @ W.T
                grad = out_grad_flat @ self.weight.data.T
                grad = grad.reshape(*original_shape, self.in_features)
                x.grad = x.grad + grad if x.grad is not None else grad
            
            if self.weight.requires_grad:
                # Gradient w.r.t. weight: dL/dW = x.T @ dL/dy
                grad = x_flat.T @ out_grad_flat
                self.weight.grad = self.weight.grad + grad if self.weight.grad is not None else grad
            
            if self.use_bias and self.bias.requires_grad:
                # Gradient w.r.t. bias: sum over batch dimension
                grad = _backend.xp.sum(out_grad_flat, axis=0)
                self.bias.grad = self.bias.grad + grad if self.bias.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self):
        """Return trainable parameters."""
        if self.use_bias:
            return [self.weight, self.bias]
        return [self.weight]


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Linear Layer Demo")
    print("=" * 50)
    
    # Create a linear layer
    in_features = 64
    out_features = 32
    layer = Linear(in_features, out_features)
    
    print(f"Weight shape: {layer.weight.shape}")
    print(f"Bias shape: {layer.bias.shape}")
    
    # Create input
    batch_size = 8
    x = Tensor(np.random.randn(batch_size, in_features), requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = layer(x)
    print(f"Output shape: {y.shape}")
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    print(f"\nGradients:")
    print(f"x.grad shape: {x.grad.shape}")
    print(f"weight.grad shape: {layer.weight.grad.shape}")
    print(f"bias.grad shape: {layer.bias.grad.shape}")
    
    # Test with 3D input (batch, seq, features)
    print("\n--- Testing 3D input ---")
    x_3d = Tensor(np.random.randn(4, 10, in_features), requires_grad=True)
    y_3d = layer(x_3d)
    print(f"3D Input shape: {x_3d.shape}")
    print(f"3D Output shape: {y_3d.shape}")
    
    print("\n✓ Linear layer working!")
