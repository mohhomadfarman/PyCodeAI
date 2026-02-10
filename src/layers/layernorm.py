"""
Layer Normalization - Stabilizing neural network training.

Deep networks are hard to train because activations can explode or vanish.
Layer Normalization stabilizes training by normalizing activations.

For each sample, it:
1. Computes mean and variance across features
2. Normalizes to zero mean and unit variance
3. Scales and shifts with learnable parameters (gamma, beta)

Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta

Learning Goals:
- Understand why normalization helps training
- Learn the difference from batch normalization
- See how normalization affects gradients
"""

import numpy as np
from ..core import backend as _backend
from ..core.tensor import Tensor
from typing import Tuple


class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes across the last dimension (features).
    Used extensively in transformers for training stability.
    
    Example:
        ln = LayerNorm(256)  # Normalize 256 features
        x = Tensor(np.randn(32, 10, 256))  # (batch, seq, features)
        y = ln(x)  # Normalized output
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        """
        Initialize layer normalization.
        
        Args:
            normalized_shape: Size of the last dimension to normalize
            eps: Small constant for numerical stability
        """
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable scale (gamma) and shift (beta)
        # Initialized to 1 and 0 respectively
        self.gamma = Tensor(_backend.xp.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(_backend.xp.zeros(normalized_shape), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., normalized_shape)
        
        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance along last dimension
        mean = _backend.xp.mean(x.data, axis=-1, keepdims=True)
        var = _backend.xp.var(x.data, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x.data - mean) / _backend.xp.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma.data * x_norm + self.beta.data
        
        out = Tensor(
            output,
            requires_grad=x.requires_grad or self.gamma.requires_grad,
            _children=(x, self.gamma, self.beta),
            _op='layernorm'
        )

        # Local variable for backward closure (avoids storing on self)
        std = _backend.xp.sqrt(var + self.eps)

        def _backward():
            n = self.normalized_shape

            if self.gamma.requires_grad:
                # Gradient w.r.t gamma: sum of x_norm * grad
                grad_gamma = _backend.xp.sum(x_norm * out.grad, axis=tuple(range(x.data.ndim - 1)))
                self.gamma.grad = self.gamma.grad + grad_gamma if self.gamma.grad is not None else grad_gamma

            if self.beta.requires_grad:
                # Gradient w.r.t beta: sum of grad
                grad_beta = _backend.xp.sum(out.grad, axis=tuple(range(x.data.ndim - 1)))
                self.beta.grad = self.beta.grad + grad_beta if self.beta.grad is not None else grad_beta

            if x.requires_grad:
                # Gradient w.r.t input (more complex due to normalization)
                # dx_norm = gamma * grad
                dx_norm = self.gamma.data * out.grad

                # Backprop through normalization
                dvar = _backend.xp.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5, axis=-1, keepdims=True)
                dmean = _backend.xp.sum(dx_norm * -1 / std, axis=-1, keepdims=True)
                dmean += dvar * _backend.xp.sum(-2 * (x.data - mean), axis=-1, keepdims=True) / n

                grad = dx_norm / std + dvar * 2 * (x.data - mean) / n + dmean / n
                x.grad = x.grad + grad if x.grad is not None else grad

        out._backward = _backward
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self):
        """Return trainable parameters."""
        return [self.gamma, self.beta]


class RMSNorm:
    """
    Root Mean Square Layer Normalization.
    
    A simpler alternative to LayerNorm used in LLaMA and other models.
    Doesn't center the data, only scales by RMS.
    
    Formula: y = x / sqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = Tensor(_backend.xp.ones(normalized_shape), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Compute RMS
        rms = _backend.xp.sqrt(_backend.xp.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps)
        x_norm = x.data / rms
        output = self.gamma.data * x_norm
        
        out = Tensor(
            output,
            requires_grad=x.requires_grad or self.gamma.requires_grad,
            _children=(x, self.gamma),
            _op='rmsnorm'
        )
        
        def _backward():
            if self.gamma.requires_grad:
                grad_gamma = _backend.xp.sum(x_norm * out.grad, axis=tuple(range(x.data.ndim - 1)))
                self.gamma.grad = self.gamma.grad + grad_gamma if self.gamma.grad is not None else grad_gamma

            if x.requires_grad:
                n = self.normalized_shape
                dx_norm = self.gamma.data * out.grad

                # Simplified RMSNorm gradient
                grad = dx_norm / rms
                grad -= x_norm * _backend.xp.mean(dx_norm * x_norm, axis=-1, keepdims=True)
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self):
        return [self.gamma]


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Layer Normalization Demo")
    print("=" * 50)
    
    # Create layer norm
    features = 64
    ln = LayerNorm(features)
    
    # Create input with mean ≠ 0 and std ≠ 1
    batch_size = 4
    seq_len = 10
    x = Tensor(np.random.randn(batch_size, seq_len, features) * 5 + 3, requires_grad=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Input mean: {np.mean(x.data):.4f}")
    print(f"Input std: {np.std(x.data):.4f}")
    
    # Apply layer norm
    y = ln(x)
    
    print(f"\nOutput shape: {y.shape}")
    print(f"Output mean per sample: ~0 (normalized)")
    print(f"Output std per sample: ~1 (normalized)")
    
    # Verify normalization (last axis should be normalized)
    sample_mean = np.mean(y.data[0, 0, :])
    sample_std = np.std(y.data[0, 0, :])
    print(f"\nFirst sample, first position:")
    print(f"  Mean: {sample_mean:.6f} (should be ~0)")
    print(f"  Std: {sample_std:.6f} (should be ~1)")
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    print(f"\nGradients computed:")
    print(f"  x.grad shape: {x.grad.shape}")
    print(f"  gamma.grad shape: {ln.gamma.grad.shape}")
    print(f"  beta.grad shape: {ln.beta.grad.shape}")
    
    print("\n✓ Layer normalization working!")
