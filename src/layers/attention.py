"""
Multi-Head Self-Attention - The heart of transformers.

This is THE mechanism that makes transformers powerful!
It allows the model to look at different parts of the input
when processing each position.

"Attention is All You Need" (Vaswani et al., 2017)

The key idea:
- Query (Q): What am I looking for?
- Key (K): What do I have?
- Value (V): What information do I provide?

Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

Learning Goals:
- Understand how attention allows global information flow
- Learn scaled dot-product attention
- See how multiple heads capture different patterns
"""

import numpy as np
from ..core import backend as _backend
from ..core import tensor as _tensor_mod
from ..core.tensor import Tensor
from ..core.activations import softmax
from .linear import Linear
from typing import Optional, Tuple


class MultiHeadAttention:
    """
    Multi-Head Self-Attention Layer.

    Uses fused QKV projection for efficiency, cached causal mask,
    and implements full backward pass for correct gradient flow.

    Example:
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        x = Tensor(np.randn(32, 100, 256))  # (batch, seq, embed)
        y = attn(x)  # Same shape, but with attention-weighted features
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, max_seq_len: int = 1024):
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)

        # Fused QKV projection (single matmul instead of 3 separate)
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim)

        # Precompute causal mask (avoids reallocation every forward call)
        self._causal_mask = _backend.xp.triu(
            _backend.xp.ones((max_seq_len, max_seq_len), dtype=_backend.xp.float32), k=1
        ) * (-1e9)

    def forward(
        self,
        x: Tensor,
        mask: Optional[np.ndarray] = None,
        is_causal: bool = True
    ) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Fused QKV projection (one matmul instead of three)
        qkv = self.qkv_proj(x)  # (batch, seq, 3 * embed_dim)

        # Split and reshape: (batch, seq, 3*embed) -> (3, batch, heads, seq, head_dim)
        qkv_data = qkv.data.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_data = _backend.xp.transpose(qkv_data, (2, 0, 3, 1, 4))
        q_heads, k_heads, v_heads = qkv_data[0], qkv_data[1], qkv_data[2]

        # Attention scores: Q @ K.T / sqrt(d_k)  ->  (batch, heads, seq, seq)
        attn_scores = _backend.xp.matmul(q_heads, _backend.xp.transpose(k_heads, (0, 1, 3, 2)))
        attn_scores = attn_scores * self.scale

        # Apply causal mask (cached, no allocation)
        if is_causal:
            attn_scores = attn_scores + self._causal_mask[:seq_len, :seq_len]

        if mask is not None:
            attn_scores = attn_scores + mask * (-1e9)

        # Softmax -> attention weights
        attn_weights = self._softmax(attn_scores)

        # Apply attention to values: (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
        attn_output = _backend.xp.matmul(attn_weights, v_heads)

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = _backend.xp.transpose(attn_output, (0, 2, 1, 3))
        attn_output_flat = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # Output projection (manual for backward control)
        x_out_flat = attn_output_flat.reshape(-1, self.embed_dim)
        out_data = x_out_flat @ self.out_proj.weight.data
        if self.out_proj.use_bias:
            out_data = out_data + self.out_proj.bias.data
        out_data = out_data.reshape(batch_size, seq_len, self.embed_dim)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op='attention'
        )

        # Skip backward setup in inference mode (no_grad)
        if not _tensor_mod._no_grad:
            def _backward():
                if not x.requires_grad and not self.qkv_proj.weight.requires_grad:
                    return

                d_out = out.grad  # (batch, seq, embed_dim)
                d_out_flat = d_out.reshape(-1, self.embed_dim)

                # -- Backprop through output projection --
                if self.out_proj.weight.requires_grad:
                    w_grad = x_out_flat.T @ d_out_flat
                    self.out_proj.weight.grad = (
                        self.out_proj.weight.grad + w_grad
                        if self.out_proj.weight.grad is not None else w_grad)
                if self.out_proj.use_bias and self.out_proj.bias.requires_grad:
                    b_grad = _backend.xp.sum(d_out_flat, axis=0)
                    self.out_proj.bias.grad = (
                        self.out_proj.bias.grad + b_grad
                        if self.out_proj.bias.grad is not None else b_grad)

                d_attn_flat = d_out_flat @ self.out_proj.weight.data.T
                d_attn = d_attn_flat.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
                d_attn_heads = _backend.xp.transpose(d_attn, (0, 2, 1, 3))

                # -- Backprop through attn_weights @ V --
                d_attn_weights = _backend.xp.matmul(d_attn_heads, _backend.xp.transpose(v_heads, (0, 1, 3, 2)))
                d_v = _backend.xp.matmul(_backend.xp.transpose(attn_weights, (0, 1, 3, 2)), d_attn_heads)

                # -- Backprop through softmax --
                sum_term = _backend.xp.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)
                d_scores = attn_weights * (d_attn_weights - sum_term)
                d_scores = d_scores * self.scale

                # -- Backprop through Q @ K.T --
                d_q = _backend.xp.matmul(d_scores, k_heads)
                d_k = _backend.xp.matmul(_backend.xp.transpose(d_scores, (0, 1, 3, 2)), q_heads)

                # Reshape back to (batch, seq, embed_dim)
                d_q = _backend.xp.transpose(d_q, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
                d_k = _backend.xp.transpose(d_k, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
                d_v = _backend.xp.transpose(d_v, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)

                # -- Backprop through fused QKV projection --
                d_qkv = _backend.xp.concatenate([d_q, d_k, d_v], axis=-1)
                d_qkv_flat = d_qkv.reshape(-1, 3 * self.embed_dim)
                x_flat = x.data.reshape(-1, self.embed_dim)

                if self.qkv_proj.weight.requires_grad:
                    qkv_w_grad = x_flat.T @ d_qkv_flat
                    self.qkv_proj.weight.grad = (
                        self.qkv_proj.weight.grad + qkv_w_grad
                        if self.qkv_proj.weight.grad is not None else qkv_w_grad)
                if self.qkv_proj.use_bias and self.qkv_proj.bias.requires_grad:
                    qkv_b_grad = _backend.xp.sum(d_qkv_flat, axis=0)
                    self.qkv_proj.bias.grad = (
                        self.qkv_proj.bias.grad + qkv_b_grad
                        if self.qkv_proj.bias.grad is not None else qkv_b_grad)

                # Gradient to input x
                if x.requires_grad:
                    x_grad = d_qkv_flat @ self.qkv_proj.weight.data.T
                    x_grad = x_grad.reshape(batch_size, seq_len, self.embed_dim)
                    x.grad = x.grad + x_grad if x.grad is not None else x_grad

            out._backward = _backward
        return out

    def _softmax(self, x):
        """Numerically stable softmax."""
        x_max = _backend.xp.max(x, axis=-1, keepdims=True)
        exp_x = _backend.xp.exp(x - x_max)
        return exp_x / _backend.xp.sum(exp_x, axis=-1, keepdims=True)

    def __call__(self, x, mask=None, is_causal=True):
        return self.forward(x, mask, is_causal)

    def parameters(self):
        """Return trainable parameters."""
        return self.qkv_proj.parameters() + self.out_proj.parameters()


class CausalSelfAttention:
    """
    Causal Self-Attention - Simplified version for GPT-style models.
    
    This is the attention used in decoder-only transformers like GPT.
    Each position can only attend to previous positions (causal/autoregressive).
    """
    
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 1024):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection (more efficient)
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
        # Precompute causal mask
        mask = _backend.xp.triu(_backend.xp.ones((max_seq_len, max_seq_len)), k=1)
        self.causal_mask = mask.astype(bool)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Combined QKV projection
        qkv = self.qkv_proj(x)
        qkv_data = qkv.data.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_data = _backend.xp.transpose(qkv_data, (2, 0, 3, 1, 4))  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv_data[0], qkv_data[1], qkv_data[2]
        
        # Attention scores
        attn = _backend.xp.matmul(q, _backend.xp.transpose(k, (0, 1, 3, 2))) * self.scale
        
        # Apply causal mask
        attn[:, :, self.causal_mask[:seq_len, :seq_len]] = float('-inf')
        
        # Softmax
        attn = self._softmax(attn)
        
        # Apply to values
        out = _backend.xp.matmul(attn, v)
        out = _backend.xp.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        out_tensor = Tensor(out, requires_grad=True)
        return self.out_proj(out_tensor)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = _backend.xp.max(x, axis=-1, keepdims=True)
        exp_x = _backend.xp.exp(x - x_max)
        return exp_x / _backend.xp.sum(exp_x, axis=-1, keepdims=True)
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self):
        return self.qkv_proj.parameters() + self.out_proj.parameters()


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Multi-Head Attention Demo")
    print("=" * 50)
    
    # Create attention layer
    embed_dim = 64
    num_heads = 4
    attn = MultiHeadAttention(embed_dim, num_heads)
    
    print(f"Embed dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Head dim: {attn.head_dim}")
    
    # Create input
    batch_size = 2
    seq_len = 10
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = attn(x, is_causal=True)
    print(f"Output shape: {y.shape}")
    
    # Check attention weights
    print(f"\nAttention weights shape: {attn._attn_weights.shape}")
    print(f"Expected: (batch={batch_size}, heads={num_heads}, seq={seq_len}, seq={seq_len})")
    
    # Verify causal masking (upper triangle should be ~0 after softmax)
    weights_sample = attn._attn_weights[0, 0]  # First batch, first head
    print(f"\nFirst head attention pattern (rows should attend only to earlier positions):")
    print(f"Lower triangle sum: {np.sum(np.tril(weights_sample)):.4f}")
    print(f"Upper triangle sum: {np.sum(np.triu(weights_sample, k=1)):.4f} (should be ~0)")
    
    # Count parameters
    num_params = sum(p.data.size for p in attn.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    print("\n✓ Multi-head attention working!")
