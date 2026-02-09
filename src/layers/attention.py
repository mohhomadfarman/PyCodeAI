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
from ..core.tensor import Tensor
from ..core.activations import softmax
from .linear import Linear
from typing import Optional, Tuple


class MultiHeadAttention:
    """
    Multi-Head Self-Attention Layer.
    
    Multiple attention heads allow the model to jointly attend
    to information from different representation subspaces.
    
    Example:
        attn = MultiHeadAttention(embed_dim=256, num_heads=8)
        x = Tensor(np.randn(32, 100, 256))  # (batch, seq, embed)
        y = attn(x)  # Same shape, but with attention-weighted features
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Total dimension of the model
            num_heads: Number of attention heads
            dropout: Dropout probability (not implemented yet)
        """
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Linear projections for Q, K, V
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = Linear(embed_dim, embed_dim)
    
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[np.ndarray] = None,
        is_causal: bool = True
    ) -> Tensor:
        """
        Apply multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
            is_causal: If True, apply causal (autoregressive) mask
        
        Returns:
            Output tensor of shape (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: (batch, seq, num_heads, head_dim)
        # Then transpose to: (batch, num_heads, seq, head_dim)
        q_heads = q.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q_heads = np.transpose(q_heads, (0, 2, 1, 3))
        
        k_heads = k.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k_heads = np.transpose(k_heads, (0, 2, 1, 3))
        
        v_heads = v.data.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v_heads = np.transpose(v_heads, (0, 2, 1, 3))
        
        # Compute attention scores: Q @ K.T / sqrt(d_k)
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq)
        # = (batch, heads, seq, seq)
        attn_scores = np.matmul(q_heads, np.transpose(k_heads, (0, 1, 3, 2)))
        attn_scores = attn_scores * self.scale
        
        # Apply causal mask (prevent attending to future positions)
        if is_causal:
            causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            attn_scores = attn_scores + causal_mask * (-1e9)
        
        # Apply optional additional mask
        if mask is not None:
            attn_scores = attn_scores + mask * (-1e9)
        
        # Softmax to get attention weights
        attn_weights = self._softmax(attn_scores)
        
        # Apply attention to values
        # (batch, heads, seq, seq) @ (batch, heads, seq, head_dim)
        # = (batch, heads, seq, head_dim)
        attn_output = np.matmul(attn_weights, v_heads)
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, embed_dim)
        attn_output = np.transpose(attn_output, (0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        
        # Store for backward
        self._q_heads = q_heads
        self._k_heads = k_heads
        self._v_heads = v_heads
        self._attn_weights = attn_weights
        
        # Final output projection
        attn_tensor = Tensor(attn_output, requires_grad=True)
        output = self.out_proj(attn_tensor)
        
        # Create output tensor with backward
        out = Tensor(
            output.data,
            requires_grad=x.requires_grad,
            _children=(x, q, k, v),
            _op='attention'
        )
        
        def _backward():
            if x.requires_grad:
                # Simplified backward (full implementation is complex)
                # This is an approximation for learning purposes
                
                # Gradient through output projection
                out_grad = out.grad
                
                # Backprop through attention (simplified)
                # In practice, you'd need to fully implement the chain rule
                # through softmax, matmul, and projections
                
                # For now, we'll backprop through the linear projections
                attn_tensor.grad = out_grad
                self.out_proj.forward(attn_tensor)._backward()
                
                # Approximate gradient to input
                # This flows gradients back through all projections
                grad = np.zeros_like(x.data)
                grad += self.q_proj.weight.data @ out_grad.reshape(-1, self.embed_dim).T
                grad = grad.T.reshape(x.shape)
                
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def __call__(
        self, 
        x: Tensor, 
        mask: Optional[np.ndarray] = None,
        is_causal: bool = True
    ) -> Tensor:
        return self.forward(x, mask, is_causal)
    
    def parameters(self):
        """Return trainable parameters."""
        params = []
        params.extend(self.q_proj.parameters())
        params.extend(self.k_proj.parameters())
        params.extend(self.v_proj.parameters())
        params.extend(self.out_proj.parameters())
        return params


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
        mask = np.triu(np.ones((max_seq_len, max_seq_len)), k=1)
        self.causal_mask = mask.astype(bool)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Combined QKV projection
        qkv = self.qkv_proj(x)
        qkv_data = qkv.data.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv_data = np.transpose(qkv_data, (2, 0, 3, 1, 4))  # (3, batch, heads, seq, head_dim)
        q, k, v = qkv_data[0], qkv_data[1], qkv_data[2]
        
        # Attention scores
        attn = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) * self.scale
        
        # Apply causal mask
        attn[:, :, self.causal_mask[:seq_len, :seq_len]] = -np.inf
        
        # Softmax
        attn = self._softmax(attn)
        
        # Apply to values
        out = np.matmul(attn, v)
        out = np.transpose(out, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        
        # Output projection
        out_tensor = Tensor(out, requires_grad=True)
        return self.out_proj(out_tensor)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
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
