"""
Transformer Block - The building block of modern language models.

A transformer block consists of:
1. Multi-Head Self-Attention (with residual connection)
2. Layer Normalization
3. Feed-Forward Network (with residual connection)
4. Layer Normalization

The key insight: residual connections allow gradients to flow
directly through the network, enabling very deep models.

Learning Goals:
- Understand how attention and feedforward work together
- Learn why residual connections are crucial
- See the complete transformer block architecture
"""

import numpy as np
from ..core.tensor import Tensor
from ..core.activations import gelu
from ..layers.attention import MultiHeadAttention
from ..layers.layernorm import LayerNorm
from ..layers.linear import Linear
from typing import Optional


class FeedForward:
    """
    Feed-Forward Network (FFN) - Position-wise MLP.
    
    After attention mixes information between positions,
    the FFN processes each position independently.
    
    Architecture:
        x -> Linear(d, 4d) -> GELU -> Linear(4d, d) -> out
    
    The expansion to 4x dimension allows for more expressive transformations.
    """
    
    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Input/output dimension
            expansion_factor: How much to expand the hidden dimension
        """
        hidden_dim = embed_dim * expansion_factor
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor of shape (..., embed_dim)
        
        Returns:
            Output tensor of same shape
        """
        # First linear: expand to hidden_dim
        h = self.fc1(x)
        
        # GELU activation
        h = gelu(h)
        
        # Second linear: project back to embed_dim
        out = self.fc2(h)
        
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    def parameters(self):
        """Return trainable parameters."""
        return self.fc1.parameters() + self.fc2.parameters()


class TransformerBlock:
    """
    Transformer Block - One layer of the transformer.
    
    This is the core building block that gets stacked N times.
    Each block allows the model to refine its representations.
    
    Architecture (Pre-LN, as used in GPT-2+):
        x -> LayerNorm -> Attention -> +x (residual)
          -> LayerNorm -> FFN -> +x (residual)
    
    The residual connections are crucial:
    - Allow gradients to flow directly backward
    - Enable training of very deep networks
    - Let the block learn "refinements" to the input
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int,
        expansion_factor: int = 4
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            expansion_factor: FFN expansion factor
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Attention sublayer
        self.ln1 = LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # FFN sublayer
        self.ln2 = LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, expansion_factor)
    
    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Apply transformer block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Output tensor of same shape
        """
        # Attention sublayer with residual connection
        # Pre-LN: normalize before attention
        h = self.ln1(x)
        attn_out = self.attention(h, mask=mask, is_causal=True)
        
        # Residual connection: x + attention output
        x_data = x.data + attn_out.data
        x = Tensor(x_data, requires_grad=True)
        
        # FFN sublayer with residual connection
        h = self.ln2(x)
        ffn_out = self.ffn(h)
        
        # Residual connection
        out_data = x.data + ffn_out.data
        out = Tensor(out_data, requires_grad=True)
        
        return out
    
    def __call__(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        return self.forward(x, mask)
    
    def parameters(self):
        """Return trainable parameters."""
        params = []
        params.extend(self.ln1.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.ln2.parameters())
        params.extend(self.ffn.parameters())
        return params


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Transformer Block Demo")
    print("=" * 50)
    
    # Create transformer block
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim, num_heads)
    
    print(f"Embed dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"FFN hidden dim: {embed_dim * 4}")
    
    # Create input
    batch_size = 2
    seq_len = 10
    x = Tensor(np.random.randn(batch_size, seq_len, embed_dim), requires_grad=True)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    y = block(x)
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    num_params = sum(p.data.size for p in block.parameters())
    print(f"\nTotal parameters: {num_params:,}")
    
    # Break down parameters
    attn_params = sum(p.data.size for p in block.attention.parameters())
    ffn_params = sum(p.data.size for p in block.ffn.parameters())
    norm_params = sum(p.data.size for p in block.ln1.parameters() + block.ln2.parameters())
    
    print(f"  Attention: {attn_params:,}")
    print(f"  FFN: {ffn_params:,}")
    print(f"  LayerNorm: {norm_params:,}")
    
    print("\n✓ Transformer block working!")
