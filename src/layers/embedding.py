"""
Embedding Layers - Converting tokens to vectors.

Before a neural network can process text (or code), we need to convert
discrete tokens (like words or characters) into continuous vectors.

Learning Goals:
- Understand token embeddings (lookup table)
- Learn positional encodings (how the model knows word order)
- See how embeddings are trained
"""

import numpy as np
from ..core.tensor import Tensor
from typing import Optional


class Embedding:
    """
    Token Embedding Layer - A learnable lookup table.
    
    Each token ID maps to a dense vector. These vectors are learned
    during training and capture semantic meaning.
    
    Example:
        vocab_size = 1000  (1000 unique tokens)
        embed_dim = 256    (each token becomes a 256-dimensional vector)
        
        embedding = Embedding(1000, 256)
        token_ids = [5, 42, 100]  # Input tokens
        vectors = embedding(token_ids)  # Shape: (3, 256)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Initialize the embedding layer.
        
        Args:
            vocab_size: Number of unique tokens in vocabulary
            embed_dim: Dimension of embedding vectors
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embedding matrix with small random values
        # Each row is the embedding for one token
        scale = np.sqrt(1.0 / embed_dim)
        self.weight = Tensor(
            np.random.randn(vocab_size, embed_dim) * scale,
            requires_grad=True
        )
    
    def forward(self, token_ids: np.ndarray) -> Tensor:
        """
        Look up embeddings for token IDs.
        
        Args:
            token_ids: Array of token IDs, shape (batch, seq_len)
        
        Returns:
            Tensor of embeddings, shape (batch, seq_len, embed_dim)
        """
        # Store for backward pass
        self._token_ids = token_ids
        original_shape = token_ids.shape
        
        # Flatten for indexing
        flat_ids = token_ids.flatten()
        
        # Look up embeddings (simple indexing)
        embeddings = self.weight.data[flat_ids]
        
        # Reshape to (batch, seq_len, embed_dim)
        output_shape = (*original_shape, self.embed_dim)
        result = embeddings.reshape(output_shape)
        
        out = Tensor(
            result,
            requires_grad=self.weight.requires_grad,
            _children=(self.weight,),
            _op='embedding'
        )
        
        def _backward():
            if self.weight.requires_grad:
                # Gradient flows back to the looked-up embeddings
                grad = np.zeros_like(self.weight.data)
                # Reshape gradient to match flat indices
                out_grad_flat = out.grad.reshape(-1, self.embed_dim)
                
                # Accumulate gradients for each token
                np.add.at(grad, flat_ids, out_grad_flat)
                
                if self.weight.grad is not None:
                    self.weight.grad = self.weight.grad + grad
                else:
                    self.weight.grad = grad
        
        out._backward = _backward
        return out
    
    def __call__(self, token_ids: np.ndarray) -> Tensor:
        return self.forward(token_ids)
    
    def parameters(self):
        """Return trainable parameters."""
        return [self.weight]


class PositionalEncoding:
    """
    Positional Encoding - Telling the model about word order.
    
    Transformers process all positions in parallel, so they don't
    naturally know the order of words. Positional encoding adds
    position information to the embeddings.
    
    We use sinusoidal encodings (from "Attention is All You Need"):
    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    
    The beauty: similar positions have similar encodings,
    and the model can learn to attend to relative positions.
    """
    
    def __init__(self, max_seq_len: int, embed_dim: int):
        """
        Initialize positional encoding.
        
        Args:
            max_seq_len: Maximum sequence length
            embed_dim: Dimension of embeddings
        """
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Precompute positional encodings
        self.encoding = self._create_encoding(max_seq_len, embed_dim)
    
    def _create_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        position = np.arange(max_len)[:, np.newaxis]  # (max_len, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = np.cos(position * div_term)  # Odd indices: cos
        
        return pe.astype(np.float32)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
        
        Returns:
            Tensor with added positional encoding
        """
        seq_len = x.shape[1]
        
        # Get positional encoding for this sequence length
        pos_enc = self.encoding[:seq_len]
        
        # Add to input (broadcasting over batch dimension)
        out = Tensor(
            x.data + pos_enc,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op='pos_enc'
        )
        
        def _backward():
            if x.requires_grad:
                # Gradient flows through unchanged
                x.grad = x.grad + out.grad if x.grad is not None else out.grad.copy()
        
        out._backward = _backward
        return out
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Embedding Layer Demo")
    print("=" * 50)
    
    # Create embedding layer
    vocab_size = 100
    embed_dim = 32
    embedding = Embedding(vocab_size, embed_dim)
    
    # Sample token IDs (batch_size=2, seq_len=4)
    token_ids = np.array([
        [1, 5, 10, 2],   # First sequence
        [3, 7, 15, 4]    # Second sequence
    ])
    
    print(f"Token IDs shape: {token_ids.shape}")
    
    # Get embeddings
    embeddings = embedding(token_ids)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Expected: (2, 4, {embed_dim})")
    
    # Add positional encoding
    pos_enc = PositionalEncoding(max_seq_len=100, embed_dim=embed_dim)
    embeddings_with_pos = pos_enc(embeddings)
    
    print(f"\nWith positional encoding shape: {embeddings_with_pos.shape}")
    
    # Test backward pass
    loss = embeddings_with_pos.sum()
    loss.backward()
    print(f"\nEmbedding gradient shape: {embedding.weight.grad.shape}")
    
    print("\n✓ Embedding layer working!")
