"""
GPT Model - Complete GPT-style decoder-only transformer.

GPT (Generative Pre-trained Transformer) architecture:
1. Token Embedding + Positional Encoding
2. N x Transformer Blocks (attention + FFN)
3. Final Layer Norm
4. Output Projection (to vocabulary)

This generates text by predicting the next token.

Learning Goals:
- Understand the complete GPT architecture
- See how all components fit together
- Learn about next-token prediction
"""

import numpy as np
from ..core.tensor import Tensor
from ..core.activations import softmax, log_softmax
from ..layers.embedding import Embedding, PositionalEncoding
from ..layers.layernorm import LayerNorm
from ..layers.linear import Linear
from .transformer import TransformerBlock
from typing import Optional, Dict, Any


class GPTConfig:
    """Configuration for GPT model."""
    
    def __init__(
        self,
        vocab_size: int = 1000,
        max_seq_len: int = 256,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        expansion_factor: int = 4
    ):
        """
        Initialize GPT configuration.
        
        Args:
            vocab_size: Size of token vocabulary
            max_seq_len: Maximum sequence length
            embed_dim: Embedding dimension (d_model)
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            expansion_factor: FFN expansion factor
        """
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.expansion_factor = expansion_factor
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "expansion_factor": self.expansion_factor
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GPTConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def __repr__(self):
        return (
            f"GPTConfig(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  max_seq_len={self.max_seq_len},\n"
            f"  embed_dim={self.embed_dim},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_layers={self.num_layers},\n"
            f"  expansion_factor={self.expansion_factor}\n"
            f")"
        )


class GPT:
    """
    GPT (Generative Pre-trained Transformer) Model.
    
    A decoder-only transformer for next-token prediction.
    This is the architecture used by ChatGPT, GPT-4, etc.
    
    Example:
        config = GPTConfig(vocab_size=5000, embed_dim=256, num_layers=6)
        model = GPT(config)
        
        tokens = np.array([[1, 42, 100, 5]])  # (batch=1, seq=4)
        logits = model(tokens)  # (batch=1, seq=4, vocab_size=5000)
    """
    
    def __init__(self, config: GPTConfig):
        """
        Initialize GPT model.
        
        Args:
            config: Model configuration
        """
        self.config = config
        
        # Token embedding
        self.token_embedding = Embedding(config.vocab_size, config.embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.max_seq_len, config.embed_dim)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(config.embed_dim, config.num_heads, config.expansion_factor)
            for _ in range(config.num_layers)
        ]
        
        # Final layer norm
        self.ln_final = LayerNorm(config.embed_dim)
        
        # Output projection to vocabulary
        self.lm_head = Linear(config.embed_dim, config.vocab_size, bias=False)
    
    def forward(self, token_ids: np.ndarray) -> Tensor:
        """
        Forward pass through GPT.
        
        Args:
            token_ids: Input token IDs, shape (batch, seq_len)
        
        Returns:
            Logits over vocabulary, shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = token_ids.shape
        
        # Check sequence length
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max {self.config.max_seq_len}")
        
        # Token embeddings: (batch, seq, embed_dim)
        x = self.token_embedding(token_ids)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)
        
        return logits
    
    def __call__(self, token_ids: np.ndarray) -> Tensor:
        return self.forward(token_ids)
    
    def generate(
        self, 
        prompt_tokens: np.ndarray, 
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate text given a prompt.
        
        Args:
            prompt_tokens: Starting token IDs, shape (1, prompt_len)
            max_new_tokens: How many new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
        
        Returns:
            Generated token IDs including prompt
        """
        tokens = prompt_tokens.copy()
        
        for _ in range(max_new_tokens):
            # Get only the last max_seq_len tokens
            tokens_cond = tokens[:, -self.config.max_seq_len:]
            
            # Forward pass
            logits = self.forward(tokens_cond)
            
            # Get logits for last position
            logits_last = logits.data[:, -1, :]  # (batch, vocab_size)
            
            # Apply temperature
            if temperature != 1.0:
                logits_last = logits_last / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                # Set all logits except top k to -infinity
                top_k_indices = np.argsort(logits_last, axis=-1)[:, -top_k:]
                mask = np.ones_like(logits_last) * (-1e10)
                np.put_along_axis(mask, top_k_indices, 0, axis=-1)
                logits_last = logits_last + mask
            
            # Convert to probabilities
            probs = self._softmax(logits_last)
            
            # Sample next token
            next_token = self._sample(probs)
            
            # Append to sequence
            tokens = np.concatenate([tokens, next_token], axis=1)
        
        return tokens
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _sample(self, probs: np.ndarray) -> np.ndarray:
        """Sample from probability distribution."""
        batch_size = probs.shape[0]
        next_tokens = []
        for b in range(batch_size):
            next_token = np.random.choice(len(probs[b]), p=probs[b])
            next_tokens.append(next_token)
        return np.array(next_tokens).reshape(batch_size, 1)
    
    def parameters(self):
        """Return all trainable parameters."""
        params = []
        
        # Embeddings
        params.extend(self.token_embedding.parameters())
        
        # Transformer blocks
        for block in self.blocks:
            params.extend(block.parameters())
        
        # Final layers
        params.extend(self.ln_final.parameters())
        params.extend(self.lm_head.parameters())
        
        return params
    
    def num_parameters(self) -> int:
        """Count total number of parameters."""
        return sum(p.data.size for p in self.parameters())
    
    def save(self, path: str):
        """Save model weights and configuration to file."""
        # Save config alongside weights
        import json
        import os
        
        # Determine config path (replace extension with .json)
        base_path = os.path.splitext(path)[0]
        config_path = f"{base_path}_config.json"
        
        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
        # Save weights
        weights = {}
        for i, param in enumerate(self.parameters()):
            weights[f"param_{i}"] = param.data
        np.savez(path, **weights)
        print(f"Model saved to {path}")
        print(f"Config saved to {config_path}")
    
    def load(self, path: str):
        """Load model weights from file."""
        import os
        
        # Check if config exists but we don't load it here
        # Loading config is done before creating the model
        
        weights = np.load(path)
        params = self.parameters()
        
        # Check if parameter count matches
        if len(weights) != len(params):
            print(f"WARNING: Model has {len(params)} parameters but file has {len(weights)}")
            print("Architectures might not match!")
            
        for i, param in enumerate(params):
            if f"param_{i}" in weights:
                data = weights[f"param_{i}"]
                if param.data.shape != data.shape:
                   raise ValueError(f"Shape mismatch for param_{i}: expected {param.data.shape}, got {data.shape}")
                param.data = data
            else:
                print(f"WARNING: param_{i} not found in weights file")
        
        print(f"Model loaded from {path}")
    
    @classmethod
    def load_from_dir(cls, path: str) -> 'GPT':
        """Load model from directory (helper to load config then weights)."""
        import json
        import os
        
        # Determine config path
        base_path = os.path.splitext(path)[0]
        config_path = f"{base_path}_config.json"
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        # Load config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config = GPTConfig.from_dict(config_dict)
        model = cls(config)
        model.load(path)
        
        return model


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("GPT Model Demo")
    print("=" * 50)
    
    # Create a small GPT model
    config = GPTConfig(
        vocab_size=100,     # Small vocabulary
        max_seq_len=64,     # Short sequences
        embed_dim=64,       # Small embedding
        num_heads=4,        # 4 attention heads
        num_layers=2        # 2 transformer blocks
    )
    
    print(config)
    
    model = GPT(config)
    
    # Count parameters
    num_params = model.num_parameters()
    print(f"\nTotal parameters: {num_params:,}")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    tokens = np.random.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"\nInput tokens shape: {tokens.shape}")
    
    # Forward pass
    logits = model(tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {seq_len}, {config.vocab_size})")
    
    # Test generation
    print("\n--- Generation Demo ---")
    prompt = np.array([[1, 5, 10]])  # Start with 3 tokens
    generated = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
    print(f"Prompt: {prompt[0].tolist()}")
    print(f"Generated: {generated[0].tolist()}")
    
    print("\n✓ GPT model working!")
