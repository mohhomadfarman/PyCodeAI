"""
Loss Functions - Measuring how wrong the model is.

The loss function tells us how far off the model's predictions are
from the correct answers. During training, we minimize the loss.

For language models, we use Cross-Entropy Loss:
- Model outputs probability distribution over vocabulary
- We want high probability on the correct next token
- Cross-entropy penalizes low probability on correct answer

Learning Goals:
- Understand how we measure model performance
- Learn cross-entropy for classification/language modeling
- See how loss gradients flow back
"""

import numpy as np
from ..core.tensor import Tensor
from typing import Optional


def cross_entropy_loss(
    logits: Tensor, 
    targets: np.ndarray,
    ignore_index: int = -100
) -> Tensor:
    """
    Cross-Entropy Loss for next-token prediction.
    
    For each position, the model outputs logits (scores) for each vocabulary token.
    We want high logits on the correct token.
    
    Loss = -log(softmax(logits)[correct_token])
    
    Args:
        logits: Model output, shape (batch, seq_len, vocab_size)
        targets: Correct token IDs, shape (batch, seq_len)
        ignore_index: Ignore positions with this target value
    
    Returns:
        Scalar loss tensor
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for easier computation
    logits_flat = logits.data.reshape(-1, vocab_size)  # (batch*seq, vocab)
    targets_flat = targets.reshape(-1)  # (batch*seq,)
    
    # Compute log softmax (numerically stable)
    logits_max = np.max(logits_flat, axis=-1, keepdims=True)
    logits_shifted = logits_flat - logits_max
    log_sum_exp = np.log(np.sum(np.exp(logits_shifted), axis=-1, keepdims=True))
    log_probs = logits_shifted - log_sum_exp  # (batch*seq, vocab)
    
    # Create mask for valid positions
    mask = (targets_flat != ignore_index).astype(np.float32)
    valid_count = np.sum(mask)
    
    if valid_count == 0:
        return Tensor(np.array(0.0), requires_grad=False)
    
    # Get log probability of correct tokens
    # Use advanced indexing: for each position, get log_prob of target token
    indices = np.arange(len(targets_flat))
    valid_targets = np.where(targets_flat == ignore_index, 0, targets_flat)
    correct_log_probs = log_probs[indices, valid_targets]  # (batch*seq,)
    
    # Apply mask and compute mean loss
    masked_log_probs = correct_log_probs * mask
    loss_value = -np.sum(masked_log_probs) / valid_count
    
    loss = Tensor(
        np.array(loss_value),
        requires_grad=True,
        _children=(logits,),
        _op='cross_entropy'
    )
    
    def _backward():
        if logits.requires_grad:
            # Gradient of cross-entropy with softmax
            # grad = softmax(logits) - one_hot(targets)
            softmax_probs = np.exp(log_probs)  # (batch*seq, vocab)
            
            # Subtract 1 from probability of correct class
            grad = softmax_probs.copy()
            grad[indices, valid_targets] -= 1.0
            
            # Apply mask and normalize
            grad = grad * mask.reshape(-1, 1) / valid_count
            
            # Reshape back
            grad = grad.reshape(batch_size, seq_len, vocab_size)
            
            # Multiply by upstream gradient (usually 1 for scalar loss)
            grad = grad * loss.grad
            
            logits.grad = logits.grad + grad if logits.grad is not None else grad
    
    loss._backward = _backward
    return loss


def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Squared Error Loss.
    
    Loss = mean((predictions - targets)^2)
    
    Useful for regression tasks.
    """
    diff = predictions.data - targets.data
    loss_value = np.mean(diff ** 2)
    
    loss = Tensor(
        np.array(loss_value),
        requires_grad=predictions.requires_grad,
        _children=(predictions,),
        _op='mse'
    )
    
    def _backward():
        if predictions.requires_grad:
            # Gradient: 2 * (pred - target) / n
            n = predictions.data.size
            grad = 2 * diff / n * loss.grad
            predictions.grad = predictions.grad + grad if predictions.grad is not None else grad
    
    loss._backward = _backward
    return loss


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Loss Functions Demo")
    print("=" * 50)
    
    # Simulate model output
    batch_size = 2
    seq_len = 4
    vocab_size = 10
    
    # Random logits (model output)
    logits = Tensor(np.random.randn(batch_size, seq_len, vocab_size), requires_grad=True)
    
    # Target tokens
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Targets: {targets}")
    
    # Compute loss
    loss = cross_entropy_loss(logits, targets)
    print(f"\nCross-entropy loss: {loss.data:.4f}")
    
    # Backward pass
    loss.backward()
    print(f"Gradient shape: {logits.grad.shape}")
    print(f"Gradient sum: {np.sum(logits.grad):.6f} (should be ~0)")
    
    # Test with ignore_index
    print("\n--- Testing ignore_index ---")
    targets_with_ignore = targets.copy()
    targets_with_ignore[0, 0] = -100  # Ignore first position
    loss2 = cross_entropy_loss(logits, targets_with_ignore, ignore_index=-100)
    print(f"Loss with ignored position: {loss2.data:.4f}")
    
    print("\n✓ Loss functions working!")
