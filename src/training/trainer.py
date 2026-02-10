"""
Trainer - The training loop that makes the model learn.

The training process:
1. Forward pass: Get model predictions
2. Compute loss: How wrong are we?
3. Backward pass: Compute gradients (how to improve)
4. Update weights: Apply gradients to improve

This is where all the pieces come together!

Learning Goals:
- Understand the training loop
- See how batching works
- Learn about training metrics
"""

import numpy as np
from ..core.backend import xp, to_device, to_numpy, is_gpu
import time
from typing import List, Callable, Optional, Dict, Any
from ..core.tensor import Tensor
from ..models.gpt import GPT
from .loss import cross_entropy_loss
from .optimizer import Adam, LearningRateScheduler






class DataLoader:
    """
    DataLoader for batching training data.

    Takes sequences of token IDs and creates batches
    of (input, target) pairs for training.

    For language modeling:
    - Input: tokens[:-1]
    - Target: tokens[1:]
    """

    def __init__(
        self,
        data: List[List[int]],
        batch_size: int = 32,
        seq_len: int = 64,
        pad_token_id: int = 0,
        shuffle: bool = True
    ):
        """
        Initialize data loader.

        Args:
            data: List of tokenized sequences
            batch_size: Number of samples per batch
            seq_len: Sequence length for training
            pad_token_id: Token ID for padding
            shuffle: Whether to shuffle data
        """
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.shuffle = shuffle

        # Flatten all data into one long sequence
        self.flat_data = np.concatenate([np.array(seq) for seq in data if len(seq) > 1])

        # Calculate number of batches
        total_tokens = len(self.flat_data)
        tokens_per_batch = batch_size * seq_len
        self.num_batches = max(1, total_tokens // tokens_per_batch)

    def __iter__(self):
        """Iterate over batches."""
        # Reshape data for batching
        usable_length = self.num_batches * self.batch_size * self.seq_len

        # Start from random position if shuffling
        if self.shuffle:
            start_idx = np.random.randint(0, min(1000, len(self.flat_data) - usable_length - 1))
        else:
            start_idx = 0

        data = self.flat_data[start_idx:start_idx + usable_length + 1]

        # Precompute index offsets for vectorized batch construction
        seq_offsets = np.arange(self.seq_len, dtype=np.int64)
        batch_offsets = np.arange(self.batch_size, dtype=np.int64) * self.seq_len

        for batch_idx in range(self.num_batches):
            start = batch_idx * self.batch_size * self.seq_len

            # Vectorized batch construction (no Python loop over batch_size)
            indices = start + batch_offsets[:, None] + seq_offsets[None, :]
            batch_input = data[indices]
            batch_target = data[indices + 1]

            yield (
                to_device(batch_input.astype(np.int64)),
                to_device(batch_target.astype(np.int64))
            )

    def __len__(self):
        return self.num_batches


class Trainer:
    """
    Trainer for GPT model.

    Handles the full training loop:
    - Forward pass
    - Loss computation
    - Backward pass
    - Weight updates
    - Logging and checkpointing
    """

    def __init__(
        self,
        model: GPT,
        optimizer: Optional[Adam] = None,
        scheduler: Optional[LearningRateScheduler] = None,
        log_interval: int = 10,
        checkpoint_interval: int = 100
    ):
        """
        Initialize trainer.

        Args:
            model: GPT model to train
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler
            log_interval: Log every N steps
            checkpoint_interval: Save checkpoint every N steps
        """
        self.model = model
        self.optimizer = optimizer or Adam(model.parameters(), lr=3e-4)
        self.scheduler = scheduler
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'lr': [],
            'tokens_per_sec': []
        }

    def train_step(self, input_ids: np.ndarray, target_ids: np.ndarray, accumulation_steps: int = 1) -> float:
        """
        Single training step.

        Args:
            input_ids: Input token IDs, shape (batch, seq_len)
            target_ids: Target token IDs, shape (batch, seq_len)
            accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Loss value (unscaled)
        """
        # Forward pass
        logits = self.model(input_ids)

        # Compute loss
        loss = cross_entropy_loss(logits, target_ids)

        # Backward pass (scale loss for accumulation)
        # We divide by accumulation_steps so that sum of gradients = average gradient
        (loss / accumulation_steps).backward()

        return float(loss.data)

    def _clip_gradients(self, max_norm: float = 1.0):
        """Clip gradients to prevent exploding gradients."""
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm += float(xp.sum(param.grad ** 2))
        total_norm = total_norm ** 0.5

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad *= scale

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 1,
        eval_dataloader: Optional[DataLoader] = None,
        accumulation_steps: int = 1
    ):
        """
        Full training loop.

        Args:
            dataloader: Training data loader
            epochs: Number of training epochs
            eval_dataloader: Optional validation data loader
            accumulation_steps: Number of steps to accumulate gradients before update (default: 1)
        """
        print("=" * 60)
        print("Starting Training")
        print(f"Model parameters: {self.model.num_parameters():,}")
        print(f"Epochs: {epochs}")
        print(f"Batches per epoch: {len(dataloader)}")
        print(f"Gradient Accumulation: {accumulation_steps} steps")
        print(f"Effective Batch Size: {dataloader.batch_size * accumulation_steps}")
        print("=" * 60)

        total_tokens = 0
        start_time = time.time()
        
        # Zero gradients initially
        self.optimizer.zero_grad()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
                
                # Training step (forward + backward)
                loss = self.train_step(input_ids, target_ids, accumulation_steps)

                # Update weights only every N steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(dataloader)):
                    # Gradient clipping
                    self._clip_gradients(max_norm=1.0)
                    
                    # Update weights
                    self.optimizer.step()
                    
                    # Update learning rate
                    if self.scheduler:
                        self.scheduler.step()
                    
                    # Reset gradients
                    self.optimizer.zero_grad()
                    
                    # Increment global step only on update
                    self.global_step += 1
                
                epoch_loss += loss
                epoch_steps += 1
                total_tokens += input_ids.size

                # Logging (every N batches, independent of updates)
                if (batch_idx + 1) % (self.log_interval * accumulation_steps) == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens / elapsed
                    current_lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.lr

                    self.history['loss'].append(loss)
                    self.history['lr'].append(current_lr)
                    self.history['tokens_per_sec'].append(tokens_per_sec)

                    print(
                        f"Epoch {epoch+1}/{epochs} | "
                        f"Step {self.global_step} | "
                        f"Loss: {loss:.4f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Tokens/s: {tokens_per_sec:.0f}"
                    )
                
                # Checkpointing (based on global update steps)
                if self.global_step > 0 and self.global_step % self.checkpoint_interval == 0 and ((batch_idx + 1) % accumulation_steps == 0):
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.save_checkpoint("best_model.npz")

            # End of epoch summary
            avg_loss = epoch_loss / epoch_steps
            print(f"\n{'='*40}")
            print(f"Epoch {epoch+1} complete | Average Loss: {avg_loss:.4f}")

            # Evaluation
            if eval_dataloader:
                eval_loss = self.evaluate(eval_dataloader)
                print(f"Validation Loss: {eval_loss:.4f}")

            print(f"{'='*40}\n")

        total_time = time.time() - start_time
        print(f"Training complete! Total time: {total_time:.1f}s")

        return self.history

    def evaluate(self, dataloader: DataLoader) -> float:
        """
        Evaluate model on data.

        Args:
            dataloader: Evaluation data loader

        Returns:
            Average loss
        """
        total_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in dataloader:
            # Forward pass only (no gradients)
            logits = self.model(input_ids)
            loss = cross_entropy_loss(logits, target_ids)
            total_loss += float(loss.data)
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        self.model.save(path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        self.model.load(path)
        print(f"Checkpoint loaded: {path}")


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Trainer Demo")
    print("=" * 50)

    from ..models.gpt import GPT, GPTConfig

    # Create a tiny model
    config = GPTConfig(
        vocab_size=100,
        max_seq_len=32,
        embed_dim=32,
        num_heads=2,
        num_layers=2
    )
    model = GPT(config)

    print(f"Model parameters: {model.num_parameters():,}")

    # Create fake training data
    fake_data = [
        list(range(50)),  # Sequence 1
        list(range(30, 80)),  # Sequence 2
        list(range(10, 60)),  # Sequence 3
    ] * 10  # Repeat for more data

    # Create data loader
    dataloader = DataLoader(
        fake_data,
        batch_size=4,
        seq_len=16
    )

    print(f"Number of batches: {len(dataloader)}")

    # Create trainer
    trainer = Trainer(
        model,
        optimizer=Adam(model.parameters(), lr=1e-3),
        log_interval=5
    )

    # Train for a few steps
    print("\n--- Training Demo (2 epochs) ---")
    history = trainer.train(dataloader, epochs=2)

    print("\n--- Loss History ---")
    print(f"Initial loss: {history['loss'][0]:.4f}")
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Loss reduction: {(1 - history['loss'][-1]/history['loss'][0])*100:.1f}%")

    print("\n✓ Trainer working!")
