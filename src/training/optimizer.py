"""
Optimizers - How the model learns.

Optimizers update the model's parameters using gradients.
The goal: gradually change parameters to minimize the loss.

Basic idea (Gradient Descent):
    parameter = parameter - learning_rate * gradient

Learning Goals:
- Understand gradient descent
- Learn about learning rate
- See momentum and Adam optimizer
"""

import math
import numpy as np
from ..core.backend import xp
from ..core.tensor import Tensor
from typing import List, Optional


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    The simplest optimizer: move parameters in the direction
    that decreases the loss.

    With momentum: remember previous updates for smoother training.

    update = momentum * previous_update + learning_rate * gradient
    parameter = parameter - update
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0
    ):
        """
        Initialize SGD optimizer.

        Args:
            parameters: List of model parameters (Tensors)
            lr: Learning rate (step size)
            momentum: Momentum factor (0 = no momentum)
        """
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

        # Initialize velocity for momentum
        if momentum > 0:
            self.velocity = [xp.zeros_like(p.data) for p in parameters]
        else:
            self.velocity = None

    def step(self):
        """
        Update parameters using computed gradients.

        Call this after loss.backward()
        """
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            if self.momentum > 0:
                # With momentum
                self.velocity[i] = (
                    self.momentum * self.velocity[i] +
                    self.lr * param.grad
                )
                param.data = param.data - self.velocity[i]
            else:
                # Simple SGD
                param.data = param.data - self.lr * param.grad

    def zero_grad(self):
        """Reset all gradients to None."""
        for param in self.parameters:
            param.grad = None


class Adam:
    """
    Adam optimizer - the most popular optimizer for deep learning.

    Adam combines:
    1. Momentum (first moment - running average of gradients)
    2. RMSprop (second moment - running average of squared gradients)

    This adapts the learning rate for each parameter based on
    the history of gradients.

    Includes bias correction for accurate estimates early in training.
    """

    def __init__(
        self,
        parameters: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize Adam optimizer.

        Args:
            parameters: List of model parameters
            lr: Learning rate
            betas: Coefficients for running averages (beta1, beta2)
            eps: Small constant for numerical stability
            weight_decay: L2 regularization factor
        """
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment estimates
        self.m = [xp.zeros_like(p.data) for p in parameters]  # First moment
        self.v = [xp.zeros_like(p.data) for p in parameters]  # Second moment
        self.t = 0  # Timestep

    def step(self):
        """Update parameters using Adam algorithm."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad

            # Apply weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data

            # Update first moment (mean of gradients)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update second moment (mean of squared gradients)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction (important early in training)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data = param.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset all gradients to None."""
        for param in self.parameters:
            param.grad = None



class AdamW(Adam):
    """
    Fused AdamW - Adam with decoupled weight decay using custom CUDA kernels.
    
    Significantly faster than standard AdamW by fusing:
    - Weight decay
    - Momentum update
    - Variance update
    - Bias correction
    - Parameter update
    
    Into a single GPU kernel launch per parameter.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._kernel = None
        
        # Compile kernel if GPU is available
        if xp.__name__ == 'cupy':
            try:
                self._kernel = xp.ElementwiseKernel(
                    'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T beta1, T beta2, T eps, T weight_decay, T m_correction, T v_correction',
                    'T m, T v, T param',
                    '''
                    // Weight decay (decoupled)
                    if (weight_decay > 0) {
                        param -= lr * weight_decay * param;
                    }
                    
                    // Update moments
                    m = beta1 * m + one_minus_beta1 * grad;
                    v = beta2 * v + one_minus_beta2 * grad * grad;
                    
                    // Bias correction
                    T m_hat = m / m_correction;
                    T v_hat = v / v_correction;
                    
                    // Update parameter
                    param -= lr * m_hat / (sqrt(v_hat) + eps);
                    ''',
                    'adamw_fused_kernel'
                )
            except Exception as e:
                print(f"WARNING: FusedAdamW kernel compilation failed (using fallback): {e}")
                self._kernel = None

    def step(self):
        """Update parameters using Fused AdamW algorithm."""
        self.t += 1
        
        # Bias correction terms (scalars)
        m_correction = 1 - self.beta1 ** self.t
        v_correction = 1 - self.beta2 ** self.t
        
        # Constants for kernel
        one_minus_beta1 = 1 - self.beta1
        one_minus_beta2 = 1 - self.beta2
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            # If using CPU or kernel not compiled, fall back to slow python version
            if self._kernel is None or not isinstance(param.data, xp.ndarray):
                self._step_cpu(i, param, m_correction, v_correction)
            else:
                self._kernel(
                    param.grad, 
                    self.lr, 
                    one_minus_beta1, 
                    one_minus_beta2, 
                    self.beta1, 
                    self.beta2, 
                    self.eps, 
                    self.weight_decay,
                    m_correction,
                    v_correction,
                    self.m[i], 
                    self.v[i], 
                    param.data
                )
                
    def _step_cpu(self, i, param, m_correction, v_correction):
        """Fallback for CPU."""
        grad = param.grad
        
        # Regular implementation
        self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
        self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
        
        m_hat = self.m[i] / m_correction
        v_hat = self.v[i] / v_correction
        
        param.data = param.data - self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
        
        if self.weight_decay > 0:
            param.data = param.data - self.lr * self.weight_decay * param.data


class LearningRateScheduler:
    """
    Learning rate scheduler for warmup and decay.

    Modern language models use:
    1. Linear warmup: gradually increase LR at start
    2. Cosine decay: smoothly decrease LR

    This helps training stability and final performance.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int = 100,
        total_steps: int = 10000,
        min_lr: float = 1e-6
    ):
        """
        Initialize scheduler.

        Args:
            optimizer: The optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        """Update learning rate based on current step."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self.optimizer.lr = lr
        return lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.lr


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Optimizers Demo")
    print("=" * 50)

    # Create some parameters
    param1 = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    param2 = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), requires_grad=True)

    print("Initial parameters:")
    print(f"param1: {param1.data}")
    print(f"param2: {param2.data}")

    # Simulate gradients
    param1.grad = np.array([0.1, 0.2, 0.3])
    param2.grad = np.array([[0.1, 0.2], [0.3, 0.4]])

    # Test SGD
    print("\n--- SGD Update ---")
    sgd = SGD([param1, param2], lr=0.1)
    sgd.step()
    print(f"param1 after SGD: {param1.data}")
    print(f"param2 after SGD: {param2.data}")

    # Reset and test Adam
    param1.data = np.array([1.0, 2.0, 3.0])
    param2.data = np.array([[1.0, 2.0], [3.0, 4.0]])
    param1.grad = np.array([0.1, 0.2, 0.3])
    param2.grad = np.array([[0.1, 0.2], [0.3, 0.4]])

    print("\n--- Adam Update ---")
    adam = Adam([param1, param2], lr=0.1)
    adam.step()
    print(f"param1 after Adam: {param1.data}")
    print(f"param2 after Adam: {param2.data}")

    # Test learning rate scheduler
    print("\n--- Learning Rate Schedule ---")
    param1.data = np.array([1.0, 2.0, 3.0])
    adam = Adam([param1], lr=0.001)
    scheduler = LearningRateScheduler(adam, warmup_steps=10, total_steps=100)

    lrs = []
    for i in range(100):
        lr = scheduler.step()
        if i % 20 == 0:
            print(f"Step {i}: LR = {lr:.6f}")
        lrs.append(lr)

    print(f"Max LR: {max(lrs):.6f}")
    print(f"Min LR: {min(lrs):.6f}")

    print("\n✓ Optimizers working!")
