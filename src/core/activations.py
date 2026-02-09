"""
Activation Functions - Adding non-linearity to neural networks.

Without activation functions, a neural network would just be a linear
transformation (matrix multiplication). Activation functions add the
non-linearity needed to learn complex patterns.

Learning Goals:
- Understand why we need activation functions
- Learn the most common activation functions used in transformers
- See how gradients are computed for each activation
"""

import numpy as np
from .tensor import Tensor
from typing import Optional


def relu(x: Tensor) -> Tensor:
    """
    ReLU (Rectified Linear Unit): f(x) = max(0, x)
    
    The simplest and most popular activation function.
    - If x > 0: output x
    - If x <= 0: output 0
    
    Gradient:
    - If x > 0: gradient is 1
    - If x <= 0: gradient is 0
    """
    out = Tensor(
        np.maximum(0, x.data),
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='relu'
    )
    
    def _backward():
        if x.requires_grad:
            # Gradient is 1 where x > 0, else 0
            grad = (x.data > 0).astype(np.float32) * out.grad
            x.grad = x.grad + grad if x.grad is not None else grad
    
    out._backward = _backward
    return out


def gelu(x: Tensor) -> Tensor:
    """
    GELU (Gaussian Error Linear Unit) - Used in GPT and BERT.
    
    GELU is smoother than ReLU and works better for transformers.
    Approximation: f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    
    This is the activation function used in most modern language models!
    """
    # Approximate GELU using tanh
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    x_cubed = x.data ** 3
    inner = sqrt_2_over_pi * (x.data + 0.044715 * x_cubed)
    tanh_inner = np.tanh(inner)
    result = 0.5 * x.data * (1 + tanh_inner)
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='gelu'
    )
    
    def _backward():
        if x.requires_grad:
            # Gradient of GELU (using chain rule)
            sech2 = 1 - tanh_inner ** 2  # sech^2 = 1 - tanh^2
            inner_grad = sqrt_2_over_pi * (1 + 3 * 0.044715 * x.data ** 2)
            
            grad = 0.5 * (1 + tanh_inner) + 0.5 * x.data * sech2 * inner_grad
            grad = grad * out.grad
            x.grad = x.grad + grad if x.grad is not None else grad
    
    out._backward = _backward
    return out


def tanh(x: Tensor) -> Tensor:
    """
    Tanh (Hyperbolic Tangent): f(x) = (e^x - e^-x) / (e^x + e^-x)
    
    Outputs values between -1 and 1.
    Often used in RNNs and LSTMs.
    
    Gradient: f'(x) = 1 - tanh(x)^2
    """
    t = np.tanh(x.data)
    out = Tensor(
        t,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='tanh'
    )
    
    def _backward():
        if x.requires_grad:
            # Gradient of tanh is 1 - tanh^2
            grad = (1 - t ** 2) * out.grad
            x.grad = x.grad + grad if x.grad is not None else grad
    
    out._backward = _backward
    return out


def sigmoid(x: Tensor) -> Tensor:
    """
    Sigmoid: f(x) = 1 / (1 + e^-x)
    
    Outputs values between 0 and 1.
    Often used for binary classification or gates.
    
    Gradient: f'(x) = sigmoid(x) * (1 - sigmoid(x))
    """
    s = 1 / (1 + np.exp(-x.data))
    out = Tensor(
        s,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='sigmoid'
    )
    
    def _backward():
        if x.requires_grad:
            # Gradient of sigmoid
            grad = s * (1 - s) * out.grad
            x.grad = x.grad + grad if x.grad is not None else grad
    
    out._backward = _backward
    return out


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Softmax: Converts scores to probabilities.
    
    f(x_i) = exp(x_i) / sum(exp(x_j))
    
    This is used at the output layer to get probability distribution
    over vocabulary (which token comes next?).
    
    The output sums to 1 and all values are positive.
    """
    # Subtract max for numerical stability
    x_max = np.max(x.data, axis=axis, keepdims=True)
    exp_x = np.exp(x.data - x_max)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    result = exp_x / sum_exp
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='softmax'
    )
    
    def _backward():
        if x.requires_grad:
            # Jacobian of softmax is: diag(s) - s @ s.T
            # For efficiency, we use: grad_input = s * (grad_output - sum(grad_output * s))
            s = result
            sum_grad = np.sum(out.grad * s, axis=axis, keepdims=True)
            grad = s * (out.grad - sum_grad)
            x.grad = x.grad + grad if x.grad is not None else grad
    
    out._backward = _backward
    return out


def log_softmax(x: Tensor, axis: int = -1) -> Tensor:
    """
    Log Softmax: log(softmax(x))
    
    More numerically stable than computing log(softmax(x)) separately.
    Used in cross-entropy loss computation.
    """
    # log_softmax = x - max(x) - log(sum(exp(x - max(x))))
    x_max = np.max(x.data, axis=axis, keepdims=True)
    shifted = x.data - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
    result = shifted - log_sum_exp
    
    out = Tensor(
        result,
        requires_grad=x.requires_grad,
        _children=(x,),
        _op='log_softmax'
    )
    
    def _backward():
        if x.requires_grad:
            # Gradient: I - softmax
            softmax_result = np.exp(result)
            sum_grad = np.sum(out.grad, axis=axis, keepdims=True)
            grad = out.grad - softmax_result * sum_grad
            x.grad = x.grad + grad if x.grad is not None else grad
    
    out._backward = _backward
    return out


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Activation Functions Demo")
    print("=" * 50)
    
    # Create test tensor
    x = Tensor([[-2, -1, 0, 1, 2]], requires_grad=True)
    print(f"Input x = {x.data}")
    
    # Test each activation
    print("\n--- ReLU ---")
    y_relu = relu(x)
    print(f"ReLU(x) = {y_relu.data}")
    
    print("\n--- GELU ---")
    y_gelu = gelu(x)
    print(f"GELU(x) = {y_gelu.data}")
    
    print("\n--- Tanh ---")
    y_tanh = tanh(x)
    print(f"Tanh(x) = {y_tanh.data}")
    
    print("\n--- Sigmoid ---")
    y_sigmoid = sigmoid(x)
    print(f"Sigmoid(x) = {y_sigmoid.data}")
    
    print("\n--- Softmax ---")
    y_softmax = softmax(x)
    print(f"Softmax(x) = {y_softmax.data}")
    print(f"Sum of softmax = {np.sum(y_softmax.data):.4f} (should be 1.0)")
    
    # Test backward pass
    print("\n--- Testing Gradient ---")
    x2 = Tensor([[1, 2, 3]], requires_grad=True)
    y = gelu(x2)
    loss = y.sum()
    loss.backward()
    print(f"x = {x2.data}")
    print(f"GELU(x) = {y.data}")
    print(f"Gradient = {x2.grad}")
    
    print("\n✓ All activations working!")
