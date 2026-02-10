"""
Code Generator - Generate JavaScript/TypeScript code with your AI.

This is where the magic happens! After training, your model can
generate code by predicting one token at a time.

Sampling strategies:
- Greedy: Always pick the highest probability token
- Temperature: Control randomness (higher = more random)
- Top-k: Only consider top k most likely tokens
- Top-p (nucleus): Only consider tokens with cumulative prob < p

Learning Goals:
- Understand autoregressive generation
- Learn different sampling strategies
- See how prompts guide generation
"""

import numpy as np
from ..core import backend as _backend
from ..core.backend import to_numpy, to_device
from ..core.tensor import no_grad
from typing import Optional, List
from ..models.gpt import GPT
from ..tokenizer.tokenizer import Tokenizer


class CodeGenerator:
    """
    Code Generator using trained GPT model.

    Generate JavaScript/TypeScript code from prompts.

    Example:
        generator = CodeGenerator(model, tokenizer)
        code = generator.generate("function fibonacci(n) {")
        print(code)
    """

    def __init__(self, model: GPT, tokenizer: Tokenizer):
        """
        Initialize code generator.

        Args:
            model: Trained GPT model
            tokenizer: Tokenizer for encoding/decoding
        """
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        stop_tokens: Optional[List[str]] = None
    ) -> str:
        """
        Generate code from a prompt.

        Args:
            prompt: Starting code/text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy, 1.0 = default)
            top_k: Only sample from top k tokens
            top_p: Nucleus sampling threshold
            stop_tokens: Stop generation when these appear

        Returns:
            Generated code string
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special=False)

        # Get stop token IDs if specified
        stop_ids = set()
        if stop_tokens:
            for token in stop_tokens:
                if token in self.tokenizer.vocab:
                    stop_ids.add(self.tokenizer.vocab[token])

        # Also stop at EOS
        stop_ids.add(self.tokenizer.eos_token_id)

        # Generate tokens one at a time (no gradients needed for inference)
        generated = list(input_ids)

        with no_grad():
            for _ in range(max_tokens):
                # Get current context (limited by max_seq_len)
                context = to_device(np.array([generated[-self.model.config.max_seq_len:]]))

                # Forward pass
                logits = self.model(context)

                # Get logits for last position
                next_logits = logits.data[0, -1, :]  # (vocab_size,)

                # Apply sampling strategy
                next_token_id = self._sample(
                    next_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p
                )

                # Check for stop condition
                if next_token_id in stop_ids:
                    break

                # Append to generated sequence
                generated.append(next_token_id)

        # Decode and return
        return self.tokenizer.decode(generated, skip_special=True)

    def _sample(
        self,
        logits,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> int:
        """
        Sample next token from logits.

        Args:
            logits: Unnormalized scores for each token
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus (top-p) filtering

        Returns:
            Sampled token ID
        """
        # Temperature scaling
        if temperature == 0.0:
            # Greedy decoding
            return int(_backend.xp.argmax(logits))

        logits = logits / temperature

        # Convert to probabilities
        logits_max = _backend.xp.max(logits)
        exp_logits = _backend.xp.exp(logits - logits_max)
        probs = exp_logits / _backend.xp.sum(exp_logits)

        # Top-k filtering
        if top_k is not None and top_k > 0:
            # Keep only top k tokens
            top_k = min(top_k, len(probs))
            top_k_indices = _backend.xp.argsort(probs)[-top_k:]
            mask = _backend.xp.zeros_like(probs)
            mask[top_k_indices] = 1
            probs = probs * mask
            probs = probs / _backend.xp.sum(probs)

        # Top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_indices = _backend.xp.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = _backend.xp.cumsum(sorted_probs)

            # Find cutoff
            cumulative_np = to_numpy(cumulative_probs)
            cutoff_idx = int(np.searchsorted(cumulative_np, top_p))
            cutoff_idx = min(cutoff_idx + 1, len(probs))

            # Zero out tokens beyond cutoff
            mask = _backend.xp.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / _backend.xp.sum(probs)

        # Sample from distribution (convert to numpy for np.random.choice)
        probs_np = to_numpy(probs)
        return int(np.random.choice(len(probs_np), p=probs_np))

    def complete(
        self,
        code: str,
        num_completions: int = 3,
        max_tokens: int = 50,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Generate multiple completions for code.

        Args:
            code: Starting code
            num_completions: Number of completions to generate
            max_tokens: Max tokens per completion
            temperature: Sampling temperature

        Returns:
            List of completion strings
        """
        completions = []

        for _ in range(num_completions):
            completed = self.generate(
                code,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=40
            )
            # Extract just the generated part
            completion = completed[len(code):] if completed.startswith(code) else completed
            completions.append(completion)

        return completions

    def interactive_session(self):
        """
        Start an interactive code generation session.
        """
        print("=" * 60)
        print("Code Generator Interactive Session")
        print("=" * 60)
        print("Enter a code prompt and press Enter to generate.")
        print("Type 'quit' to exit.\n")

        while True:
            prompt = input(">>> ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt.strip():
                continue

            print("\nGenerating...\n")

            try:
                result = self.generate(
                    prompt,
                    max_tokens=100,
                    temperature=0.7,
                    top_k=50
                )
                print("Generated code:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                print()
            except Exception as e:
                print(f"Error: {e}\n")


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Code Generator Demo")
    print("=" * 50)

    # This demo shows the API - actual generation requires a trained model
    print("\nTo use the generator:")
    print("1. Train a model using the Trainer")
    print("2. Create a CodeGenerator with model and tokenizer")
    print("3. Call generate() with your prompt")
    print("\nExample:")
    print("  generator = CodeGenerator(model, tokenizer)")
    print('  code = generator.generate("function add(a, b) {")')
    print("  print(code)")

    print("\n✓ Generator module ready!")
