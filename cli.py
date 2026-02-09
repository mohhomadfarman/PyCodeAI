#!/usr/bin/env python3
"""
PyCodeAI - Build Your Own AI From Scratch

Command-line interface for training and using your code generation AI.

Usage:
    python cli.py train                    # Train the model
    python cli.py generate "your prompt"   # Generate code
    python cli.py interactive              # Interactive session
    python cli.py test                     # Test all components
"""

import argparse
import sys
import os
import time

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


def test_components():
    """Test all components of the system."""
    print("=" * 60)
    print("🧪 Testing PyCodeAI Components")
    print("=" * 60)
    
    # Test 1: Tensor
    print("\n1. Testing Tensor (autograd)...")
    from src.core.tensor import Tensor
    x = Tensor([1, 2, 3], requires_grad=True)
    y = x * 2
    z = y.sum()
    z.backward()
    assert x.grad is not None, "Gradient not computed"
    print("   ✓ Tensor works!")
    
    # Test 2: Activations
    print("\n2. Testing Activations...")
    from src.core.activations import relu, gelu, softmax
    x = Tensor([[1, -1, 2]])
    y = relu(x)
    assert np.all(y.data >= 0), "ReLU failed"
    print("   ✓ Activations work!")
    
    # Test 3: Linear layer
    print("\n3. Testing Linear layer...")
    from src.layers.linear import Linear
    layer = Linear(10, 5)
    x = Tensor(np.random.randn(2, 10))
    y = layer(x)
    assert y.shape == (2, 5), "Wrong output shape"
    print("   ✓ Linear layer works!")
    
    # Test 4: Attention
    print("\n4. Testing Attention...")
    from src.layers.attention import MultiHeadAttention
    attn = MultiHeadAttention(embed_dim=32, num_heads=4)
    x = Tensor(np.random.randn(2, 8, 32))
    y = attn(x)
    assert y.shape == x.shape, "Wrong output shape"
    print("   ✓ Attention works!")
    
    # Test 5: Transformer Block
    print("\n5. Testing Transformer Block...")
    from src.models.transformer import TransformerBlock
    block = TransformerBlock(embed_dim=32, num_heads=4)
    x = Tensor(np.random.randn(2, 8, 32))
    y = block(x)
    assert y.shape == x.shape, "Wrong output shape"
    print("   ✓ Transformer Block works!")
    
    # Test 6: GPT Model
    print("\n6. Testing GPT Model...")
    from src.models.gpt import GPT, GPTConfig
    config = GPTConfig(vocab_size=50, max_seq_len=16, embed_dim=32, num_heads=2, num_layers=2)
    model = GPT(config)
    tokens = np.array([[1, 5, 10, 2]])
    logits = model(tokens)
    assert logits.shape == (1, 4, 50), "Wrong output shape"
    print(f"   ✓ GPT works! Parameters: {model.num_parameters():,}")
    
    # Test 7: Tokenizer
    print("\n7. Testing Tokenizer...")
    from src.tokenizer.tokenizer import Tokenizer
    tokenizer = Tokenizer(vocab_size=200)
    tokenizer.build_vocab(["function test() { return 1; }"])
    encoded = tokenizer.encode("function add()")
    decoded = tokenizer.decode(encoded)
    print(f"   ✓ Tokenizer works! Vocab size: {len(tokenizer.vocab)}")
    
    # Test 8: Loss function
    print("\n8. Testing Loss function...")
    from src.training.loss import cross_entropy_loss
    logits = Tensor(np.random.randn(2, 4, 50), requires_grad=True)
    targets = np.random.randint(0, 50, (2, 4))
    loss = cross_entropy_loss(logits, targets)
    loss.backward()
    assert logits.grad is not None, "Loss gradient not computed"
    print("   ✓ Loss function works!")
    
    # Test 9: Optimizer
    print("\n9. Testing Optimizer...")
    from src.training.optimizer import Adam
    from src.core.tensor import Tensor
    param = Tensor([1.0, 2.0], requires_grad=True)
    param.grad = np.array([0.1, 0.2])
    optimizer = Adam([param], lr=0.1)
    old_data = param.data.copy()
    optimizer.step()
    assert not np.allclose(param.data, old_data), "Weights not updated"
    print("   ✓ Optimizer works!")
    
    print("\n" + "=" * 60)
    print("✅ All components working!")
    print("=" * 60)


def train_model(args):
    """Train the model."""
    from src.models.gpt import GPT, GPTConfig
    from src.tokenizer.tokenizer import Tokenizer
    from src.training.trainer import Trainer, DataLoader
    from src.training.optimizer import Adam, LearningRateScheduler
    from src.training.optimizer import Adam, LearningRateScheduler
    from data.samples import get_training_data, get_chat_only_data
    
    print("=" * 60)
    print("🚀 Training PyCodeAI")
    print("=" * 60)
    
    # Load training data
    print("\n1. Loading training data...")
    if hasattr(args, 'chat_only') and args.chat_only:
        print("   🗣️  Chat Mode: Loading dedicated conversation data")
        code_samples = get_chat_only_data()
    else:
        code_samples = get_training_data()
    print(f"   Loaded {len(code_samples)} samples")
    
    # Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = Tokenizer(vocab_size=args.vocab_size)
    
    if args.load_model:
        print(f"   Loading tokenizer from {args.output_tokenizer}...")
        try:
            tokenizer.load(args.output_tokenizer)
        except Exception as e:
            print(f"   ⚠️ Could not load tokenizer: {e}. Re-building.")
            tokenizer.build_vocab(code_samples)
    else:
        tokenizer.build_vocab(code_samples)
        
    print(f"   Vocabulary size: {len(tokenizer.vocab)}")
    
    # Tokenize all samples
    print("\n3. Tokenizing data...")
    tokenized_data = []
    for sample in code_samples:
        tokens = tokenizer.encode(sample)
        if len(tokens) > 2:  # Skip empty samples
            tokenized_data.append(tokens)
    print(f"   Tokenized {len(tokenized_data)} samples")
    
    # Create model
    print("\n4. Creating model...")
    config = GPTConfig(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers
    )
    model = GPT(config)
    
    if args.load_model:
        print(f"   Loading weights from {args.load_model}...")
        try:
            model.load(args.load_model)
            print("   ✅ Weights loaded successfully!")
        except Exception as e:
            print(f"   ⚠️ Could not load weights: {e}. Starting from scratch.")
            
    print(f"   Model parameters: {model.num_parameters():,}")
    print(config)
    
    # Create data loader
    print("\n5. Creating data loader...")
    dataloader = DataLoader(
        tokenized_data,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pad_token_id=tokenizer.pad_token_id
    )
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # Create trainer
    print("\n6. Setting up trainer...")
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(dataloader) * args.epochs
    scheduler = LearningRateScheduler(
        optimizer, 
        warmup_steps=min(100, total_steps // 10),
        total_steps=total_steps
    )
    
    trainer = Trainer(
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        log_interval=args.log_interval
    )
    
    # Train!
    print("\n7. Starting training...")
    history = trainer.train(
        dataloader, 
        epochs=args.epochs,
        accumulation_steps=args.grad_accum
    )
    
    # Save model and tokenizer
    print("\n8. Saving model...")
    model.save(args.output_model)
    tokenizer.save(args.output_tokenizer)
    
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print(f"   Model saved to: {args.output_model}")
    print(f"   Tokenizer saved to: {args.output_tokenizer}")
    print("=" * 60)
    
    return model, tokenizer


def generate_code(args):
    """Generate code from a prompt."""
    from src.models.gpt import GPT, GPTConfig
    from src.tokenizer.tokenizer import Tokenizer
    from src.inference.generator import CodeGenerator
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer_path)
    
    # Try to load from config first
    try:
        model = GPT.load_from_dir(args.model_path)
    except FileNotFoundError:
        print("Config file not found, falling back to command line arguments...")
        config = GPTConfig(
            vocab_size=len(tokenizer.vocab),
            max_seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers
        )
        model = GPT(config)
        model.load(args.model_path)
    
    # Generate
    generator = CodeGenerator(model, tokenizer)
    
    print("\nGenerating code...")
    print("-" * 40)
    
    result = generator.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )
    
    print(result)
    print("-" * 40)


def interactive_mode(args):
    """Interactive generation mode."""
    from src.models.gpt import GPT, GPTConfig
    from src.tokenizer.tokenizer import Tokenizer
    from src.inference.generator import CodeGenerator
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        print("Please train the model first: python cli.py train")
        return
    
    # Load model and tokenizer
    print("Loading model...")
    tokenizer = Tokenizer()
    tokenizer.load(args.tokenizer_path)
    
    # Try to load from config first
    try:
        model = GPT.load_from_dir(args.model_path)
    except FileNotFoundError:
        print("Config file not found, falling back to command line arguments...")
        config = GPTConfig(
            vocab_size=len(tokenizer.vocab),
            max_seq_len=args.seq_len,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers
        )
        model = GPT(config)
        model.load(args.model_path)
    
    # Start interactive session
    generator = CodeGenerator(model, tokenizer)
    generator.interactive_session()


def crawl_repo(args):
    """Crawl a GitHub repository."""
    from src.data.crawler import GitHubCrawler
    
    print("=" * 60)
    print("🕷️  GitHub Crawler")
    print("=" * 60)
    
    # Default to data/crawled if not specified
    output_dir = args.output if args.output != "data/training_data.txt" else "data/crawled"
    
    crawler = GitHubCrawler(
        output_dir=output_dir,
        token=args.token
    )
    
    crawler.crawl(args.repo)


def crawl_web(args):
    """Crawl a web article."""
    from src.data.web_crawler import WebCrawler
    
    print("=" * 60)
    print("🕷️  Web Crawler")
    print("=" * 60)
    
    crawler = WebCrawler(output_dir=args.output)
    crawler.crawl_article(args.url)


def main():
    parser = argparse.ArgumentParser(
        description="PyCodeAI - Build Your Own AI From Scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test all components")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    train_parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    train_parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    train_parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    train_parser.add_argument("--vocab-size", type=int, default=5000, help="Vocabulary size")
    train_parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    train_parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    train_parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    train_parser.add_argument("--chat-only", action="store_true", help="Train only on chat data")
    train_parser.add_argument("--load-model", type=str, help="Path to existing model to finetune")
    train_parser.add_argument("--output-model", type=str, default="model.npz", help="Output model path")
    train_parser.add_argument("--output-tokenizer", type=str, default="tokenizer.json", help="Output tokenizer path")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate code")
    gen_parser.add_argument("prompt", type=str, help="Code prompt")
    gen_parser.add_argument("--model-path", type=str, default="model.npz", help="Model path")
    gen_parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json", help="Tokenizer path")
    gen_parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling")
    gen_parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    gen_parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    gen_parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    gen_parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")
    
    # Interactive command
    int_parser = subparsers.add_parser("interactive", help="Interactive mode")
    int_parser.add_argument("--model-path", type=str, default="model.npz", help="Model path")
    int_parser.add_argument("--tokenizer-path", type=str, default="tokenizer.json", help="Tokenizer path")
    int_parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    int_parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    int_parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    int_parser.add_argument("--num-layers", type=int, default=4, help="Number of transformer layers")

    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl GitHub repo")
    crawl_parser.add_argument("repo", type=str, help="GitHub repo (owner/repo)")
    crawl_parser.add_argument("--output", type=str, default="data/crawled", help="Output directory")
    crawl_parser.add_argument("--token", type=str, default=None, help="GitHub API token")
    
    # Web Crawl command
    web_parser = subparsers.add_parser("crawl-web", help="Crawl web article")
    web_parser.add_argument("url", type=str, help="Article URL")
    web_parser.add_argument("--output", type=str, default="data/articles", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "test":
        test_components()
    elif args.command == "train":
        train_model(args)
    elif args.command == "generate":
        generate_code(args)
    elif args.command == "interactive":
        interactive_mode(args)
    elif args.command == "crawl":
        crawl_repo(args)
    elif args.command == "crawl-web":
        crawl_web(args)
    else:
        parser.print_help()



if __name__ == "__main__":
    main()
