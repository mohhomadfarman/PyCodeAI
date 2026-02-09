"""
Tokenizer - Converting code to tokens and back.

Before a neural network can process code, we need to convert
text into numbers. This is called tokenization.

Our tokenizer:
1. Breaks code into meaningful pieces (tokens)
2. Maps each token to a unique ID
3. Special handling for code syntax (keywords, operators, etc.)

Learning Goals:
- Understand how text is converted to numbers
- Learn about vocabulary building
- See special tokens (PAD, BOS, EOS, UNK)
"""

import re
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter


class Tokenizer:
    """
    Code Tokenizer for JavaScript/TypeScript.
    
    Features:
    - Word-piece style tokenization
    - Code-aware splitting (keywords, operators)
    - Special tokens for sequence control
    
    Example:
        tokenizer = Tokenizer()
        tokenizer.build_vocab(["function add(a, b) { return a + b; }"])
        
        tokens = tokenizer.encode("function test()")
        text = tokenizer.decode(tokens)
    """
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    BOS_TOKEN = "<BOS>"  # Beginning of sequence
    EOS_TOKEN = "<EOS>"  # End of sequence
    UNK_TOKEN = "<UNK>"  # Unknown token
    
    # Common JavaScript/TypeScript keywords
    KEYWORDS = {
        'function', 'const', 'let', 'var', 'return', 'if', 'else', 'for',
        'while', 'do', 'switch', 'case', 'break', 'continue', 'class',
        'extends', 'constructor', 'this', 'new', 'import', 'export',
        'from', 'default', 'async', 'await', 'try', 'catch', 'finally',
        'throw', 'typeof', 'instanceof', 'null', 'undefined', 'true',
        'false', 'void', 'type', 'interface', 'enum', 'implements',
        'public', 'private', 'protected', 'static', 'readonly', 'any',
        'number', 'string', 'boolean', 'object', 'never', 'unknown'
    }
    
    # Operators and punctuation
    OPERATORS = [
        '===', '!==', '==', '!=', '<=', '>=', '=>', '&&', '||', '??',
        '++', '--', '+=', '-=', '*=', '/=', '**', '...', '?.', 
        '+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|', '^', '~',
        '(', ')', '{', '}', '[', ']', ';', ':', ',', '.', '?'
    ]
    
    def __init__(self, vocab_size: int = 5000):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        
        # Initialize special tokens
        self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Add special tokens to vocabulary."""
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
    
    def _tokenize_code(self, code: str) -> List[str]:
        """
        Split code into tokens.
        
        This is a simple tokenizer that:
        1. Handles strings and comments specially
        2. Splits on operators and punctuation
        3. Preserves whitespace for indentation
        """
        tokens = []
        i = 0
        
        while i < len(code):
            # Newlines (important for code structure)
            if code[i] == '\n':
                tokens.append('\n')
                i += 1
                continue
            
            # Whitespace (collapse multiple spaces, but keep for indentation)
            if code[i] in ' \t':
                space = ''
                while i < len(code) and code[i] in ' \t':
                    space += code[i]
                    i += 1
                if space:
                    # Simplify: just use <SPACE> or <INDENT>
                    if len(space) >= 2:
                        tokens.append('<INDENT>')
                    else:
                        tokens.append('<SPACE>')
                continue
            
            # Strings (single or double quotes)
            if code[i] in '"\'`':
                quote = code[i]
                string_content = quote
                i += 1
                while i < len(code) and code[i] != quote:
                    if code[i] == '\\' and i + 1 < len(code):
                        string_content += code[i:i+2]
                        i += 2
                    else:
                        string_content += code[i]
                        i += 1
                if i < len(code):
                    string_content += code[i]
                    i += 1
                tokens.append('<STRING>')  # Replace string content with token
                continue
            
            # Comments
            if code[i:i+2] == '//':
                while i < len(code) and code[i] != '\n':
                    i += 1
                tokens.append('<COMMENT>')
                continue
            
            if code[i:i+2] == '/*':
                while i < len(code) - 1 and code[i:i+2] != '*/':
                    i += 1
                i += 2
                tokens.append('<COMMENT>')
                continue
            
            # Operators (check longest first)
            found_op = False
            for op in self.OPERATORS:
                if code[i:i+len(op)] == op:
                    tokens.append(op)
                    i += len(op)
                    found_op = True
                    break
            if found_op:
                continue
            
            # Numbers
            if code[i].isdigit():
                num = ''
                while i < len(code) and (code[i].isdigit() or code[i] == '.'):
                    num += code[i]
                    i += 1
                tokens.append('<NUMBER>')  # Replace number with token
                continue
            
            # Identifiers/Keywords
            if code[i].isalpha() or code[i] == '_':
                word = ''
                while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                    word += code[i]
                    i += 1
                tokens.append(word)
                continue
            
            # Unknown character
            tokens.append(code[i])
            i += 1
        
        return tokens
    
    def build_vocab(self, code_samples: List[str], min_freq: int = 1):
        """
        Build vocabulary from code samples.
        
        Args:
            code_samples: List of code strings
            min_freq: Minimum frequency for a token to be included
        """
        # Count token frequencies
        counter = Counter()
        for code in code_samples:
            tokens = self._tokenize_code(code)
            counter.update(tokens)
        
        # Start with special tokens (already added)
        current_id = len(self.vocab)
        
        # Add keywords first (guaranteed to be in vocab)
        for keyword in sorted(self.KEYWORDS):
            if keyword not in self.vocab:
                self.vocab[keyword] = current_id
                self.inverse_vocab[current_id] = keyword
                current_id += 1
        
        # Add code-specific special tokens
        code_special = ['<SPACE>', '<INDENT>', '<STRING>', '<NUMBER>', '<COMMENT>', '\n']
        for token in code_special:
            if token not in self.vocab:
                self.vocab[token] = current_id
                self.inverse_vocab[current_id] = token
                current_id += 1
        
        # Add operators
        for op in self.OPERATORS:
            if op not in self.vocab:
                self.vocab[op] = current_id
                self.inverse_vocab[current_id] = op
                current_id += 1
        
        # Add most common tokens up to vocab_size
        for token, freq in counter.most_common():
            if current_id >= self.vocab_size:
                break
            if token not in self.vocab and freq >= min_freq:
                self.vocab[token] = current_id
                self.inverse_vocab[current_id] = token
                current_id += 1
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
    
    def encode(self, code: str, add_special: bool = True) -> List[int]:
        """
        Encode code string to token IDs.
        
        Args:
            code: Code string to encode
            add_special: If True, add BOS and EOS tokens
        
        Returns:
            List of token IDs
        """
        tokens = self._tokenize_code(code)
        
        ids = []
        if add_special:
            ids.append(self.vocab[self.BOS_TOKEN])
        
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.UNK_TOKEN])
        
        if add_special:
            ids.append(self.vocab[self.EOS_TOKEN])
        
        return ids
    
    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs to code string.
        
        Args:
            ids: List of token IDs
            skip_special: If True, skip BOS, EOS, PAD tokens
        
        Returns:
            Decoded string
        """
        special_ids = {
            self.vocab[self.PAD_TOKEN], 
            self.vocab[self.BOS_TOKEN], 
            self.vocab[self.EOS_TOKEN]
        }
        
        tokens = []
        for id in ids:
            if skip_special and id in special_ids:
                continue
            if id in self.inverse_vocab:
                token = self.inverse_vocab[id]
                # Convert special tokens back
                if token == '<SPACE>':
                    tokens.append(' ')
                elif token == '<INDENT>':
                    tokens.append('  ')
                elif token == '<STRING>':
                    tokens.append('"..."')
                elif token == '<NUMBER>':
                    tokens.append('0')
                elif token == '<COMMENT>':
                    tokens.append('/* ... */')
                else:
                    tokens.append(token)
            else:
                tokens.append(self.UNK_TOKEN)
        
        # Join tokens (add spaces between words)
        result = []
        for i, token in enumerate(tokens):
            if i > 0:
                prev = tokens[i-1]
                # Don't add space before punctuation
                if token not in '(){}[];,.:' and prev not in '({[':
                    if prev not in '(){}[];,.:\n' and token != '\n':
                        if not (prev in '<SPACE><INDENT>' or token in '<SPACE><INDENT>'):
                            result.append(' ')
            result.append(token)
        
        return ''.join(result)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size
            }, f, indent=2)
        print(f"Tokenizer saved to {path}")
    
    def load(self, path: str):
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"Tokenizer loaded from {path}")
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab[self.PAD_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.vocab[self.BOS_TOKEN]
    
    @property  
    def eos_token_id(self) -> int:
        return self.vocab[self.EOS_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.UNK_TOKEN]


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("=" * 50)
    print("Code Tokenizer Demo")
    print("=" * 50)
    
    # Sample JavaScript/TypeScript code
    code_samples = [
        '''function add(a, b) {
    return a + b;
}''',
        '''const greet = (name) => {
    console.log("Hello, " + name);
};''',
        '''class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(x) {
        this.result += x;
        return this;
    }
}''',
        '''async function fetchData(url) {
    try {
        const response = await fetch(url);
        return response.json();
    } catch (error) {
        console.error(error);
    }
}''',
        '''interface User {
    name: string;
    age: number;
    email?: string;
}

type UserList = User[];''',
    ]
    
    # Create tokenizer and build vocabulary
    tokenizer = Tokenizer(vocab_size=500)
    tokenizer.build_vocab(code_samples)
    
    print(f"\nVocabulary size: {len(tokenizer.vocab)}")
    print(f"Sample vocab entries: {list(tokenizer.vocab.items())[:10]}")
    
    # Test encoding
    test_code = "function test(x) { return x * 2; }"
    print(f"\n--- Encoding Test ---")
    print(f"Input: {test_code}")
    
    tokens = tokenizer._tokenize_code(test_code)
    print(f"Tokens: {tokens}")
    
    ids = tokenizer.encode(test_code)
    print(f"IDs: {ids}")
    
    # Test decoding
    decoded = tokenizer.decode(ids)
    print(f"Decoded: {decoded}")
    
    # Show special token IDs
    print(f"\n--- Special Tokens ---")
    print(f"PAD: {tokenizer.pad_token_id}")
    print(f"BOS: {tokenizer.bos_token_id}")
    print(f"EOS: {tokenizer.eos_token_id}")
    print(f"UNK: {tokenizer.unk_token_id}")
    
    print("\n✓ Tokenizer working!")
