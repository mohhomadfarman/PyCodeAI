"""
PyCodeAI Chat UI - Browser-based chat interface for testing your model.

Usage:
    python chat_ui.py                     # Auto-detect device
    python chat_ui.py --device gpu        # Force GPU
    python chat_ui.py --port 8080         # Custom port
    python chat_ui.py --model best_model.npz  # Load specific model
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── HTML Chat Interface ───────────────────────────────────────────────────────

CHAT_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyCodeAI Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0d1117;
    color: #e6edf3;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* Header */
  .header {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-shrink: 0;
  }
  .header h1 {
    font-size: 18px;
    font-weight: 600;
    color: #58a6ff;
  }
  .header .status {
    font-size: 12px;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .header .status .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3fb950;
  }

  /* Settings bar */
  .settings {
    background: #161b22;
    border-bottom: 1px solid #30363d;
    padding: 8px 20px;
    display: flex;
    gap: 16px;
    align-items: center;
    flex-shrink: 0;
    flex-wrap: wrap;
  }
  .settings label {
    font-size: 12px;
    color: #8b949e;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .settings input, .settings select {
    background: #0d1117;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
    width: 70px;
  }
  .settings select { width: auto; }

  /* Chat area */
  .chat-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .message {
    display: flex;
    gap: 12px;
    max-width: 85%;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .message.user { align-self: flex-end; flex-direction: row-reverse; }
  .message.bot { align-self: flex-start; }

  .avatar {
    width: 32px; height: 32px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
  }
  .message.user .avatar { background: #1f6feb; }
  .message.bot .avatar { background: #238636; }

  .bubble {
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.5;
    font-size: 14px;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .message.user .bubble {
    background: #1f6feb;
    border-bottom-right-radius: 4px;
  }
  .message.bot .bubble {
    background: #21262d;
    border: 1px solid #30363d;
    border-bottom-left-radius: 4px;
  }

  /* Code blocks inside messages */
  .bubble code {
    background: #161b22;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
  }
  .bubble pre {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px;
    margin: 8px 0;
    overflow-x: auto;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
    font-size: 13px;
  }

  /* Typing indicator */
  .typing {
    display: flex;
    gap: 4px;
    padding: 4px 0;
  }
  .typing span {
    width: 6px; height: 6px;
    background: #8b949e;
    border-radius: 50%;
    animation: bounce 1.4s ease-in-out infinite;
  }
  .typing span:nth-child(2) { animation-delay: 0.16s; }
  .typing span:nth-child(3) { animation-delay: 0.32s; }
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }

  /* Input area */
  .input-area {
    background: #161b22;
    border-top: 1px solid #30363d;
    padding: 16px 20px;
    display: flex;
    gap: 12px;
    flex-shrink: 0;
  }
  .input-area textarea {
    flex: 1;
    background: #0d1117;
    border: 1px solid #30363d;
    color: #e6edf3;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    outline: none;
    min-height: 44px;
    max-height: 120px;
    line-height: 1.4;
    transition: border-color 0.2s;
  }
  .input-area textarea:focus { border-color: #58a6ff; }
  .input-area textarea::placeholder { color: #484f58; }

  .send-btn {
    background: #238636;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0 20px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
    flex-shrink: 0;
  }
  .send-btn:hover { background: #2ea043; }
  .send-btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

  /* Welcome screen */
  .welcome {
    text-align: center;
    padding: 60px 20px;
    color: #8b949e;
  }
  .welcome h2 { color: #58a6ff; font-size: 24px; margin-bottom: 12px; }
  .welcome p { margin-bottom: 8px; font-size: 14px; }
  .welcome .examples {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 20px;
  }
  .welcome .example-btn {
    background: #21262d;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 13px;
    transition: border-color 0.2s;
  }
  .welcome .example-btn:hover { border-color: #58a6ff; }

  /* Timer */
  .gen-time {
    font-size: 11px;
    color: #484f58;
    margin-top: 4px;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #484f58; }
</style>
</head>
<body>

<div class="header">
  <h1>PyCodeAI Chat</h1>
  <div class="status">
    <span class="dot"></span>
    <span id="model-info">Loading...</span>
  </div>
</div>

<div class="settings">
  <label>Temperature
    <input type="number" id="temperature" value="0.7" min="0" max="2" step="0.1">
  </label>
  <label>Max Tokens
    <input type="number" id="max-tokens" value="100" min="10" max="500" step="10">
  </label>
  <label>Top-K
    <input type="number" id="top-k" value="50" min="1" max="200" step="5">
  </label>
  <label>Top-P
    <input type="number" id="top-p" value="0.9" min="0.1" max="1.0" step="0.05">
  </label>
</div>

<div class="chat-area" id="chat-area">
  <div class="welcome" id="welcome">
    <h2>PyCodeAI</h2>
    <p>Your AI code assistant, built from scratch.</p>
    <p>Type a prompt below or try an example:</p>
    <div class="examples">
      <button class="example-btn" onclick="useExample(this)">function fibonacci(n) {</button>
      <button class="example-btn" onclick="useExample(this)">const add = (a, b) =></button>
      <button class="example-btn" onclick="useExample(this)">class Stack {</button>
      <button class="example-btn" onclick="useExample(this)">async function fetchData(url) {</button>
      <button class="example-btn" onclick="useExample(this)">User: Hello!</button>
      <button class="example-btn" onclick="useExample(this)">User: Write a function to sort an array.</button>
    </div>
  </div>
</div>

<div class="input-area">
  <textarea id="input" placeholder="Type your message... (Shift+Enter for new line)" rows="1"></textarea>
  <button class="send-btn" id="send-btn" onclick="sendMessage()">Send</button>
</div>

<script>
const chatArea = document.getElementById('chat-area');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send-btn');
const welcome = document.getElementById('welcome');

// Auto-resize textarea
input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 120) + 'px';
});

// Enter to send, Shift+Enter for newline
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Load model info
fetch('/api/info')
  .then(r => r.json())
  .then(info => {
    document.getElementById('model-info').textContent =
      `${info.params} params | ${info.device} | seq_len=${info.seq_len}`;
  })
  .catch(() => {
    document.getElementById('model-info').textContent = 'Connected';
  });

function useExample(btn) {
  input.value = btn.textContent;
  input.focus();
  sendMessage();
}

function addMessage(text, role) {
  if (welcome) welcome.style.display = 'none';

  const msg = document.createElement('div');
  msg.className = 'message ' + role;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? 'U' : 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatArea.appendChild(msg);
  chatArea.scrollTop = chatArea.scrollHeight;
  return bubble;
}

function addTyping() {
  if (welcome) welcome.style.display = 'none';

  const msg = document.createElement('div');
  msg.className = 'message bot';
  msg.id = 'typing-indicator';

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = 'AI';

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  chatArea.appendChild(msg);
  chatArea.scrollTop = chatArea.scrollHeight;
}

function removeTyping() {
  const el = document.getElementById('typing-indicator');
  if (el) el.remove();
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;

  // Disable input while generating
  input.value = '';
  input.style.height = 'auto';
  sendBtn.disabled = true;

  addMessage(text, 'user');
  addTyping();

  const startTime = Date.now();

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: text,
        temperature: parseFloat(document.getElementById('temperature').value),
        max_tokens: parseInt(document.getElementById('max-tokens').value),
        top_k: parseInt(document.getElementById('top-k').value),
        top_p: parseFloat(document.getElementById('top-p').value),
      })
    });

    const data = await res.json();
    removeTyping();

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    if (data.error) {
      const bubble = addMessage('Error: ' + data.error, 'bot');
      bubble.style.color = '#f85149';
    } else {
      const bubble = addMessage(data.generated, 'bot');
      // Add generation time
      const timeEl = document.createElement('div');
      timeEl.className = 'gen-time';
      timeEl.textContent = `${elapsed}s | ${data.tokens_generated} tokens`;
      bubble.appendChild(timeEl);
    }
  } catch (err) {
    removeTyping();
    const bubble = addMessage('Connection error: ' + err.message, 'bot');
    bubble.style.color = '#f85149';
  }

  sendBtn.disabled = false;
  input.focus();
}

input.focus();
</script>
</body>
</html>"""


# ─── HTTP Server ────────────────────────────────────────────────────────────────

class ChatHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the chat UI."""

    generator = None
    model_info = {}

    def log_message(self, format, *args):
        """Suppress default logging (too noisy)."""
        pass

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self._send_html(CHAT_HTML)
        elif self.path == '/api/info':
            self._send_json(self.model_info)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/generate':
            try:
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                params = json.loads(body.decode('utf-8'))

                prompt = params.get('prompt', '')
                temperature = float(params.get('temperature', 0.7))
                max_tokens = int(params.get('max_tokens', 100))
                top_k = int(params.get('top_k', 50))
                top_p = float(params.get('top_p', 0.9))

                if not prompt:
                    self._send_json({'error': 'Empty prompt'}, 400)
                    return

                # Generate
                start = time.time()
                result = self.generator.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                elapsed = time.time() - start

                # Extract the generated part (after prompt)
                generated = result[len(prompt):] if result.startswith(prompt) else result

                prompt_tokens = len(self.generator.tokenizer.encode(prompt, add_special=False))
                total_tokens = len(self.generator.tokenizer.encode(result, add_special=False))

                self._send_json({
                    'generated': generated,
                    'full_text': result,
                    'tokens_generated': total_tokens - prompt_tokens,
                    'time': round(elapsed, 2),
                })

            except Exception as e:
                self._send_json({'error': str(e)}, 500)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()


# ─── Main ───────────────────────────────────────────────────────────────────────

def start_chat_server(
    model_path='model.npz',
    tokenizer_path='tokenizer.json',
    device='auto',
    port=5000,
):
    """Load model and start the chat server."""
    # IMPORTANT: Set backend BEFORE importing model classes
    # so all modules pick up the correct xp (numpy vs cupy)
    from src.core import backend as _backend
    from src.core.backend import GPU_AVAILABLE, use_gpu, use_cpu, is_gpu

    # For inference, CPU is faster than GPU (small batch, no CuPy overhead)
    # GPU only helps during training with large batches
    if device == 'gpu':
        use_gpu()
    elif device == 'cpu':
        use_cpu()
    else:
        # auto: use CPU for chat (faster inference for small models)
        use_cpu()

    # Now import model classes (they'll use the correct backend)
    from src.models.gpt import GPT, GPTConfig
    from src.tokenizer.tokenizer import Tokenizer
    from src.inference.generator import CodeGenerator

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = Tokenizer(vocab_size=5000)
    tokenizer.load(tokenizer_path)
    print(f"   Vocabulary: {len(tokenizer.vocab)} tokens")

    # Load model
    print("\n2. Loading model...")
    model = GPT.load_from_dir(model_path)
    print(f"   Parameters: {model.num_parameters():,}")
    print(f"   Config: embed={model.config.embed_dim}, heads={model.config.num_heads}, "
          f"layers={model.config.num_layers}, seq_len={model.config.max_seq_len}")

    # Create generator
    generator = CodeGenerator(model, tokenizer)

    # Setup handler
    ChatHandler.generator = generator
    ChatHandler.model_info = {
        'params': f"{model.num_parameters():,}",
        'device': 'GPU' if is_gpu() else 'CPU',
        'embed_dim': model.config.embed_dim,
        'num_heads': model.config.num_heads,
        'num_layers': model.config.num_layers,
        'seq_len': model.config.max_seq_len,
        'vocab_size': len(tokenizer.vocab),
    }

    # Start server
    server = HTTPServer(('127.0.0.1', port), ChatHandler)
    url = f"http://127.0.0.1:{port}"

    print(f"\n{'=' * 50}")
    print(f"  PyCodeAI Chat is running!")
    print(f"  Open: {url}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'=' * 50}\n")

    # Open browser after short delay
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyCodeAI Chat UI')
    parser.add_argument('--model', default='model.npz', help='Model file path')
    parser.add_argument('--tokenizer', default='tokenizer.json', help='Tokenizer file path')
    parser.add_argument('--device', choices=['auto', 'gpu', 'cpu'], default='auto')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    args = parser.parse_args()

    start_chat_server(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        device=args.device,
        port=args.port,
    )
