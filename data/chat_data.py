# Chat Training Data
# This file contains conversation examples to teach the AI how to be a helpful assistant.

def get_chat_samples():
    """Generate a rich dataset of conversations."""
    samples = []
    
    # --- 1. General Small Talk ---
    small_talk = [
        ("Hello!", "Hello! I am PyCodeAI, your coding assistant. How can I help you today?"),
        ("Who are you?", "I am PyCodeAI, an AI built from scratch to help you write code."),
        ("What can you do?", "I can generate code, explain concepts, and help you debug issues."),
        ("Hi there", "Hi! Ready to write some code?"),
        ("Good morning", "Good morning! Let's build something awesome today."),
    ]
    
    # --- 2. Coding Requests (Python) ---
    python_tasks = [
        ("Write a function to add numbers.", 
         "Sure! Here is a Python function to add two numbers:\n```python\ndef add(a, b):\n    return a + b\n```"),
         
        ("How do I read a file in Python?", 
         "You can use the built-in `open` function:\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"),
         
        ("Create a class for a Car.", 
         "Here is a simple Car class:\n```python\nclass Car:\n    def __init__(self, make, model):\n        self.make = make\n        self.model = model\n        \n    def drive(self):\n        print(f'{self.make} {self.model} is driving!')\n```"),
         
        ("Explain list comprehension.", 
         "List comprehension is a concise way to create lists. For example:\n```python\nsquares = [x**2 for x in range(10)]\n```\nIt's faster and more readable than a loop."),
    ]
    
    # --- 3. Coding Requests (JavaScript) ---
    js_tasks = [
        ("How do I filter an array in JS?", 
         "Use the `.filter()` method:\n```javascript\nconst numbers = [1, 2, 3, 4];\nconst evens = numbers.filter(n => n % 2 === 0);\n```"),
         
        ("Write a React button component.", 
         "Here is a simple button component:\n```javascript\nfunction Button({ onClick, children }) {\n    return (\n        <button onClick={onClick} className=\"btn\">\n            {children}\n        </button>\n    );\n}\n```"),
         
        ("Explain async/await.", 
         "Async/await makes asynchronous code look synchronous. Example:\n```javascript\nasync function getData() {\n    const res = await fetch('/api/data');\n    const data = await res.json();\n    return data;\n}\n```"),
    ]
    
    # --- 4. Debugging & Explanations ---
    explanations = [
        ("What is a deadlock?", "A deadlock happens when two processes are waiting for each other to release a resource, so neither can proceed."),
        ("What is recursion?", "Recursion is when a function calls itself. You need a base case to stop it from running forever."),
        ("Why is my loop infinite?", "Check your condition! If the loop variable isn't updated correctly, the condition `i < 10` might always be true."),
        ("What is the difference between list and tuple?", "Lists are mutable (changeable), while tuples are immutable (cannot be changed after creation)."),
    ]
    
    # Combine and Format
    all_pairs = small_talk + python_tasks + js_tasks + explanations
    
    # Create variations for each pair
    for user, assistant in all_pairs:
        # Format 1: Standard Chat
        samples.append(f"User: {user}\nAssistant: {assistant}")
        
        # Format 2: Q&A
        samples.append(f"Q: {user}\nA: {assistant}")
        
        # Format 3: Instruction (System style)
        samples.append(f"Instruction: {user}\nResponse: {assistant}")
        
    # Replicate to ensure volume (we want ~1000 samples for a good epoch)
    # We have ~20 base samples * 3 formats = 60 samples.
    # Replicate 20 times -> 1200 samples.
    
    final_dataset = samples * 20
    
    return final_dataset
