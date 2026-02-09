"""
English text corpus for training the tokenizer and model.
This helps the AI understand natural language instructions.
"""

ENGLISH_SAMPLES = [
    # Greetings & Common Chat
    "Hello, how can I help you today?",
    "I am an AI programming assistant.",
    "Can you help me write some code?",
    "Sure! I can generate JavaScript and TypeScript code for you.",
    "What is your name?",
    "I am PyCodeAI, built from scratch in Python.",
    "Nice to meet you.",
    "Goodbye!",
    
    # Technical Questions
    "How do I write a function in JavaScript?",
    "What is the difference between var, let, and const?",
    "Explain how Promises work.",
    "Show me an example of a React component.",
    "How can I sort an array of numbers?",
    "What is a closure?",
    "How do I handle errors in async functions?",
    
    # Technical Explanations
    "A function is a block of code designed to perform a particular task.",
    "React is a JavaScript library for building user interfaces.",
    "TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.",
    "Node.js is a JavaScript runtime built on Chrome's V8 JavaScript engine.",
    "An array is a special variable, which can hold more than one value.",
    "JSON stands for JavaScript Object Notation.",
    "API stands for Application Programming Interface.",
    
    # Instructions
    "Write a function to add two numbers.",
    "Create a class for a User.",
    "Implement a binary search algorithm.",
    "Make an HTTP request using fetch.",
    "Debug this code snippet.",
    "Refactor this function to be more efficient.",
    
    # Mixed Code/Text
    "Here is the function you requested:",
    "To install dependencies, run npm install.",
    "You can use console.log() to print debugging information.",
    "The syntax for a for loop is: for (initialization; condition; increment) { ... }",
    "Don't forget to export your module using module.exports or export default.",
]

def get_english_samples():
    """Get all English text samples."""
    return ENGLISH_SAMPLES
