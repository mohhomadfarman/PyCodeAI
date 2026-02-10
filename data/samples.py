"""
Sample JavaScript/TypeScript code for training.

This module contains sample code to train your AI on.
For better results, add more code samples!

The more diverse and high-quality code you add,
the better your model will learn.
"""

# Sample JavaScript/TypeScript code for training
JAVASCRIPT_SAMPLES = [
    # Basic functions
    '''function add(a, b) {
    return a + b;
}''',
    
    '''function subtract(a, b) {
    return a - b;
}''',
    
    '''function multiply(a, b) {
    return a * b;
}''',
    
    '''function divide(a, b) {
    if (b === 0) {
        throw new Error("Cannot divide by zero");
    }
    return a / b;
}''',
    
    # Arrow functions
    '''const square = (x) => x * x;''',
    
    '''const double = (x) => x * 2;''',
    
    '''const greet = (name) => {
    return "Hello, " + name + "!";
};''',
    
    '''const isEven = (n) => n % 2 === 0;''',
    
    '''const isOdd = (n) => n % 2 !== 0;''',
    
    # Array methods
    '''const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(x => x * 2);''',
    
    '''const numbers = [1, 2, 3, 4, 5];
const evens = numbers.filter(x => x % 2 === 0);''',
    
    '''const numbers = [1, 2, 3, 4, 5];
const sum = numbers.reduce((acc, x) => acc + x, 0);''',
    
    '''const fruits = ["apple", "banana", "cherry"];
fruits.forEach(fruit => {
    console.log(fruit);
});''',
    
    # Loops
    '''for (let i = 0; i < 10; i++) {
    console.log(i);
}''',
    
    '''let i = 0;
while (i < 10) {
    console.log(i);
    i++;
}''',
    
    '''const items = [1, 2, 3];
for (const item of items) {
    console.log(item);
}''',
    
    # Conditionals
    '''function checkAge(age) {
    if (age < 18) {
        return "minor";
    } else if (age < 65) {
        return "adult";
    } else {
        return "senior";
    }
}''',
    
    '''const status = age >= 18 ? "adult" : "minor";''',
    
    # Classes
    '''class Animal {
    constructor(name) {
        this.name = name;
    }
    
    speak() {
        console.log(this.name + " makes a sound.");
    }
}''',
    
    '''class Dog extends Animal {
    constructor(name, breed) {
        super(name);
        this.breed = breed;
    }
    
    speak() {
        console.log(this.name + " barks.");
    }
}''',
    
    '''class Calculator {
    constructor() {
        this.result = 0;
    }
    
    add(x) {
        this.result += x;
        return this;
    }
    
    subtract(x) {
        this.result -= x;
        return this;
    }
    
    multiply(x) {
        this.result *= x;
        return this;
    }
    
    getResult() {
        return this.result;
    }
}''',
    
    # Async/Await
    '''async function fetchData(url) {
    const response = await fetch(url);
    const data = await response.json();
    return data;
}''',
    
    '''async function getData() {
    try {
        const result = await fetchData("/api/data");
        console.log(result);
    } catch (error) {
        console.error("Error:", error);
    }
}''',
    
    '''const fetchUser = async (id) => {
    const response = await fetch("/api/users/" + id);
    if (!response.ok) {
        throw new Error("User not found");
    }
    return response.json();
};''',
    
    # Promises
    '''function delay(ms) {
    return new Promise(resolve => {
        setTimeout(resolve, ms);
    });
}''',
    
    '''const getData = () => {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve({ data: "success" });
        }, 1000);
    });
};''',
    
    # Object manipulation
    '''const user = {
    name: "John",
    age: 30,
    email: "john@example.com"
};''',
    
    '''const { name, age } = user;''',
    
    '''const newUser = { ...user, city: "New York" };''',
    
    '''const keys = Object.keys(user);
const values = Object.values(user);''',
    
    # String methods
    '''const str = "Hello, World!";
const upper = str.toUpperCase();
const lower = str.toLowerCase();''',
    
    '''const str = "Hello, World!";
const parts = str.split(", ");
const joined = parts.join(" - ");''',
    
    '''const str = "  hello  ";
const trimmed = str.trim();''',
    
    # Common patterns
    '''function debounce(fn, delay) {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
}''',
    
    '''function throttle(fn, limit) {
    let lastCall = 0;
    return function(...args) {
        const now = Date.now();
        if (now - lastCall >= limit) {
            lastCall = now;
            fn.apply(this, args);
        }
    };
}''',
    
    '''function memoize(fn) {
    const cache = new Map();
    return function(...args) {
        const key = JSON.stringify(args);
        if (cache.has(key)) {
            return cache.get(key);
        }
        const result = fn.apply(this, args);
        cache.set(key, result);
        return result;
    };
}''',
    
    # Algorithms
    '''function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}''',
    
    '''function factorial(n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}''',
    
    '''function binarySearch(arr, target) {
    let left = 0;
    let right = arr.length - 1;
    
    while (left <= right) {
        const mid = Math.floor((left + right) / 2);
        if (arr[mid] === target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}''',
    
    '''function quickSort(arr) {
    if (arr.length <= 1) return arr;
    
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const middle = arr.filter(x => x === pivot);
    const right = arr.filter(x => x > pivot);
    
    return [...quickSort(left), ...middle, ...quickSort(right)];
}''',
    
    '''function bubbleSort(arr) {
    const n = arr.length;
    for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            }
        }
    }
    return arr;
}''',
    
    # DOM manipulation
    '''const element = document.getElementById("myElement");
element.textContent = "Hello!";''',
    
    '''document.querySelector(".button").addEventListener("click", () => {
    console.log("Button clicked!");
});''',
    
    '''const items = document.querySelectorAll(".item");
items.forEach(item => {
    item.classList.add("active");
});''',

    # --- NEW SAMPLES ---
    
    # Error handling
    '''try {
    const data = JSON.parse(jsonString);
    console.log(data);
} catch (error) {
    console.error("Invalid JSON:", error.message);
}''',
    
    # Event handling
    '''window.addEventListener("resize", () => {
    const width = window.innerWidth;
    const height = window.innerHeight;
    console.log(`Resized to ${width}x${height}`);
});''',
    
    # Modern JS features
    '''const merged = { ...obj1, ...obj2 };''',
    
    '''const cloned = [...originalArray];''',
    
    '''const [first, second, ...rest] = array;''',
    
    '''const { name, age, address: { city } } = person;''',
    
    # Modules
    '''import { useState, useEffect } from "react";

function Counter() {
    const [count, setCount] = useState(0);
    
    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);
    
    return count;
}''',
    
    '''export default function main() {
    console.log("App started");
}''',
    
    # Date handling
    '''const now = new Date();
const year = now.getFullYear();
const month = now.getMonth() + 1;
const day = now.getDate();''',
    
    '''function formatDate(date) {
    return date.toISOString().split("T")[0];
}''',
    
    # Math utilities
    '''function getRandomInt(min, max) {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min) + min);
}''',
    
    '''const hypotenuse = (a, b) => Math.sqrt(a*a + b*b);''',
    
    # String manipulation
    '''const slugify = (text) => {
    return text.toString().toLowerCase()
        .replace(/\s+/g, "-")           // Replace spaces with -
        .replace(/[^\w\-]+/g, "")       // Remove all non-word chars
        .replace(/\-\-+/g, "-")         // Replace multiple - with single -
        .replace(/^-+/, "")             // Trim - from start of text
        .replace(/-+$/, "");            // Trim - from end of text
};''',
    
    '''function capitalize(str) {
    if (!str) return "";
    return str.charAt(0).toUpperCase() + str.slice(1);
}''',
]

TYPESCRIPT_SAMPLES = [
    # Type annotations
    '''function add(a: number, b: number): number {
    return a + b;
}''',
    
    '''const greet = (name: string): string => {
    return "Hello, " + name;
};''',
    
    # Interfaces
    '''interface User {
    name: string;
    age: number;
    email?: string;
}''',
    
    '''interface Product {
    id: number;
    name: string;
    price: number;
    inStock: boolean;
}''',
    
    '''interface ApiResponse<T> {
    data: T;
    status: number;
    message: string;
}''',
    
    # Types
    '''type Status = "pending" | "approved" | "rejected";''',
    
    '''type Point = {
    x: number;
    y: number;
};''',
    
    '''type UserList = User[];''',
    
    '''type Nullable<T> = T | null;''',
    
    # Generics
    '''function identity<T>(arg: T): T {
    return arg;
}''',
    
    '''function getFirst<T>(arr: T[]): T | undefined {
    return arr[0];
}''',
    
    '''class Stack<T> {
    private items: T[] = [];
    
    push(item: T): void {
        this.items.push(item);
    }
    
    pop(): T | undefined {
        return this.items.pop();
    }
}''',
    
    # Classes with types
    '''class UserService {
    private users: User[] = [];
    
    addUser(user: User): void {
        this.users.push(user);
    }
    
    getUser(name: string): User | undefined {
        return this.users.find(u => u.name === name);
    }
    
    getAllUsers(): User[] {
        return this.users;
    }
}''',
    
    '''class HttpClient {
    private baseUrl: string;
    
    constructor(baseUrl: string) {
        this.baseUrl = baseUrl;
    }
    
    async get<T>(endpoint: string): Promise<T> {
        const response = await fetch(this.baseUrl + endpoint);
        return response.json() as Promise<T>;
    }
    
    async post<T>(endpoint: string, data: object): Promise<T> {
        const response = await fetch(this.baseUrl + endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });
        return response.json() as Promise<T>;
    }
}''',
    
    # Enums
    '''enum Direction {
    Up = "UP",
    Down = "DOWN",
    Left = "LEFT",
    Right = "RIGHT"
}''',
    
    '''enum HttpStatus {
    OK = 200,
    Created = 201,
    BadRequest = 400,
    NotFound = 404,
    InternalError = 500
}''',
    
    # Async with types
    '''async function fetchUser(id: number): Promise<User> {
    const response = await fetch("/api/users/" + id);
    const data: User = await response.json();
    return data;
}''',
    
    '''interface FetchOptions {
    method?: string;
    headers?: Record<string, string>;
    body?: string;
}

async function fetchData<T>(url: string, options?: FetchOptions): Promise<T> {
    const response = await fetch(url, options);
    return response.json() as Promise<T>;
}''',
]

# Combine all samples
ALL_SAMPLES = JAVASCRIPT_SAMPLES + TYPESCRIPT_SAMPLES


# --- INSTRUCT TUNING DATA ---
# This teaches the model to answer questions like a chat bot

def get_instruct_samples():
    """Generate instruction-response pairs."""
    samples = []
    
    # Templates for questions
    prompts = [
        ("Write a function to add two numbers.", 
         "function add(a, b) {\n    return a + b;\n}"),
        
        ("Create a function that subtracts numbers.", 
         "function subtract(a, b) {\n    return a - b;\n}"),
         
        ("How do I calculate factorial in JS?", 
         "function factorial(n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}"),
         
        ("Write a React component for a button.", 
         "function Button({ label, onClick }) {\n    return <button onClick={onClick}>{label}</button>;\n}"),
         
        ("Show me how to filter an array.", 
         "const filtered = array.filter(item => item > 10);"),
         
        ("Explain how to use map.", 
         "const mapped = array.map(item => item * 2);"),
         
        ("Create a class for a person.", 
         "class Person {\n    constructor(name) {\n        this.name = name;\n    }\n}"),
    ]
    
    # Create variations
    for prompt, code in prompts:
        # Style 1: Q&A
        samples.append(f"Q: {prompt}\nA: {code}")
        
        # Style 2: Chat
        samples.append(f"User: {prompt}\nAssistant: Sure! Here is the code:\n{code}")
        
        # Style 3: Comment
        samples.append(f"// {prompt}\n{code}")
        
        # Style 4: Direct request
        samples.append(f"{prompt}\n```javascript\n{code}\n```")

    return samples


import os
import re

# ... existing code ...

# ... existing code ...
# ... existing code ...
from src.data.english_corpus import get_english_samples
from data.chat_data import get_chat_samples

def get_training_data():
    """Get all training samples."""
    
    # 1. Base samples (hardcoded code)
    data = ALL_SAMPLES.copy()
    
    # 2. English samples (natural language)
    # Replicate to ensure high frequency for vocab building
    data.extend(get_english_samples() * 100)
    
    # 3. Instruct samples (mixed chat/code)
    data.extend(get_instruct_samples())
    
    # 4. Chat samples (NEW)
    data.extend(get_chat_samples())
    
    # 5. Old Crawled data (flat file)
    # ... (rest of function) ...
    crawler_file = "data/training_data.txt"
    if os.path.exists(crawler_file):
        try:
            with open(crawler_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            crawled_count = 0
            if "# FILE: " in content:
                parts = re.split(r'\n# FILE: .+\n', content)
                for part in parts:
                    if part.strip():
                        data.append(part.strip())
                        crawled_count += 1
            elif content.strip():
                 data.append(content)
                 crawled_count += 1
                 
            print(f"[OK] Loaded {crawled_count} legacy crawled files")
            
        except Exception as e:
            print(f"[WARN] Error reading {crawler_file}: {e}")
            
    # 6. New Structured Data (Folders)
    crawled_dir = "data/crawled"
    if os.path.exists(crawled_dir):
        count = 0
        for root, _, files in os.walk(crawled_dir):
            for file in files:
                if file.endswith(('.js', '.ts', '.jsx', '.tsx')):
                    try:
                        path = os.path.join(root, file)
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                            
                        # Extract metadata
                        rel_path = os.path.relpath(path, crawled_dir)
                        # data/crawled/facebook_react/src/index.js -> facebook_react
                        project_name = rel_path.split(os.sep)[0] 
                        ext = os.path.splitext(file)[1][1:] # js, ts
                        
                        # Add metadata header
                        meta_content = f"Project: {project_name}\nLanguage: {ext}\nPath: {rel_path}\n\n{content}"
                        data.append(meta_content)
                        count += 1
                    except Exception as e:
                        print(f"Error reading {path}: {e}")
        
        if count > 0:
            print(f"[OK] Loaded {count} structured crawled files")
            
    # 7. Web Articles
    data.extend(get_article_samples())

    return data


def get_article_samples():
    """Load samples from data/articles directory."""
    data = []
    articles_dir = "data/articles"
    if os.path.exists(articles_dir):
        count = 0
        for root, _, files in os.walk(articles_dir):
            for file in files:
                if file.endswith('.txt'):
                    try:
                        path = os.path.join(root, file)
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                            
                        # Metadata already in file?
                        # Or add type tag
                        meta_content = f"Type: Article\n\n{content}"
                        # Replicate valuable articles to ensure they are learned
                        data.append(meta_content) 
                        count += 1
                    except Exception as e:
                        print(f"Error reading article {path}: {e}")
        
        if count > 0:
            print(f"[OK] Loaded {count} articles")
    return data


def get_chat_only_data():
    """Get ONLY chat samples for focused training."""
    data = []
    
    # 1. Instruct samples
    data.extend(get_instruct_samples())
    
    # 2. English samples
    data.extend(get_english_samples() * 50)
    
    # 3. Rich Chat samples
    data.extend(get_chat_samples())

    # 4. Web Articles (Good for language understanding)
    data.extend(get_article_samples() * 5) # Boost articles priority
    
    print(f"[OK] Loaded {len(data)} chat/article samples")
    return data


def get_javascript_samples():
    """Get JavaScript samples only."""
    return JAVASCRIPT_SAMPLES


def get_typescript_samples():
    """Get TypeScript samples only."""
    return TYPESCRIPT_SAMPLES


if __name__ == "__main__":
    print(f"Total samples: {len(ALL_SAMPLES)}")
    
    # Check for crawled data
    data = get_training_data()
    print(f"Total including crawled: {len(data)}")
