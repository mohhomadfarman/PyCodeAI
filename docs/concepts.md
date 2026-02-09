# AI Concepts Guide 🧠

Welcome to the world of AI! Here is a simple explanation of the terms you see in PyCodeAI.

## 1. Tokens & Vocabulary
**Tokens** are the atoms of language for an AI.
*   Imagine reading a sentence letter by letter. That's hard!
*   Imagine reading it word by word. Better.
*   AI uses **Tokens**, which can be whole words (`apple`) or parts of words (`ing`, `ed`).

**Vocabulary Size** is how many unique tokens the AI knows.
*   **Small (e.g., 1000):** The AI has a limited dictionary. It might see "React" as `<UNK>` (Unknown).
*   **Large (e.g., 5000+):** The AI knows more specific words like `useEffect`, `div`, `const`.
*   **Too Large:** The model becomes slow and hard to train.

## 2. Loss
**Loss** is the "Error Score".
*   When training starts, the AI guesses randomly. The Loss is high (e.g., 5.0).
*   As it learns patterns ("after `function` comes a `name`"), the Loss goes down.
*   **Goal:** Get the loss as close to 0 as possible (usually ~1.0-2.0 is great for code).

## 3. Embeddings
**Embeddings** are how AI understands meaning.
*   Computers only know numbers, not words.
*   An embedding converts a token (like `cat`) into a list of numbers (e.g., `[0.1, -0.5, 0.8...]`).
*   **Magic:** Words with similar meanings (`cat`, `dog`) end up with similar number lists!
*   In PyCodeAI, we use `embed-dim=128`, meaning each word is described by 128 numbers.

## 4. Temperature (Creativity)
**Temperature** controls how "wild" the AI's guesses are.
*   **Low (0.1 - 0.4):** The AI plays it safe. Good for strict code.
*   **High (0.7 - 1.0):** The AI takes risks. Good for creative writing or brainstorming.
*   **Too High:** The AI starts babbling nonsense.

## 5. Structured Data
We organize data to help the AI learn context:
*   **Project:** "This code is from React."
*   **Language:** "This is JavaScript."
*   **Path:** "This is a configuration file."

By adding these "tags" (Metadata) to the training data, the AI learns to write code that fits the specific project style!
