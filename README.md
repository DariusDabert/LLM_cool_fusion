# Common Word-Level Boundaries for Multi-LLM Text Generation

This project demonstrates a method for generating text segments from multiple language models (LLMs) without training.


# TODO List :
The following points to achieve in the projects:
   - draw a tree with possibilities and perplexity per tokens
   - run experiments MMLU
   - try with a small model and a big one

## Overview

In multi-LLM fusion, each source LLM generates text segments token-by-token. However, the tokenizers for different models may have different definitions of word boundaries. Our solution:
- Generates a text segment for each model.
- Iteratively appends tokens until the candidate text ends on a common word-level boundary across all tokenizers.
- Ensures that each model's final segment respects both its own tokenizer’s boundaries and the common boundaries shared with the other models.

## Project Structure

- **`source`**: Contains the source code to implement cool fusion
- **`experiments`**: This folder contains executable notebooks to reproduce experiments

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/)
- Transformers library (or any similar library providing `.encode()`, `.decode()`, and `.generate()` methods for LLMs)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/DariusDanert/multi-llm-text-segmentation.git
   cd multi-llm-text-segmentation
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate  # Windows
   ```

3. Install the required packages:
   ```bash
   pip install torch transformers
   ```

## Usage

```python
from source.cool_fusion import CoolFusion

# Example dictionaries mapping model names to model/tokenizer objects
models = {
    "llama3": llama3_model,
    "phi3": phi3_model
}

tokenizers = {
    "llama3": llama3_tokenizer,
    "phi3": phi3_tokenizer
}

context = "Once upon a time"
segments = generate_text_segments(models, tokenizers, context, max_length=50)

for name, text in segments.items():
    print(f"Model {name} generated: {text}")
```

## How It Works

1. **Initialization:**  
   Each model's context is encoded using its tokenizer.

2. **Token-by-Token Generation:**  
   Each model generates one token at a time. The new token is appended to the running context.

3. **Decoding & Boundary Check:**  
   The candidate text is decoded for each model and split into words. The function computes the longest common prefix (across all tokenizers) to ensure that the generated text segment ends at a common word boundary.

4. **Finalization:**  
   If the candidate text exactly matches the common prefix or ends with a space (indicating a complete word), it is finalized as the model's output segment. If not, a fallback trimming approach is applied.
