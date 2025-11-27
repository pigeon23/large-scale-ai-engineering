from transformers import AutoTokenizer
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/cosmopedia", "stories", split="train", num_proc=12)
print(len(ds))
sample = ds[0]
for k, v in sample.items():
    print(f"{k}: {v}")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")

# Example of converting text to tokens and back
text = "Hello, I am a language model."

# Convert text to tokens (IDs)
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")

# Convert tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")
