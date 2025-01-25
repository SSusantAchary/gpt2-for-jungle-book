import re
import torch
from transformers import GPT2Tokenizer
from config import config

def clean_text(text):
    """Clean the raw text by removing unnecessary spaces and newlines."""
    text = re.sub(r"\n+", "\n", text)  # Replace multiple newlines with one
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    return text.strip()

def tokenize_and_save():
    """Tokenize the dataset and save train/val splits."""
    with open(config["dataset_path"], "r", encoding="utf-8") as file:
        text = file.read()

    # Clean the text
    cleaned_text = clean_text(text)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Tokenize and split
    tokenized_text = tokenizer.encode(cleaned_text, add_special_tokens=True)
    train_size = int(0.9 * len(tokenized_text))
    train_data = tokenized_text[:train_size]
    val_data = tokenized_text[train_size:]

    # Save datasets
    torch.save(train_data, config["tokenized_data"]["train"])
    torch.save(val_data, config["tokenized_data"]["val"])
    print("Data tokenized and saved!")

if __name__ == "__main__":
    tokenize_and_save()
