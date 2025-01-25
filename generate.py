# generate.py
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from config import config

def generate_text(model_name, prompt, max_length=50):
    """Generate text using a specified model."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained(f"{config['output_dir']}/{model_name}")
    model.eval()
    model.to(config["device"])

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config["device"])
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = "Deep in the jungle, Mowgli"  # Your starting prompt

    for model_size in config["model_sizes"]:
        model_name = model_size["name"]
        print(f"Generating text with model: {model_name}")

        try:
            generated_text = generate_text(model_name, prompt)
            print(f"\nModel: {model_name}\nGenerated Text: {generated_text}\n")
        except Exception as e:
            print(f"Error generating text with model {model_name}: {e}")
