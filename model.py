from transformers import GPT2LMHeadModel, GPT2Config

def create_model(model_size):
    """Create a GPT2 model with the specified size."""
    config = GPT2Config(
        vocab_size=50257,       # GPT-2 tokenizer vocab size
        n_positions=128,        # Context length
        n_embd=model_size["n_embd"],
        n_layer=model_size["n_layer"],
        n_head=model_size["n_head"]
    )
    model = GPT2LMHeadModel(config)
    model.resize_token_embeddings(50257)
    return model
