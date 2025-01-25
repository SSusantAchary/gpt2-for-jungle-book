# config.py
config = {
    "dataset_path": "/kaggle/working/jungle_book.txt",
    "tokenized_data": {
        "train": "/kaggle/working/train_data.pt",
        "val": "/kaggle/working/val_data.pt"
    },
    "context_length": 128,
    "batch_size": 64,
    "learning_rate": 5e-4,
    "epochs": 5,
    "device": "cuda",
    "huggingface_user": "Susant-Achary",  # Replace with your Hugging Face username
    "model_repo_base": "text-gen-model",  # Base name for repositories
    #"use_parallel": True,  # Enable multi-GPU
    "model_sizes": [
        {"name": "15M", "n_layer": 8, "n_embd": 512, "n_head": 8},
        {"name": "22M", "n_layer": 10, "n_embd": 640, "n_head": 8},
        {"name": "37M", "n_layer": 12, "n_embd": 768, "n_head": 12}
    ],
    "output_dir": "/kaggle/working/models"
}
