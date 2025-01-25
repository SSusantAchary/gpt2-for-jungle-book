import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.cuda.amp import autocast, GradScaler  # For mixed-precision training
from config import config
from model import create_model

# Dataset class
class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.context_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + 1 + self.context_length], dtype=torch.long)
        return x, y

def train_model(model, train_loader, val_loader, optimizer, epochs, model_name):
    device = config["device"]
    model.to(device)

    # Mixed precision training
    scaler = GradScaler()

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast():  # Use mixed-precision for forward pass
                outputs = model(inputs, labels=targets)
                loss = outputs.loss

            scaler.scale(loss).backward()  # Scale loss for mixed precision
            scaler.step(optimizer)  # Optimizer step
            scaler.update()  # Update scaler

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                with autocast():  # Mixed precision for validation
                    outputs = model(inputs, labels=targets)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}")

    # Save the model
    output_path = f"{config['output_dir']}/{model_name}"
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Load datasets
    train_data = torch.load(config["tokenized_data"]["train"])
    val_data = torch.load(config["tokenized_data"]["val"])
    train_dataset = TextDataset(train_data, config["context_length"])
    val_dataset = TextDataset(val_data, config["context_length"])

    # DataLoader with efficient settings
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], num_workers=4, pin_memory=True)

    # Train multiple models
    for model_size in config["model_sizes"]:
        model_name = model_size["name"]
        print(f"Training model: {model_name}")
        model = create_model(model_size)
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"])
        train_model(model, train_loader, val_loader, optimizer, config["epochs"], model_name)
