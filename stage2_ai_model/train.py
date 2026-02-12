"""Training script for gravitational wave detection model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import math

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage2_ai_model.model import create_model
from utils.config_loader import load_config
from utils.data_utils import load_dataset, normalize_data


class GWDataset(Dataset):
    """Dataset for gravitational wave detection."""
    
    def __init__(self, data, labels):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add channel dimension: [seq_len] -> [1, seq_len]
        x = self.data[idx].unsqueeze(0)
        y = self.labels[idx]
        return x, y


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        
        # Check for invalid values
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("Warning: Invalid logits detected, skipping batch")
            continue
        
        loss = criterion(logits, y)
        
        # Check for invalid loss
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Invalid loss detected, skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    if len(dataloader) == 0:
        return float('inf'), 0.0
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            
            # Check for invalid values
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print("Warning: Invalid logits in validation, skipping batch")
                continue
            
            loss = criterion(logits, y)
            
            # Check for invalid loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: Invalid loss in validation, skipping batch")
                continue
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            valid_batches += 1
    
    if valid_batches == 0:
        return float('inf'), 0.0
    
    avg_loss = total_loss / valid_batches
    accuracy = 100 * correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def train(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    checkpoint_dir: str = "stage2_ai_model/checkpoints",
    config: dict = None,
):
    """Train the model."""
    if config is None:
        config = load_config()
    
    # Load data
    print("Loading dataset...")
    data, labels = load_dataset(data_path)
    print(f"Dataset shape: {data.shape}, Labels: {labels.shape}")
    
    # Create dataset
    dataset = GWDataset(data, labels)
    
    # Split into train and validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    
    # Create model
    print("Creating model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = create_model(config).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = config.get("model", {}).get("training", {}).get("early_stopping_patience", 10)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}\n")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Check for invalid losses
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"Warning: Invalid train loss: {train_loss}, skipping epoch")
            continue
        if np.isnan(val_loss) or np.isinf(val_loss):
            print(f"Warning: Invalid val loss: {val_loss}, skipping epoch")
            continue
        
        # Update learning rate (only if valid loss)
        if not (np.isnan(val_loss) or np.isinf(val_loss)):
            scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model (only if val_loss is valid)
        if not (np.isnan(val_loss) or np.isinf(val_loss)) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, f"{checkpoint_dir}/best_model.pth")
            print("✓ Saved best model")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, f"{checkpoint_dir}/final_model.pth")
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    # Format best validation loss
    if math.isinf(best_val_loss) or math.isnan(best_val_loss):
        print(f"\nTraining complete! Best validation loss: {best_val_loss}")
        print("Warning: Model may not have converged properly. Check training logs.")
    else:
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {checkpoint_dir}/")


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train gravitational wave detection model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset HDF5 file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="stage2_ai_model/checkpoints",
        help="Directory to save checkpoints",
    )
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Override config with command line args if provided
    if args.epochs:
        config.setdefault("model", {}).setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("model", {}).setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("model", {}).setdefault("training", {})["learning_rate"] = args.lr
    
    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
