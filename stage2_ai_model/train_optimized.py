"""Optimized training script for high-performance multi-core/GPU training."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import math
import time
import os
import warnings

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


def train_epoch_optimized(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    """Optimized training epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in tqdm(dataloader, desc="Training"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if use_amp and scaler is not None:
            with autocast():
                logits = model(x)
                
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    continue
                
                loss = criterion(logits, y)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue
            
            loss = criterion(logits, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    if total == 0:
        return float('inf'), 0.0
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float('inf')
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate_optimized(model, dataloader, criterion, device, use_amp=False):
    """Optimized validation with mixed precision."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    valid_batches = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Validating"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if use_amp:
                with autocast():
                    logits = model(x)
            else:
                logits = model(x)
            
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                continue
            
            loss = criterion(logits, y)
            
            if torch.isnan(loss) or torch.isinf(loss):
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


def train_optimized(
    data_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    validation_split: float = 0.2,
    checkpoint_dir: str = "stage2_ai_model/checkpoints",
    config: dict = None,
    num_workers: int = None,
    pin_memory: bool = True,
    use_amp: bool = True,
    use_compile: bool = False,
    multi_gpu: bool = True,
    prefetch_factor: int = 2,
):
    """Optimized training function with multi-core/GPU support."""
    if config is None:
        config = load_config()
    
    # Auto-detect optimal number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, 8)  # Cap at 8 to avoid overhead
    
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
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Adjust batch size for multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and multi_gpu:
        batch_size = batch_size * torch.cuda.device_count()
        print(f"Multi-GPU detected ({torch.cuda.device_count()} GPUs), adjusted batch size to {batch_size}")
    
    # Create optimized data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config).to(device)
    
    # Multi-GPU support
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and multi_gpu:
        if torch.cuda.device_count() > 1:
            print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
            model = DataParallel(model)
        device = torch.device("cuda:0")
    
    # Compile model for PyTorch 2.0+ (optional, can speed up training)
    # Note: torch.compile is most beneficial on GPU. On CPU/Apple Silicon, it causes warnings
    # and doesn't provide significant speedup, so we disable it by default on CPU
    if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='reduce-overhead')
    elif use_compile and device.type == 'cpu':
        print("torch.compile disabled on CPU (causes warnings and minimal benefit on Apple Silicon)")
        use_compile = False
        # Suppress the cudagraph warnings that would appear
        warnings.filterwarnings('ignore', message='.*cudagraph.*', category=UserWarning)
    
    # Mixed precision training
    scaler = None
    if use_amp and device.type == 'cuda':
        print("Using mixed precision training (AMP)")
        scaler = GradScaler()
    elif use_amp and device.type == 'cpu':
        print("AMP not available on CPU, disabling")
        use_amp = False
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
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
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Workers: {num_workers}")
    print(f"  Mixed precision: {use_amp}")
    print(f"  Model compiled: {use_compile}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"\nStarting training...\n")
    
    # Suppress torch.compile warnings on CPU
    if device.type == 'cpu' and use_compile:
        import warnings
        warnings.filterwarnings('ignore', message='.*cudagraph.*')
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch_optimized(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        
        # Validate
        val_loss, val_acc = validate_optimized(
            model, val_loader, criterion, device, use_amp
        )
        
        # Check for invalid losses
        if np.isnan(train_loss) or np.isinf(train_loss):
            print(f"Warning: Invalid train loss: {train_loss}, skipping epoch")
            continue
        if np.isnan(val_loss) or np.isinf(val_loss):
            print(f"Warning: Invalid val loss: {val_loss}, skipping epoch")
            continue
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Handle DataParallel model saving
            model_to_save = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
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
    
    total_time = time.time() - start_time
    
    # Save final model
    model_to_save = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'history': history,
    }, f"{checkpoint_dir}/final_model.pth")
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    # Final summary
    if math.isinf(best_val_loss) or math.isnan(best_val_loss):
        print(f"\nTraining complete! Best validation loss: {best_val_loss}")
        print("Warning: Model may not have converged properly.")
    else:
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average time per epoch: {total_time/epochs:.2f} seconds")
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
    plt.savefig(f"{output_dir}/training_history.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Optimized training for gravitational wave detection model")
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
        help="Batch size (will be multiplied by number of GPUs)",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of data loading workers (default: auto-detect)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile optimization",
    )
    parser.add_argument(
        "--no-multi-gpu",
        action="store_true",
        help="Disable multi-GPU training",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable pinned memory",
    )
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Override config with command line args
    if args.epochs:
        config.setdefault("model", {}).setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("model", {}).setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("model", {}).setdefault("training", {})["learning_rate"] = args.lr
    
    train_optimized(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        config=config,
        num_workers=args.workers,
        pin_memory=not args.no_pin_memory,
        use_amp=not args.no_amp,
        use_compile=not args.no_compile and hasattr(torch, 'compile'),
        multi_gpu=not args.no_multi_gpu,
    )


if __name__ == "__main__":
    main()
