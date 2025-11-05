"""
Example training script.

Replace this with your actual training implementation.

Usage:
    python scripts/training/train_example.py --epochs 50 --batch-size 32
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.example_model import ExampleCNN
from src.data.dataset import ExampleDataset, get_transforms
from src.utils.metrics import calculate_accuracy


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item()
        running_acc += calculate_accuracy(outputs, labels)

        pbar.set_postfix({'loss': loss.item(), 'acc': calculate_accuracy(outputs, labels)})

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += calculate_accuracy(outputs, labels)

    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    return avg_loss, avg_acc


def main(args):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_dataset = ExampleDataset(
        args.data_path, split='train', transform=get_transforms('train')
    )
    val_dataset = ExampleDataset(
        args.data_path, split='val', transform=get_transforms('val')
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Model
    model = ExampleCNN(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train example model")
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='models/example_cnn',
                       help='Output directory for models')

    args = parser.parse_args()
    main(args)
