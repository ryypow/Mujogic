"""
Model evaluation script.

Usage:
    python scripts/evaluation/evaluate.py --model models/best_model.pth --data data/processed
"""

import argparse
import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.example_model import ExampleCNN
from src.data.dataset import ExampleDataset, get_transforms
from src.utils.metrics import calculate_metrics, get_confusion_matrix
from src.utils.visualization import plot_confusion_matrix, plot_per_class_accuracy


def evaluate(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def main(args):
    """Main evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = ExampleDataset(args.data_path, split=args.split, transform=get_transforms(args.split))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Load model
    model = ExampleCNN(num_classes=len(dataset.classes)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    # Evaluate
    y_true, y_pred = evaluate(model, dataloader, device)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")

    # Per-class accuracy
    cm = get_confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-class accuracy:")
    for i, (cls, acc) in enumerate(zip(dataset.classes, per_class_acc)):
        print(f"{cls}: {acc:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics JSON
    results = {
        'overall_metrics': metrics,
        'per_class_accuracy': {cls: float(acc) for cls, acc in zip(dataset.classes, per_class_acc)}
    }
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Plot confusion matrix
    plot_confusion_matrix(
        cm, dataset.classes,
        save_path=os.path.join(args.output_dir, 'confusion_matrix.png')
    )

    # Plot per-class accuracy
    plot_per_class_accuracy(
        per_class_acc, dataset.classes,
        save_path=os.path.join(args.output_dir, 'per_class_accuracy.png')
    )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-path', type=str, default='data/processed',
                       help='Path to dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Output directory for results')

    args = parser.parse_args()
    main(args)
