"""
Dataset preparation script.

Organize your raw data into train/val/test splits.

Usage:
    python scripts/preprocessing/prepare_dataset.py --input data/raw --output data/processed
"""

import argparse
import os
import shutil
from pathlib import Path


def prepare_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Organize dataset into train/val/test splits.

    Args:
        input_dir (str): Path to raw data
        output_dir (str): Path to processed data
        train_ratio (float): Fraction for training set
        val_ratio (float): Fraction for validation set
    """
    # TODO: Implement your dataset preparation logic
    print(f"Preparing dataset from {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {1-train_ratio-val_ratio}")

    # Example structure:
    # Create train/val/test directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

    print("Dataset preparation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with raw data')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for processed data')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')

    args = parser.parse_args()
    prepare_dataset(args.input, args.output, args.train_ratio, args.val_ratio)
