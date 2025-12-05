"""
Visualization utilities for results and metrics.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_confusion_matrix(cm, class_names, save_path=None, figsize=(10, 8)):
    """
    Plot confusion matrix heatmap.

    Args:
        cm (np.array): Confusion matrix
        class_names (list): List of class names
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss',
                       'train_acc', 'val_acc' lists
        save_path (str, optional): Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_per_class_accuracy(accuracies, class_names, save_path=None, figsize=(12, 6)):
    """
    Plot per-class accuracy bar chart.

    Args:
        accuracies (list or np.array): Accuracy for each class
        class_names (list): List of class names
        save_path (str, optional): Path to save the figure
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    bars = plt.bar(class_names, accuracies)

    # Color bars based on accuracy
    for i, bar in enumerate(bars):
        if accuracies[i] >= 0.9:
            bar.set_color('green')
        elif accuracies[i] >= 0.7:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
