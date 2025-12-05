"""
Evaluation metrics and utilities.
"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def calculate_accuracy(outputs, labels):
    """
    Calculate accuracy from model outputs and labels.

    Args:
        outputs (torch.Tensor): Model outputs (logits)
        labels (torch.Tensor): Ground truth labels

    Returns:
        float: Accuracy between 0 and 1
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total


def calculate_metrics(y_true, y_pred, average='macro'):
    """
    Calculate precision, recall, and F1-score.

    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels
        average (str): Averaging method ('macro', 'micro', 'weighted')

    Returns:
        dict: Dictionary containing metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    acc = accuracy_score(y_true, y_pred)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def get_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix.

    Args:
        y_true (list or np.array): True labels
        y_pred (list or np.array): Predicted labels

    Returns:
        np.array: Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)
