"""
Evaluation metrics and utilities for defect detection
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class EarlyStopping:
    """Early stopping utility for training"""

    def __init__(
        self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = False
    ):
        """
        Args:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in monitored metric to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, val_loss: float, model: Optional[torch.nn.Module] = None) -> bool:
        """
        Check if training should stop

        Args:
            val_loss: Current validation loss
            model: Model to save weights from (if restore_best_weights=True)

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights and model is not None:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None and model is not None:
                model.load_state_dict(self.best_weights)
            return True

        return False

    def reset(self):
        """Reset early stopping state"""
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1-score

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method ('micro', 'macro', 'weighted')

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

    return precision, recall, f1


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute comprehensive multiclass classification metrics

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for AUC computation)
        class_names: List of class names

    Returns:
        Dictionary containing all metrics
    """
    if class_names is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
        class_names = [f"Class_{label}" for label in labels]
    else:
        labels = list(range(len(class_names)))

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_micro = precision_score(
        y_true, y_pred, labels=labels, average="micro", zero_division=0
    )
    precision_weighted = precision_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall_micro = recall_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    recall_weighted = recall_score(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )

    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    # AUC scores (if probabilities provided)
    auc_scores = {}
    if y_prob is not None:
        try:
            # One-vs-rest AUC for multiclass
            for label, class_name in zip(labels, class_names):
                if len(np.unique(y_true == label)) > 1:
                    auc_scores[f"auc_{class_name}"] = roc_auc_score(
                        y_true == label, y_prob[:, label]
                    )
        except Exception as e:
            print(f"Warning: Could not compute AUC scores: {e}")

    metrics = {
        "accuracy": float(accuracy),
        "precision": {
            "macro": float(precision_macro),
            "micro": float(precision_micro),
            "weighted": float(precision_weighted),
        },
        "recall": {
            "macro": float(recall_macro),
            "micro": float(recall_micro),
            "weighted": float(recall_weighted),
        },
        "f1": {"macro": float(f1_macro), "micro": float(f1_micro), "weighted": float(f1_weighted)},
        "confusion_matrix": cm.tolist(),
        "per_class_report": report,
        "class_names": class_names,
    }

    if auc_scores:
        metrics["auc_scores"] = auc_scores

    return metrics


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training history curves"""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    ax2.set_title("Training and Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # F1 Score
    if "val_f1" in history:
        ax3.plot(epochs, history["val_f1"], "g-", label="Val F1")
        ax3.set_title("Validation F1 Score")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1 Score")
        ax3.legend()
        ax3.grid(True)

    # Learning Rate
    if "lr" in history:
        ax4.plot(epochs, history["lr"], "purple", label="Learning Rate")
        ax4.set_title("Learning Rate Schedule")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_yscale("log")
        ax4.legend()
        ax4.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], normalize: bool = False, save_path: Optional[str] = None
):
    """Plot confusion matrix"""
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count" if not normalize else "Proportion"},
    )

    plt.title(title, fontsize=16)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: Optional[str] = None
):
    """Plot precision-recall curves for each class"""
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true == i, y_prob[:, i])
        plt.plot(
            recall, precision, label=f"{class_names[i]} (AUC={np.trapz(precision, recall):.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str], save_path: Optional[str] = None
):
    """Plot ROC curves for each class"""
    n_classes = len(class_names)

    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
