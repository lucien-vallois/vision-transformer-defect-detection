"""
Analysis utilities for defect detection model evaluation

Provides functions for error analysis, ROC/PR curves, and detailed performance metrics.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
)
import torch

warnings.filterwarnings("ignore")


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for all classes (one-vs-rest).

    Args:
        y_true: True labels (shape: [n_samples])
        y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
        class_names: List of class names
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    n_classes = len(class_names)

    # Convert to one-hot encoding for multi-class ROC
    y_true_onehot = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC = {roc_auc:.3f})")

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for all classes.

    Args:
        y_true: True labels (shape: [n_samples])
        y_pred_proba: Predicted probabilities (shape: [n_samples, n_classes])
        class_names: List of class names
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    n_classes = len(class_names)

    # Convert to one-hot encoding
    y_true_onehot = np.eye(n_classes)[y_true]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot PR curve for each class
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        avg_precision = average_precision_score(y_true_onehot[:, i], y_pred_proba[:, i])

        ax.plot(recall, precision, lw=2, label=f"{class_name} (AP = {avg_precision:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def plot_confusion_matrix_enhanced(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot enhanced confusion matrix with annotations and statistics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        ax=ax,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def analyze_errors(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray, class_names: List[str]
) -> Dict:
    """
    Analyze prediction errors and return detailed statistics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names

    Returns:
        Dictionary containing error analysis
    """
    errors = y_true != y_pred
    error_indices = np.where(errors)[0]

    # Error statistics
    error_rate = errors.sum() / len(y_true)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Per-class error analysis
    class_errors = {}
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        class_pred = y_pred[class_mask]
        class_errors[class_name] = {
            "total": class_mask.sum(),
            "correct": (class_pred == i).sum(),
            "errors": (class_pred != i).sum(),
            "error_rate": (class_pred != i).sum() / class_mask.sum() if class_mask.sum() > 0 else 0,
            "most_confused_with": {},
        }

        # Find most common misclassifications
        if class_mask.sum() > 0:
            incorrect_mask = class_pred != i
            if incorrect_mask.sum() > 0:
                incorrect_preds = class_pred[incorrect_mask]
                unique, counts = np.unique(incorrect_preds, return_counts=True)
                for pred_class, count in zip(unique, counts):
                    if pred_class != i:
                        class_errors[class_name]["most_confused_with"][
                            class_names[pred_class]
                        ] = int(count)

    # Confidence analysis for errors
    error_confidences = (
        y_pred_proba[error_indices].max(axis=1) if len(error_indices) > 0 else np.array([])
    )
    correct_confidences = y_pred_proba[~errors].max(axis=1) if (~errors).sum() > 0 else np.array([])

    return {
        "error_rate": float(error_rate),
        "total_errors": int(errors.sum()),
        "total_samples": int(len(y_true)),
        "confusion_matrix": cm.tolist(),
        "class_errors": class_errors,
        "error_confidence_mean": float(error_confidences.mean())
        if len(error_confidences) > 0
        else 0.0,
        "error_confidence_std": float(error_confidences.std())
        if len(error_confidences) > 0
        else 0.0,
        "correct_confidence_mean": float(correct_confidences.mean())
        if len(correct_confidences) > 0
        else 0.0,
        "correct_confidence_std": float(correct_confidences.std())
        if len(correct_confidences) > 0
        else 0.0,
    }


def plot_error_analysis(
    error_analysis: Dict, class_names: List[str], save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comprehensive error analysis visualization.

    Args:
        error_analysis: Dictionary from analyze_errors function
        class_names: List of class names
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Error rate by class
    class_error_rates = [error_analysis["class_errors"][name]["error_rate"] for name in class_names]
    axes[0, 0].bar(class_names, class_error_rates, color="coral")
    axes[0, 0].set_ylabel("Error Rate", fontsize=11)
    axes[0, 0].set_title("Error Rate by Class", fontsize=12, fontweight="bold")
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 2. Confidence distribution for errors vs correct
    error_conf = error_analysis.get("error_confidence_mean", 0)
    correct_conf = error_analysis.get("correct_confidence_mean", 0)

    categories = ["Errors", "Correct"]
    confidences = [error_conf, correct_conf]
    colors = ["coral", "lightblue"]

    axes[0, 1].bar(categories, confidences, color=colors)
    axes[0, 1].set_ylabel("Mean Confidence", fontsize=11)
    axes[0, 1].set_title("Confidence: Errors vs Correct", fontsize=12, fontweight="bold")
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis="y", alpha=0.3)

    # 3. Most confused pairs
    confusion_pairs = []
    for true_class in class_names:
        confused_with = error_analysis["class_errors"][true_class].get("most_confused_with", {})
        for pred_class, count in confused_with.items():
            confusion_pairs.append((true_class, pred_class, count))

    if confusion_pairs:
        confusion_pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = confusion_pairs[:10]  # Top 10

        pairs_labels = [f"{true} → {pred}" for true, pred, _ in top_pairs]
        pairs_counts = [count for _, _, count in top_pairs]

        axes[1, 0].barh(pairs_labels, pairs_counts, color="steelblue")
        axes[1, 0].set_xlabel("Count", fontsize=11)
        axes[1, 0].set_title("Top Misclassification Pairs", fontsize=12, fontweight="bold")
        axes[1, 0].grid(axis="x", alpha=0.3)

    # 4. Summary statistics
    axes[1, 1].axis("off")
    summary_text = f"""
    Error Analysis Summary

    Total Samples: {error_analysis['total_samples']}
    Total Errors: {error_analysis['total_errors']}
    Overall Error Rate: {error_analysis['error_rate']:.2%}

    Confidence Statistics:
    - Errors: {error_analysis['error_confidence_mean']:.3f} ± {error_analysis['error_confidence_std']:.3f}
    - Correct: {error_analysis['correct_confidence_mean']:.3f} ± {error_analysis['correct_confidence_std']:.3f}
    """
    axes[1, 1].text(
        0.1,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def plot_confidence_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confidence distribution for correct and incorrect predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        save_path: Optional path to save the figure

    Returns:
        Matplotlib figure object
    """
    correct_mask = y_true == y_pred
    error_mask = ~correct_mask

    correct_conf = (
        y_pred_proba[correct_mask].max(axis=1) if correct_mask.sum() > 0 else np.array([])
    )
    error_conf = y_pred_proba[error_mask].max(axis=1) if error_mask.sum() > 0 else np.array([])

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(correct_conf) > 0:
        ax.hist(correct_conf, bins=30, alpha=0.7, label="Correct", color="green", density=True)
    if len(error_conf) > 0:
        ax.hist(error_conf, bins=30, alpha=0.7, label="Errors", color="red", density=True)

    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Errors", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return fig


def generate_comprehensive_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    class_names: List[str],
    output_dir: str,
) -> Dict:
    """
    Generate comprehensive evaluation report with all visualizations.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        class_names: List of class names
        output_dir: Directory to save all outputs

    Returns:
        Dictionary with paths to generated files and statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    roc_fig = plot_roc_curves(
        y_true, y_pred_proba, class_names, save_path=str(output_path / "roc_curves.png")
    )

    pr_fig = plot_pr_curves(
        y_true, y_pred_proba, class_names, save_path=str(output_path / "pr_curves.png")
    )

    cm_fig = plot_confusion_matrix_enhanced(
        y_true, y_pred, class_names, save_path=str(output_path / "confusion_matrix.png")
    )

    error_analysis = analyze_errors(y_true, y_pred, y_pred_proba, class_names)
    error_fig = plot_error_analysis(
        error_analysis, save_path=str(output_path / "error_analysis.png")
    )

    conf_fig = plot_confidence_distribution(
        y_true,
        y_pred,
        y_pred_proba,
        class_names,
        save_path=str(output_path / "confidence_distribution.png"),
    )

    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save report as JSON
    import json

    with open(output_path / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return {
        "output_dir": str(output_path),
        "files": {
            "roc_curves": str(output_path / "roc_curves.png"),
            "pr_curves": str(output_path / "pr_curves.png"),
            "confusion_matrix": str(output_path / "confusion_matrix.png"),
            "error_analysis": str(output_path / "error_analysis.png"),
            "confidence_distribution": str(output_path / "confidence_distribution.png"),
            "classification_report": str(output_path / "classification_report.json"),
        },
        "error_analysis": error_analysis,
        "classification_report": report,
    }
