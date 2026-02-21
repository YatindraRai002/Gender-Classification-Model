"""
evaluate.py — Comprehensive evaluation and model comparison.

Computes:
    - Accuracy, Precision, Recall, F1-score, ROC-AUC
    - Confusion matrix
    - Classification report
    - ROC curve
    - Training/validation curves
    - Model comparison table

Usage:
    # Evaluate a single model
    python evaluate.py --model efficientnet_b0 --checkpoint outputs/best_efficientnet_b0.pth

    # Compare all trained models
    python evaluate.py --compare-all

    # Evaluate best model on TEST set (do this ONLY ONCE)
    python evaluate.py --model efficientnet_b0 --checkpoint outputs/best_efficientnet_b0.pth --test
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)
from tqdm import tqdm

from utils import set_seed, get_device, ensure_output_dir, plot_training_history
from data_loader import load_datasets, create_dataloaders
from model import get_model


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def get_predictions(model, loader, device):
    """
    Get all predictions and labels from a data loader.

    Returns:
        Tuple of (all_labels, all_probs, all_preds) as numpy arrays.
    """
    model.eval()
    all_labels = []
    all_probs = []

    for images, labels in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_probs.extend(outputs.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs > 0.5).astype(int)

    return all_labels, all_probs, all_preds


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(labels, probs, preds):
    """
    Compute comprehensive evaluation metrics.

    Returns dict with accuracy, precision, recall, f1, roc_auc.

    Metric Explanations:
        - Accuracy: Overall correct predictions / total
        - Precision: Of predicted Female, how many are actually Female?
          (Important to avoid misclassifying Male as Female)
        - Recall: Of actual Female, how many did we correctly identify?
          (Important to ensure we don't miss Female)
        - F1-Score: Harmonic mean of precision & recall — balances both
        - ROC-AUC: Area under ROC curve — measures discriminative ability
          across all thresholds (1.0 = perfect, 0.5 = random)
    """
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
        "roc_auc": roc_auc_score(labels, probs),
    }
    return metrics


def print_metrics(metrics: dict, title: str = "Evaluation Metrics"):
    """Print formatted metrics table."""
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    for name, value in metrics.items():
        print(f"  {name.upper():<15} {value:.4f}")
    print(f"{'='*50}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_confusion_matrix(labels, preds, class_names, save_path=None):
    """
    Plot confusion matrix heatmap.

    The confusion matrix shows:
        - True Positives (correctly identified Female)
        - True Negatives (correctly identified Male)
        - False Positives (Male misclassified as Female)
        - False Negatives (Female misclassified as Male)
    """
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 16}, ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title("Confusion Matrix", fontsize=15, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Confusion matrix saved: {save_path}")
    plt.close()


def plot_roc_curve(labels, probs, save_path=None):
    """
    Plot ROC curve.

    The ROC curve plots True Positive Rate vs False Positive Rate
    at various classification thresholds. A perfect classifier
    hugs the top-left corner (AUC = 1.0).
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc_score = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#3498db", linewidth=2,
            label=f"ROC Curve (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
            label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#3498db")

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] ROC curve saved: {save_path}")
    plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(output_dir: str = "outputs"):
    """
    Compare all trained models using their history CSVs and checkpoints.

    Displays a comparison table and plots comparing all models.
    """
    import glob
    checkpoints = glob.glob(os.path.join(output_dir, "best_*.pth"))

    if not checkpoints:
        print("[ERROR] No checkpoints found in outputs/")
        return

    print(f"\n{'='*80}")
    print("MODEL COMPARISON")
    print(f"{'='*80}")

    results = []
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_name = ckpt.get("model_name", os.path.basename(ckpt_path))
        val_f1 = ckpt.get("val_f1", 0.0)
        val_acc = ckpt.get("val_acc", 0.0)
        epoch = ckpt.get("epoch", 0)
        size_mb = os.path.getsize(ckpt_path) / 1e6

        results.append({
            "Model": model_name,
            "Val Accuracy": val_acc,
            "Val F1": val_f1,
            "Best Epoch": epoch,
            "Size (MB)": size_mb,
        })

    df = pd.DataFrame(results).sort_values("Val F1", ascending=False)
    print(df.to_string(index=False))

    # Plot training histories
    histories = glob.glob(os.path.join(output_dir, "history_*.csv"))
    if histories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#3498db", "#e74c3c", "#2ecc71"]

        for idx, hist_path in enumerate(histories):
            hist = pd.read_csv(hist_path)
            model_name = os.path.basename(hist_path).replace("history_", "").replace(".csv", "")
            c = colors[idx % len(colors)]

            axes[0].plot(hist["epoch"], hist["val_loss"], f"-o", markersize=3,
                         color=c, label=model_name)
            axes[1].plot(hist["epoch"], hist["val_f1"], f"-o", markersize=3,
                         color=c, label=model_name)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Validation Loss")
        axes[0].set_title("Validation Loss Comparison")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation F1")
        axes[1].set_title("Validation F1 Comparison")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[INFO] Comparison plot saved: {save_path}")
        plt.close()

    best = df.iloc[0]
    print(f"\n[BEST] RECOMMENDED MODEL: {best['Model']} (F1={best['Val F1']:.4f})")

    return df


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    use_test: bool = False,
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "Dataset"),
    output_dir: str = "outputs",
):
    """
    Full evaluation of a single model.

    Args:
        model_name: Model architecture name.
        checkpoint_path: Path to saved checkpoint.
        use_test: If True, evaluate on test set (do this ONLY ONCE).
        data_dir: Dataset root.
        output_dir: Where to save plots.
    """
    ensure_output_dir(output_dir)
    set_seed(42)
    device = get_device()

    # Load data
    train_ds, val_ds, test_ds = load_datasets(data_dir)
    _, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=64,
        num_workers=4,
    )

    # Load model
    model = get_model(model_name, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"\n[SUCCESS] Loaded checkpoint: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', '?')}, Val F1: {checkpoint.get('val_f1', '?')}")

    # Choose loader
    split_name = "TEST" if use_test else "VALIDATION"
    loader = test_loader if use_test else val_loader

    if use_test:
        print("\n[WARNING] EVALUATING ON TEST SET - this should be done ONLY ONCE!")

    # Get predictions
    # ImageFolder: Female=0, Male=1 (alphabetical)
    class_names = ["Female", "Male"]  # Maps to indices 0, 1
    labels, probs, preds = get_predictions(model, loader, device)

    # Compute metrics
    metrics = compute_metrics(labels, probs, preds)
    print_metrics(metrics, f"{model_name.upper()} - {split_name} Set Metrics")

    # Classification report
    print(f"\n[INFO] CLASSIFICATION REPORT ({split_name}):")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # Confusion matrix
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}_{split_name.lower()}.png")
    plot_confusion_matrix(labels, preds, class_names, save_path=cm_path)

    # ROC curve
    roc_path = os.path.join(output_dir, f"roc_curve_{model_name}_{split_name.lower()}.png")
    plot_roc_curve(labels, probs, save_path=roc_path)

    # Training curves
    history_path = os.path.join(output_dir, f"history_{model_name}.csv")
    if os.path.exists(history_path):
        plot_training_history(history_path, output_dir)

    return metrics


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate gender classification models")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["resnet50", "efficientnet_b0", "mobilenetv2"],
                        help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--test", action="store_true",
                        help="Evaluate on test set (use ONLY ONCE)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all trained models")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset"))
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare_all:
        compare_models(args.output_dir)
        return

    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, f"best_{args.model}.pth")
        if not os.path.exists(args.checkpoint):
            print(f"[ERROR] No checkpoint found at {args.checkpoint}")
            print("  Train a model first: python train.py --model {args.model}")
            return

    evaluate_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        use_test=args.test,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
