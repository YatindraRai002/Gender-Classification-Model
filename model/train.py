

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm

from utils import set_seed, get_device, compute_class_weights, ensure_output_dir
from data_loader import load_datasets, create_dataloaders, print_dataset_info
from model import get_model, freeze_base, unfreeze_top_layers, unfreeze_all, print_model_summary




DEFAULT_CONFIG = {
    "seed": 42,
    "image_size": 224,
    "batch_size": 64,
    "num_workers": 4,
    "lr_phase1": 1e-3,        # Feature extraction LR
    "lr_phase2": 1e-4,        # Fine-tuning LR
    "epochs_phase1": 10,
    "epochs_phase2": 10,        # Reduced from 20 to 10 as per user request (Total 20)
    "patience": 5,            # Early stopping patience
    "min_delta": 0.001,       # Minimum improvement for early stopping
    "use_amp": True,          # Mixed precision training
    "output_dir": "outputs",
}




class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for accuracy/f1.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n[EARLY STOP] No improvement for {self.patience} epochs.")
                return True

        return False



def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """
    Train model for one epoch.

    Returns:
        Tuple of (average_loss, accuracy, f1_score).
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  Training", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (outputs.detach() > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int))

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average="binary")

    return epoch_loss, epoch_acc, epoch_f1


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, criterion, device):
    """
    Evaluate model on validation/test set.

    Returns:
        Tuple of (average_loss, accuracy, f1_score).
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Validating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).long().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy().astype(int))

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average="binary")

    return epoch_loss, epoch_acc, epoch_f1



def train_model(
    model_name: str,
    config: dict = None,
    data_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "Dataset"),
):
    """
    Complete two-phase training pipeline for a single model.

    Phase 1: Feature extraction (frozen base, high LR)
    Phase 2: Fine-tuning (unfrozen top layers, low LR)

    Args:
        model_name: One of 'resnet50', 'efficientnet_b0', 'mobilenetv2'.
        config: Training configuration dict.
        data_dir: Path to dataset root.

    Returns:
        Dictionary with training results and best checkpoint path.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    output_dir = ensure_output_dir(config["output_dir"])
    set_seed(config["seed"])
    device = get_device()

    # ---- Load Data ----
    print("\n[INFO] Loading datasets...")
    train_ds, val_ds, test_ds = load_datasets(data_dir, config["image_size"])
    print_dataset_info(train_ds, val_ds, test_ds)

    train_loader, val_loader, _ = create_dataloaders(
        train_ds, val_ds, test_ds,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    # ---- Compute Class Weights ----
    class_weights = compute_class_weights(train_ds)
    # For BCELoss: weight each sample by its class weight
    # ImageFolder assigns alphabetically: Female=0, Male=1
    pos_weight = class_weights[1] / class_weights[0]  # Weight for positive class
    print(f"[INFO] pos_weight for BCEWithLogitsLoss alternative: {pos_weight:.4f}")

    # ---- Create Model ----
    print(f"\n[INFO] Creating model: {model_name}")
    model = get_model(model_name, pretrained=True)
    model = model.to(device)

    # ---- Training History ----
    history = []
    best_val_f1 = 0.0
    best_checkpoint_path = os.path.abspath(os.path.join(output_dir, f"best_{model_name}.pth"))
    live_log_path = os.path.abspath(os.path.join(output_dir, "training_log.csv"))
    start_time = time.time()

    # ---- Mixed Precision Scaler ----
    scaler = GradScaler("cuda") if config["use_amp"] and device.type == "cuda" else None

    # ---- Resume Logic ----
    if os.path.exists(best_checkpoint_path):
        print(f"[INFO] Found existing checkpoint: {best_checkpoint_path}")
        try:
            # Use weights_only=False to allow loading the full checkpoint dictionary
            checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_f1 = checkpoint.get("val_f1", 0.0)
            print(f"[SUCCESS] Loaded checkpoint (val_f1={best_val_f1:.4f})")
            
            if os.path.exists(live_log_path):
                history = pd.read_csv(live_log_path).to_dict("records")
                print(f"[INFO] Restored history: {len(history)} epochs")
                p1 = sum(1 for h in history if str(h.get("phase")) == "phase1")
                p2 = sum(1 for h in history if str(h.get("phase")) == "phase2")
                print(f"[DEBUG] Phase count - P1: {p1}, P2: {p2}")
        except Exception as e:
            print(f"[ERROR] Could not load checkpoint: {repr(e)}. Starting fresh.")
    else:
        print(f"[INFO] No existing checkpoint found. Starting fresh.")

    # ============================================
    # PHASE 1: FEATURE EXTRACTION (FROZEN BASE)
    # ============================================
    print(f"\n{'='*60}")
    print(f"PHASE 1: FEATURE EXTRACTION ({config['epochs_phase1']} epochs)")
    print(f"{'='*60}")

    freeze_base(model)
    print_model_summary(model, f"{model_name} (Phase 1)")

    # Loss with class weights
    criterion = nn.BCELoss(
        weight=None  # We use sample-level weighting below
    )

    # Custom weighted loss
    sample_weights = class_weights.to(device)

    class WeightedBCELoss(nn.Module):
        def __init__(self, class_weights):
            super().__init__()
            self.class_weights = class_weights

        def forward(self, pred, target):
            # Weight each sample by its class weight
            weights = self.class_weights[0] * (1 - target) + self.class_weights[1] * target
            bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
            return (bce * weights).mean()

    criterion = WeightedBCELoss(class_weights.to(device))

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_phase1"],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    early_stop = EarlyStopping(patience=config["patience"], mode="max")

    current_phase1_epochs = sum(1 for h in history if h.get("phase") == "phase1")
    
    if current_phase1_epochs >= config["epochs_phase1"]:
        print("\n[INFO] Phase 1 already completed. Skipping to Phase 2.")
    else:
        for epoch in range(current_phase1_epochs + 1, config["epochs_phase1"] + 1):
            print(f"\nEpoch {epoch}/{config['epochs_phase1']}  (LR: {optimizer.param_groups[0]['lr']:.6f})")

            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

            scheduler.step(val_f1)

            print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}")
            print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

            history.append({
                "epoch": epoch,
                "phase": "phase1",
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
                "train_f1": train_f1, "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            })

            # Flush live CSV after each epoch for monitoring
            pd.DataFrame(history).to_csv(live_log_path, index=False)

            # Checkpoint best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "config": config,
                }, best_checkpoint_path)
                print(f"  💾 Checkpoint saved (val_f1={val_f1:.4f})")

            if early_stop(val_f1):
                break

    # ============================================
    # PHASE 2: FINE-TUNING (UNFROZEN TOP LAYERS)
    # ============================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: FINE-TUNING ({config['epochs_phase2']} epochs)")
    print(f"{'='*60}")

    unfreeze_top_layers(model, n_layers=4)
    print_model_summary(model, f"{model_name} (Phase 2)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr_phase2"],
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    early_stop = EarlyStopping(patience=config["patience"], mode="max")

    current_phase2_epochs = sum(1 for h in history if h.get("phase") == "phase2")
    
    if current_phase2_epochs >= config["epochs_phase2"]:
        print("\n[INFO] Phase 2 already completed. Training is finished.")
    else:
        epoch_offset = current_phase1_epochs
        for epoch in range(current_phase2_epochs + 1, config["epochs_phase2"] + 1):
            global_epoch = epoch_offset + epoch
            print(f"\nEpoch {epoch}/{config['epochs_phase2']} (Global: {global_epoch})  "
                  f"(LR: {optimizer.param_groups[0]['lr']:.6f})")

            train_loss, train_acc, train_f1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)

            scheduler.step(val_f1)

            print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  F1: {train_f1:.4f}")
            print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")

            history.append({
                "epoch": global_epoch,
                "phase": "phase2",
                "train_loss": train_loss, "val_loss": val_loss,
                "train_acc": train_acc, "val_acc": val_acc,
                "train_f1": train_f1, "val_f1": val_f1,
                "lr": optimizer.param_groups[0]["lr"],
            })

            # Flush live CSV after each epoch for monitoring
            pd.DataFrame(history).to_csv(live_log_path, index=False)

            # Checkpoint best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "epoch": global_epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "config": config,
                }, best_checkpoint_path)
                print(f"  💾 Checkpoint saved (val_f1={val_f1:.4f})")

            if early_stop(val_f1):
                break

    total_time = time.time() - start_time

    # ---- Save History ----
    history_df = pd.DataFrame(history)
    history_path = os.path.join(output_dir, f"history_{model_name}.csv")
    history_df.to_csv(history_path, index=False)
    print(f"\n[INFO] Training history saved: {history_path}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE: {model_name}")
    print(f"{'='*60}")
    print(f"  Best validation F1:    {best_val_f1:.4f}")
    print(f"  Total training time:   {total_time / 60:.1f} minutes")
    print(f"  Best checkpoint:       {best_checkpoint_path}")
    print(f"  History CSV:           {history_path}")

    # Get model size
    model_size_mb = os.path.getsize(best_checkpoint_path) / 1e6

    return {
        "model_name": model_name,
        "best_val_f1": best_val_f1,
        "training_time_min": total_time / 60,
        "model_size_mb": model_size_mb,
        "checkpoint_path": best_checkpoint_path,
        "history_path": history_path,
    }




def parse_args():
    parser = argparse.ArgumentParser(description="Train gender classification models")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["resnet50", "efficientnet_b0", "mobilenetv2", "all"],
                        help="Model to train (default: efficientnet_b0)")
    parser.add_argument("--data-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Dataset"),
                        help="Path to dataset root")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64)")
    parser.add_argument("--epochs-phase1", type=int, default=10,
                        help="Epochs for phase 1 (default: 10)")
    parser.add_argument("--epochs-phase2", type=int, default=10,
                        help="Epochs for phase 2 (default: 20)")
    parser.add_argument("--lr1", type=float, default=1e-3,
                        help="Learning rate for phase 1")
    parser.add_argument("--lr2", type=float, default=1e-4,
                        help="Learning rate for phase 2")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update({
        "batch_size": args.batch_size,
        "epochs_phase1": args.epochs_phase1,
        "epochs_phase2": args.epochs_phase2,
        "lr_phase1": args.lr1,
        "lr_phase2": args.lr2,
        "patience": args.patience,
        "use_amp": not args.no_amp,
        "num_workers": args.num_workers,
        "output_dir": args.output_dir,
    })

    models_to_train = (
        ["resnet50", "efficientnet_b0", "mobilenetv2"]
        if args.model == "all"
        else [args.model]
    )

    all_results = []
    for model_name in models_to_train:
        print(f"\n{'#'*60}")
        print(f"# TRAINING: {model_name.upper()}")
        print(f"{'#'*60}")
        result = train_model(model_name, config, args.data_dir)
        all_results.append(result)

    # Print comparison table
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'Val F1':>10} {'Time (min)':>12} {'Size (MB)':>12}")
        print("-" * 70)
        for r in sorted(all_results, key=lambda x: x["best_val_f1"], reverse=True):
            print(f"{r['model_name']:<20} {r['best_val_f1']:>10.4f} "
                  f"{r['training_time_min']:>12.1f} {r['model_size_mb']:>12.1f}")
        print(f"{'='*70}")
        best = max(all_results, key=lambda x: x["best_val_f1"])
        print(f"\n🏆 BEST MODEL: {best['model_name']} (F1={best['best_val_f1']:.4f})")


if __name__ == "__main__":
    main()
