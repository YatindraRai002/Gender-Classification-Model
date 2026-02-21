"""
utils.py — Shared utilities for the gender classification pipeline.

Includes:
    - Reproducibility (seed setting)
    - Device detection
    - Class weight computation
    - Grad-CAM visualization
    - Test Time Augmentation (TTA)
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """Set random seed for full reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to {seed}")


# ============================================================
# 2. DEVICE DETECTION
# ============================================================

def get_device() -> torch.device:
    """Detect the best available device (CUDA > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
        gpu_props = torch.cuda.get_device_properties(0)
        gpu_mem = getattr(gpu_props, "total_memory", getattr(gpu_props, "total_mem", 0))
        print(f"[INFO] GPU Memory: {gpu_mem / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU")
    return device


# ============================================================
# 3. CLASS WEIGHT COMPUTATION
# ============================================================

def compute_class_weights(dataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Args:
        dataset: A torchvision ImageFolder dataset.

    Returns:
        Tensor of shape (num_classes,) with weights.
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    total = len(targets)

    # Inverse frequency weighting: w_c = N / (num_classes * n_c)
    weights = total / (len(class_counts) * class_counts)
    weights_tensor = torch.FloatTensor(weights)

    print(f"[INFO] Class counts: {dict(enumerate(class_counts))}")
    print(f"[INFO] Class weights: {dict(enumerate(weights.round(4)))}")
    return weights_tensor


# ============================================================
# 4. GRAD-CAM VISUALIZATION
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Highlights which image regions the model focuses on for its prediction.
    Essential for understanding model decisions and detecting bias.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: Trained PyTorch model.
            target_layer: The convolutional layer to visualize (e.g., model.base[-1]).
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Preprocessed image tensor (1, C, H, W).
            target_class: Class index (0 or 1). If None, uses predicted class.

        Returns:
            Numpy heatmap of shape (H, W), values in [0, 1].
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = (output > 0.5).long().item()

        self.model.zero_grad()
        # For binary classification with sigmoid, the output is a single scalar
        target = output if target_class == 1 else (1 - output)
        target.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive contributions
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def visualize(self, image_path, input_tensor, save_path=None):
        """
        Overlay Grad-CAM heatmap on the original image.

        Args:
            image_path: Path to the original image.
            input_tensor: Preprocessed tensor.
            save_path: Optional path to save the visualization.
        """
        heatmap = self.generate(input_tensor)

        # Load original image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img) / 255.0

        # Resize heatmap to image size
        import cv2
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
        ) / 255.0

        # Create overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")

        overlay = img_np * 0.6 + plt.cm.jet(heatmap_resized)[:, :, :3] * 0.4
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Grad-CAM saved to {save_path}")
        plt.show()
        plt.close()


# ============================================================
# 5. TEST TIME AUGMENTATION (TTA)
# ============================================================

def predict_with_tta(model, image_path, device, n_augmentations=5):
    """
    Test Time Augmentation — make multiple augmented predictions and average.

    TTA improves robustness by averaging predictions across different
    augmented versions of the same input image.

    Args:
        model: Trained model.
        image_path: Path to the image.
        device: torch device.
        n_augmentations: Number of augmented versions (default: 5).

    Returns:
        dict with 'class', 'label', 'probability', and 'all_probs'.
    """
    # Base transform (no augmentation)
    base_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # TTA transforms (light augmentations)
    tta_transforms = [
        base_transform,  # Original
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),  # Always flip
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),  # Will need special handling
            transforms.Lambda(lambda crops: crops[0]),  # top-left
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: crops[1]),  # top-right
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        transforms.Compose([
            transforms.Resize(256),
            transforms.FiveCrop(224),
            transforms.Lambda(lambda crops: crops[4]),  # center
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    ]

    model.eval()
    img = Image.open(image_path).convert("RGB")
    probs = []

    with torch.no_grad():
        for t in tta_transforms[:n_augmentations]:
            tensor = t(img).unsqueeze(0).to(device)
            output = model(tensor)
            prob = output.item()
            probs.append(prob)

    avg_prob = np.mean(probs)
    predicted_class = 1 if avg_prob > 0.5 else 0
    label = "Female" if predicted_class == 1 else "Male"

    return {
        "class": predicted_class,
        "label": label,
        "probability": round(float(avg_prob), 4),
        "all_probs": [round(p, 4) for p in probs],
    }


# ============================================================
# 6. PLOTTING HELPERS
# ============================================================

def plot_training_history(history_csv_path, save_dir="outputs"):
    """
    Plot training vs validation loss and accuracy curves.

    Args:
        history_csv_path: Path to CSV with columns [epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1].
        save_dir: Directory to save plots.
    """
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(history_csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(df["epoch"], df["train_loss"], "b-o", label="Train Loss", markersize=3)
    axes[0].plot(df["epoch"], df["val_loss"], "r-o", label="Val Loss", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(df["epoch"], df["train_acc"], "b-o", label="Train Acc", markersize=3)
    axes[1].plot(df["epoch"], df["val_acc"], "r-o", label="Val Acc", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # F1
    axes[2].plot(df["epoch"], df["train_f1"], "b-o", label="Train F1", markersize=3)
    axes[2].plot(df["epoch"], df["val_f1"], "r-o", label="Val F1", markersize=3)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Training vs Validation F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Training curves saved to {save_path}")
    plt.close()


def ensure_output_dir(path: str = "outputs") -> str:
    """Create output directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path
