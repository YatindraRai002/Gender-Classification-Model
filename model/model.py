"""
model.py — Transfer learning model architectures for gender classification.

Provides three pretrained architectures:
    1. ResNet50        — Deep residual network, strong baseline
    2. EfficientNetB0  — Efficient scaling, best accuracy/compute trade-off
    3. MobileNetV2     — Lightweight, fast inference, mobile-ready

Each model uses the same custom classifier head:
    AdaptiveAvgPool2d → Flatten → Linear → BatchNorm → ReLU → Dropout → Linear(1) → Sigmoid

Why Transfer Learning?
    - ImageNet pretrained models already understand edges, textures, facial features
    - Training from scratch on ~160K images would require much longer and risk overfitting
    - Transfer learning achieves higher accuracy with fewer epochs
    - The pretrained features generalize well to face-related tasks
"""

import torch
import torch.nn as nn
from torchvision import models


def _safe_load(model_fn, weights, **kwargs):
    """Try loading pretrained weights; fallback to random init on network error."""
    try:
        return model_fn(weights=weights, **kwargs)
    except Exception as e:
        print(f"[WARNING] Pretrained weight download failed: {e}")
        print("[WARNING] Training from scratch (random initialization)")
        return model_fn(weights=None, **kwargs)


# ============================================================
# CUSTOM CLASSIFIER HEAD
# ============================================================

class ClassifierHead(nn.Module):
    """
    Custom binary classification head.

    Architecture:
        GlobalAveragePooling → Linear(in, 512) → BatchNorm → ReLU → Dropout(0.3) → Linear(512, 1) → Sigmoid

    Design choices:
        - BatchNorm: Stabilizes training, allows higher learning rates
        - Dropout(0.3): Prevents overfitting on the dense layer
        - 512 hidden units: Good balance between capacity and overfitting risk
        - Sigmoid: Binary classification output (probability of Female)
    """

    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(x).squeeze(1)


# ============================================================
# MODEL BUILDERS
# ============================================================

def create_resnet50(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    ResNet50 with custom classifier head.

    - 25.6M parameters (base) + ~1.3M (head)
    - Strong baseline, well-understood architecture
    - 50 layers with skip connections prevent gradient vanishing
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    base = _safe_load(models.resnet50, weights)

    # Remove the original FC layer
    in_features = base.fc.in_features  # 2048
    base.fc = nn.Identity()

    # Get all layers except the last pooling + fc
    # ResNet50 structure: conv1 → bn1 → relu → maxpool → layer1-4 → avgpool → fc
    model = nn.Sequential()
    for name, module in base.named_children():
        if name not in ("avgpool", "fc"):
            model.add_module(name, module)

    # Add custom head
    model.add_module("classifier", ClassifierHead(in_features, dropout))

    return model


def create_efficientnet_b0(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    EfficientNetB0 with custom classifier head.

    - 5.3M parameters (base) + ~0.7M (head) = very efficient
    - Uses compound scaling (depth, width, resolution)
    - Best accuracy-to-compute ratio among the three models
    - Recommended for competitions where both speed and accuracy matter
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    base = _safe_load(models.efficientnet_b0, weights)
    in_features = base.classifier[1].in_features  # 1280

    # Remove original classifier
    base.classifier = nn.Identity()
    base.avgpool = nn.Identity()

    model = nn.Sequential()
    model.add_module("features", base.features)
    model.add_module("classifier", ClassifierHead(in_features, dropout))

    return model


def create_mobilenet_v2(pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    MobileNetV2 with custom classifier head.

    - 3.4M parameters (base) + ~0.7M (head) = smallest model
    - Uses depthwise separable convolutions for efficiency
    - Inverted residual blocks with linear bottlenecks
    - Ideal for real-time inference and mobile deployment
    """
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    base = _safe_load(models.mobilenet_v2, weights)
    in_features = base.classifier[1].in_features  # 1280

    # Remove original classifier
    base.classifier = nn.Identity()

    model = nn.Sequential()
    model.add_module("features", base.features)
    model.add_module("classifier", ClassifierHead(in_features, dropout))

    return model


# ============================================================
# MODEL FACTORY
# ============================================================

MODEL_REGISTRY = {
    "resnet50": create_resnet50,
    "efficientnet_b0": create_efficientnet_b0,
    "mobilenetv2": create_mobilenet_v2,
}


def get_model(name: str, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """
    Factory function to create a model by name.

    Args:
        name: One of 'resnet50', 'efficientnet_b0', 'mobilenetv2'.
        pretrained: Whether to load ImageNet weights.
        dropout: Dropout rate for classifier head.

    Returns:
        nn.Module with pretrained base + custom classifier.
    """
    name = name.lower().replace("-", "_")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")

    model = MODEL_REGISTRY[name](pretrained=pretrained, dropout=dropout)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[MODEL] {name}")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    return model


# ============================================================
# FREEZE / UNFREEZE UTILITIES
# ============================================================

def freeze_base(model: nn.Module) -> None:
    """
    Freeze all layers except the classifier head.

    Used in Phase 1 (Feature Extraction): only train the classifier
    while keeping the pretrained features fixed. This prevents
    catastrophic forgetting of learned ImageNet features.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Base frozen. Trainable parameters: {trainable:,}")


def unfreeze_top_layers(model: nn.Module, n_layers: int = 2) -> None:
    """
    Unfreeze the top N layers of the base model for fine-tuning.

    Used in Phase 2 (Fine-Tuning): gradually unfreeze top layers
    to adapt high-level features to face/gender-specific patterns
    while keeping low-level features (edges, textures) frozen.

    Args:
        model: The model to unfreeze.
        n_layers: Number of top-level modules to unfreeze.
    """
    # Get all named children of the model (excluding classifier)
    children = []
    for name, child in model.named_children():
        if "classifier" not in name:
            if hasattr(child, 'named_children'):
                sub_children = list(child.named_children())
                if sub_children:
                    children.extend([(f"{name}.{sn}", sc) for sn, sc in sub_children])
                else:
                    children.append((name, child))
            else:
                children.append((name, child))

    # Unfreeze the last n_layers
    layers_to_unfreeze = children[-n_layers:]
    for name, layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
        print(f"[INFO] Unfrozen: {name}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Trainable parameters after unfreezing: {trainable:,}")


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] All layers unfrozen. Trainable parameters: {trainable:,}")


# ============================================================
# MODEL INFO
# ============================================================

def print_model_summary(model: nn.Module, name: str = "Model"):
    """Print a summary of model architecture and parameter counts."""
    print(f"\n{'=' * 60}")
    print(f"MODEL SUMMARY: {name}")
    print(f"{'=' * 60}")

    total = 0
    trainable = 0
    for pname, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    frozen = total - trainable
    print(f"  Total parameters:     {total:>12,}")
    print(f"  Trainable parameters: {trainable:>12,}")
    print(f"  Frozen parameters:    {frozen:>12,}")
    print(f"  Model size (MB):      {total * 4 / 1e6:>12.1f}")
    print(f"{'=' * 60}")
