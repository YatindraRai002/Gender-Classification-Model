"""
inference.py — Production-ready inference pipeline.

Functions:
    - load_model(): Load a saved checkpoint
    - preprocess_image(): Apply validation transforms to a single image
    - predict(): Return class label + probability

Usage:
    # CLI single image
    python inference.py --image path/to/face.jpg --checkpoint outputs/best_efficientnet_b0.pth --model efficientnet_b0

    # Python API
    from inference import GenderClassifier
    clf = GenderClassifier("outputs/best_efficientnet_b0.pth", "efficientnet_b0")
    result = clf.predict("path/to/face.jpg")
    print(result)  # {"class": 0, "label": "Male", "probability": 0.1234}
"""

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import get_model
from utils import get_device, predict_with_tta


# ============================================================
# CONSTANTS
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ImageFolder assigns alphabetically: Female=0, Male=1
# Problem statement: Male=0, Female=1
# We remap at inference time:
IMAGEFOLDER_TO_LABEL = {0: "Female", 1: "Male"}
LABEL_TO_PROBLEM = {"Male": 0, "Female": 1}


# ============================================================
# PREPROCESSING
# ============================================================

def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """
    Get deterministic preprocessing transform for inference.

    Same as validation transform:
        Resize(256) → CenterCrop(224) → ToTensor → Normalize
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.

    Args:
        image_path: Path to the image file.
        image_size: Target size (default: 224).

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224).
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    transform = get_inference_transform(image_size)
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path: str, model_name: str, device=None):
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        model_name: Architecture name ('resnet50', 'efficientnet_b0', 'mobilenetv2').
        device: torch device (auto-detected if None).

    Returns:
        Loaded model on the specified device.
    """
    if device is None:
        device = get_device()

    # Create model architecture (no pretrained weights — we load from checkpoint)
    model = get_model(model_name, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[INFO] Model loaded: {model_name}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Best epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Val F1:     {checkpoint.get('val_f1', '?')}")

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, image_path: str, device=None):
    """
    Predict gender from a single facial image.

    Args:
        model: Loaded PyTorch model (in eval mode).
        image_path: Path to the image.
        device: torch device.

    Returns:
        dict with:
            - 'class': int (0=Male, 1=Female per problem statement)
            - 'label': str ('Male' or 'Female')
            - 'probability': float (probability of the predicted class)
            - 'raw_score': float (raw sigmoid output)
    """
    if device is None:
        device = next(model.parameters()).device

    tensor = preprocess_image(image_path)
    tensor = tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        raw_score = output.item()  # Probability from sigmoid

    # ImageFolder mapping: Female=0, Male=1 (alphabetical)
    # raw_score > 0.5 → class 1 → Male (ImageFolder)
    if raw_score > 0.5:
        imagefolder_class = 1
        label = "Male"
        confidence = raw_score
    else:
        imagefolder_class = 0
        label = "Female"
        confidence = 1 - raw_score

    # Problem statement mapping: Male=0, Female=1
    problem_class = LABEL_TO_PROBLEM[label]

    return {
        "class": problem_class,
        "label": label,
        "probability": round(float(confidence), 4),
        "raw_score": round(float(raw_score), 4),
    }


# ============================================================
# HIGH-LEVEL API
# ============================================================

class GenderClassifier:
    """
    High-level gender classification API.

    Example:
        clf = GenderClassifier("outputs/best_efficientnet_b0.pth", "efficientnet_b0")
        result = clf.predict("face.jpg")
        print(f"Gender: {result['label']} ({result['probability']:.1%})")
    """

    def __init__(self, checkpoint_path: str, model_name: str, device=None):
        self.device = device or get_device()
        self.model = load_model(checkpoint_path, model_name, self.device)
        self.model_name = model_name

    def predict(self, image_path: str) -> dict:
        """Predict gender for a single image."""
        return predict(self.model, image_path, self.device)

    def predict_with_tta(self, image_path: str, n_augmentations: int = 5) -> dict:
        """Predict with Test Time Augmentation for higher accuracy."""
        return predict_with_tta(self.model, image_path, self.device, n_augmentations)

    def predict_batch(self, image_paths: list) -> list:
        """Predict gender for multiple images."""
        return [self.predict(p) for p in image_paths]


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Gender classification inference")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to facial image")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["resnet50", "efficientnet_b0", "mobilenetv2"],
                        help="Model architecture")
    parser.add_argument("--tta", action="store_true",
                        help="Use Test Time Augmentation")
    args = parser.parse_args()

    clf = GenderClassifier(args.checkpoint, args.model)

    if args.tta:
        result = clf.predict_with_tta(args.image)
        print(f"\n[INFO] TTA Prediction:")
    else:
        result = clf.predict(args.image)
        print(f"\n[INFO] Prediction:")

    print(f"  Image:       {args.image}")
    print(f"  Gender:      {result['label']}")
    print(f"  Class:       {result['class']} (0=Male, 1=Female)")
    print(f"  Confidence:  {result['probability']:.1%}")

    if "all_probs" in result:
        print(f"  TTA Probs:   {result['all_probs']}")


if __name__ == "__main__":
    main()
