

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]



def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Training transforms with data augmentation.

    Pipeline:
        1. Resize to 256px (slight padding for crop)
        2. RandomResizedCrop to 224px (scale 0.8–1.0 = zoom augmentation)
        3. RandomHorizontalFlip (p=0.5)
        4. RandomRotation (±15°)
        5. ColorJitter (brightness ±20%)
        6. ToTensor (scales to [0, 1])
        7. Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Validation/Test transforms — deterministic, no augmentation.

    Pipeline:
        1. Resize to 256px
        2. CenterCrop to 224px
        3. ToTensor
        4. Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])



def load_datasets(data_dir: str = os.path.join(os.path.dirname(__file__), "..", "..", "Dataset"), image_size: int = 224):

    train_dir = os.path.join(data_dir, "Train")
    val_dir = os.path.join(data_dir, "Validation")
    test_dir = os.path.join(data_dir, "Test")

    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms(image_size),
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms(image_size),
    )
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_val_transforms(image_size),
    )

    # Print class-to-index mapping (ImageFolder assigns alphabetically)
    print(f"[INFO] ImageFolder class_to_idx: {train_dataset.class_to_idx}")
    # Expected: {'Female': 0, 'Male': 1} — alphabetical
    # Problem statement wants: Male=0, Female=1
    # We'll handle this mapping in evaluation/inference

    return train_dataset, val_dataset, test_dataset


# ============================================================
# DATALOADER CREATION
# ============================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """
    Create optimized DataLoaders for training, validation, and testing.

    Args:
        train_dataset: Training ImageFolder dataset.
        val_dataset: Validation ImageFolder dataset.
        test_dataset: Test ImageFolder dataset.
        batch_size: Batch size (default 64; increase for larger GPUs).
        num_workers: Number of parallel data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,            # Shuffle training data each epoch
        num_workers=num_workers,
        pin_memory=True,         # Faster GPU transfer
        drop_last=True,          # Drop incomplete last batch for stable BN
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,           # No shuffling for evaluation
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    print(f"[INFO] DataLoaders created:")
    print(f"  Train: {len(train_loader)} batches (batch_size={batch_size})")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ============================================================
# QUICK DATASET INFO
# ============================================================

def print_dataset_info(train_dataset, val_dataset, test_dataset):
    """Print summary of loaded datasets."""
    for name, ds in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        class_counts = {}
        for _, label in ds.samples:
            cls_name = ds.classes[label]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        print(f"\n[{name}] Total: {len(ds)}")
        for cls_name, count in sorted(class_counts.items()):
            pct = count / len(ds) * 100
            print(f"  {cls_name}: {count} ({pct:.1f}%)")
