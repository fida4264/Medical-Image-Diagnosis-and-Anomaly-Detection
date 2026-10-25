"""
training/v2_baseline/transforms.py
==================================
Modality-aware preprocessing and conservative baseline augmentation for medical images.
Strict rule: Validation and test transforms are 100% deterministic (NO random augmentation).
"""

from typing import Tuple
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    modality: str = "general",
    is_train: bool = False,
    image_size: Tuple[int, int] = (224, 224),
) -> T.Compose:
    """
    Returns appropriate torchvision transform pipeline for the modality and split.

    Args:
        modality: Medical imaging modality name.
        is_train: True for training transforms (with mild augmentation),
                  False for validation/test (strictly deterministic).
        image_size: Target (height, width) for model input.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    modality = modality.lower()

    if not is_train:
        # Deterministic evaluation transform (validation / test)
        return T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # Conservative baseline training augmentation
    transforms_list = [
        T.Resize(image_size),
    ]

    # Modality-specific clinical orientation rules:
    # Brain MRI and Brain CT lateral structures are anatomically diagnostic — do not flip horizontally.
    # Histopathology, skin lesions, ultrasound, and chest X-ray allow mild horizontal flipping.
    if modality in ["histopathology", "skin", "ultrasound", "chest_xray"]:
        transforms_list.append(T.RandomHorizontalFlip(p=0.5))

    # Mild affine transformation: small rotation (+/- 8 degrees), tiny translation (+/- 4%), mild scaling
    transforms_list.append(
        T.RandomAffine(
            degrees=8,
            translate=(0.04, 0.04),
            scale=(0.96, 1.04),
        )
    )

    # Mild lighting adjustment (contrast and brightness)
    if modality in ["skin", "retina", "histopathology", "chest_xray", "ultrasound"]:
        transforms_list.append(
            T.ColorJitter(brightness=0.08, contrast=0.08)
        )

    # Convert to Tensor and ImageNet normalization
    transforms_list.extend([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return T.Compose(transforms_list)
