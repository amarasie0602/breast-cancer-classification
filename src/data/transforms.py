"""Image transforms for training and evaluation.

Normalization stats are ImageNet's, matching the pretrained backbone.
"""

from torchvision import transforms
from torchvision.transforms import Compose

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def eval_transform() -> Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform() -> Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def strong_train_transform() -> Compose:
    """Heavier augmentation for the subtype task, where 3 of the 4 classes
    have only 4-6 training patients and the plain train_transform overfits
    within two epochs.

    Two additions matter most here:
    - RandomResizedCrop, so the model sees many sub-regions of each slide
      instead of the same full field every epoch. With this few patients,
      crop diversity is the cheapest source of genuinely new views.
    - Much stronger color jitter. H&E stain intensity varies substantially
      between slides and labs, and with 4 patients per class the model can
      otherwise latch onto one patient's staining as a class shortcut.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.06),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ]
    )
