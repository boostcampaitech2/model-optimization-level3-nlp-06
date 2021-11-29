"""PyTorch transforms for data augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import torchvision.transforms as transforms

from src.augmentation.methods import RandAugmentation, SequentialAugmentation
from src.augmentation.transforms import FILLCOLOR, SquarePad

import albumentations as A
from albumentations.pytorch import ToTensorV2

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}


def simple_augment_train(
    dataset: str = "CIFAR10", img_size: float = 32
) -> A.Compose:
    return A.Compose(
                [
                    A.LongestMaxSize(int(img_size * 1.2)),
                    A.PadIfNeeded(int(img_size * 1.2), int(img_size * 1.2), border_mode=0),
                    A.RandomCrop(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize(    
                        DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                        DATASET_NORMALIZE_INFO[dataset]["STD"],
                    ),
                    ToTensorV2(),
                ])

def simple_augment_test(
    dataset: str = "CIFAR10", img_size: float = 32
) -> A.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return A.Compose(
        [
            A.LongestMaxSize(img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=0),
            A.Normalize(    
                        DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                        DATASET_NORMALIZE_INFO[dataset]["STD"],
                    ),
            ToTensorV2(),
        ]
    )


def randaugment_train(
    dataset: str = "CIFAR10",
    img_size: float = 32,
    n_select: int = 2,
    level: int = 14,
    n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            SquarePad(),
            transforms.Resize((img_size, img_size)),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomHorizontalFlip(),
            SequentialAugmentation([("Cutout", 0.8, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )
