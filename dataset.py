import os
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class UrbanLandscapeDataset(Dataset):
    """
    Paired image dataset for urban landscape generation from satellite imagery.
    Input (A): Satellite images (256x256)
    Output (B): Landscape/schematic designs (256x256)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        paired_format: str = "separate",  # "separate" or "side_by_side"
        augment: bool = True,
        normalize: bool = True,
    ):
        """
        Args:
            data_root: Root directory containing data
            split: "train", "val", or "test"
            image_size: Target image size (256x256)
            paired_format: "separate" (A/ and B/ folders) or "side_by_side" (combined images)
            augment: Apply data augmentation
            normalize: Normalize to ImageNet statistics
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.paired_format = paired_format
        self.augment = augment
        self.normalize = normalize

        # Verify directory structure
        self.split_dir = self.data_root / split
        if not self.split_dir.exists():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        if self.paired_format == "separate":
            self.imageA_dir = self.split_dir / "A"  # Satellite images
            self.imageB_dir = self.split_dir / "B"  # Landscape designs

            if not self.imageA_dir.exists() or not self.imageB_dir.exists():
                raise ValueError(
                    f"Expected separate A/ and B/ folders in {self.split_dir}"
                )

            self.imageA_paths = sorted(
                [p for p in self.imageA_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".tif", ".tiff"]]
            )

            if not self.imageA_paths:
                raise ValueError(f"No images found in {self.imageA_dir}")

        elif self.paired_format == "side_by_side":
            self.images_dir = self.split_dir
            self.imageA_paths = sorted(
                [p for p in self.images_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".tif", ".tiff"]]
            )

            if not self.imageA_paths:
                raise ValueError(f"No images found in {self.images_dir}")

        else:
            raise ValueError(f"Unknown paired_format: {self.paired_format}")

        self._init_augmentation()

    def _init_augmentation(self):
        """Initialize albumentations pipeline for consistent augmentation on paired images."""

        if self.augment and self.split == "train":
            self.augmentation = A.Compose(
                [
                    # Spatial transformations
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5, border_mode=1),
                    A.Affine(scale=(0.9, 1.1), p=0.5),

                    # Color augmentation (only on input A, not on target B)
                    A.OneOf(
                        [
                            A.GaussNoise(p=0.3),
                            A.GaussianBlur(blur_limit=3, p=0.3),
                            A.MotionBlur(p=0.3),
                        ],
                        p=0.3,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2,
                                contrast_limit=0.2,
                                p=0.5,
                            ),
                            A.RandomGamma(p=0.3),
                        ],
                        p=0.4,
                    ),
                    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),

                    # Normalization and conversion
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ) if self.normalize else A.NoOp(),
                    ToTensorV2(),
                ],
                additional_targets={"image_B": "image"},
            )
        else:
            # Validation/test: only normalization
            self.augmentation = A.Compose(
                [
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ) if self.normalize else A.NoOp(),
                    ToTensorV2(),
                ],
                additional_targets={"image_B": "image"},
            )

    def _load_image(self, path: Path) -> Image.Image:
        """Load and convert image to RGB."""
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def _get_paired_paths(self, imageA_path: Path) -> Tuple[Path, Path]:
        """Get corresponding B image path."""
        if self.paired_format == "separate":
            filename = imageA_path.name
            imageB_path = self.imageB_dir / filename

            if not imageB_path.exists():
                raise FileNotFoundError(
                    f"Corresponding B image not found: {imageB_path}"
                )
            return imageA_path, imageB_path

        else:  # side_by_side
            return imageA_path, imageA_path

    def __len__(self) -> int:
        return len(self.imageA_paths)

    def __getitem__(self, idx: int) -> dict:
        imageA_path = self.imageA_paths[idx]
        imageA_path, imageB_path = self._get_paired_paths(imageA_path)

        # Load images
        imageA = self._load_image(imageA_path)
        imageB = self._load_image(imageB_path)

        # Resize to target size
        imageA = imageA.resize((self.image_size, self.image_size), Image.LANCZOS)
        imageB = imageB.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to numpy arrays
        imageA_np = np.array(imageA)
        imageB_np = np.array(imageB)

        # Apply augmentation (ensures spatial consistency between A and B)
        augmented = self.augmentation(image=imageA_np, image_B=imageB_np)

        imageA_tensor = augmented["image"]
        imageB_tensor = augmented["image_B"]

        return {
            "A": imageA_tensor,  # Satellite image (input)
            "B": imageB_tensor,  # Landscape design (target)
            "A_path": str(imageA_path),
            "B_path": str(imageB_path),
        }


class UnpairedUrbanLandscapeDataset(Dataset):
    """
    Unpaired image dataset for urban landscape generation (CycleGAN style).
    Useful when paired satellite-to-design data is unavailable.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = True,
        normalize: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.normalize = normalize

        self.split_dir = self.data_root / split
        self.imageA_dir = self.split_dir / "A"
        self.imageB_dir = self.split_dir / "B"

        for dir_path in [self.imageA_dir, self.imageB_dir]:
            if not dir_path.exists():
                raise ValueError(f"Directory not found: {dir_path}")

        self.imageA_paths = sorted(
            [p for p in self.imageA_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".tif"]]
        )
        self.imageB_paths = sorted(
            [p for p in self.imageB_dir.glob("*") if p.suffix.lower() in [".jpg", ".png", ".tif"]]
        )

        self._init_augmentation()

    def _init_augmentation(self):
        """Initialize augmentation pipeline."""
        if self.augment and self.split == "train":
            self.augmentation = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=20, p=0.5, border_mode=1),
                    A.RandomBrightnessContrast(p=0.4),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ) if self.normalize else A.NoOp(),
                    ToTensorV2(),
                ],
            )
        else:
            self.augmentation = A.Compose(
                [
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                    ) if self.normalize else A.NoOp(),
                    ToTensorV2(),
                ],
            )

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def __len__(self) -> int:
        return max(len(self.imageA_paths), len(self.imageB_paths))

    def __getitem__(self, idx: int) -> dict:
        idx_A = idx % len(self.imageA_paths)
        idx_B = idx % len(self.imageB_paths)

        imageA = self._load_image(self.imageA_paths[idx_A])
        imageB = self._load_image(self.imageB_paths[idx_B])

        imageA = imageA.resize((self.image_size, self.image_size), Image.LANCZOS)
        imageB = imageB.resize((self.image_size, self.image_size), Image.LANCZOS)

        imageA_np = np.array(imageA)
        imageB_np = np.array(imageB)

        imageA_tensor = self.augmentation(image=imageA_np)["image"]
        imageB_tensor = self.augmentation(image=imageB_np)["image"]

        return {
            "A": imageA_tensor,
            "B": imageB_tensor,
            "A_path": str(self.imageA_paths[idx_A]),
            "B_path": str(self.imageB_paths[idx_B]),
        }


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    paired: bool = True,
    **dataset_kwargs,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_root: Root directory of dataset
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        paired: Use paired dataset (True) or unpaired (False)
        **dataset_kwargs: Additional arguments for dataset

    Returns:
        (train_loader, val_loader, test_loader)
    """

    dataset_class = UrbanLandscapeDataset if paired else UnpairedUrbanLandscapeDataset

    train_dataset = dataset_class(data_root, split="train", augment=True, **dataset_kwargs)
    val_dataset = dataset_class(data_root, split="val", augment=False, **dataset_kwargs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = None
    test_dir = Path(data_root) / "test"
    if test_dir.exists():
        test_dataset = dataset_class(data_root, split="test", augment=False, **dataset_kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Urban Landscape Dataset Module")
    print("=" * 50)
    print("\nDirectory structure expected:")
    print("""
    data_root/
    ├── train/
    │   ├── A/  (satellite images)
    │   └── B/  (landscape designs)
    ├── val/
    │   ├── A/
    │   └── B/
    └── test/
        ├── A/
        └── B/
    """)
