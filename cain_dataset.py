"""
CAIN-GAN Specialized Dataset Implementation
Based on "Automated site planning using CAIN-GAN model" (Automation in Construction, 2024)

CAIN-GAN specifics:
- Two-stage: Footprint construction → Height completion
- Multi-channel conditioning (site context, planning guidance, neighboring features)
- Mask-based design area specification
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CAINDataset(Dataset):
    """
    CAIN-GAN Dataset for automated urban site planning.

    Inputs:
    - Site context (road=1, vegetation=2, water=3)
    - Planning guidance (land value, land use types)
    - Neighboring building footprints
    - Mask (0=site to design, 1=surrounding context)

    Outputs:
    - Footprint construction (Stage 1) OR Height completion (Stage 2)
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        stage: str = "footprint",  # "footprint" or "height"
        augment: bool = True,
        normalize: bool = True,
    ):
        """
        Args:
            data_root: Root directory with CAIN-specific structure
            split: "train", "val", or "test"
            image_size: 256x256 standard
            stage: "footprint" (stage 1) or "height" (stage 2)
            augment: Apply augmentation
            normalize: ImageNet normalization
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.stage = stage
        self.augment = augment
        self.normalize = normalize

        if stage not in ["footprint", "height"]:
            raise ValueError(f"stage must be 'footprint' or 'height', got {stage}")

        # CAIN-GAN directory structure
        self.split_dir = self.data_root / split
        self._verify_directory_structure()

    def _verify_directory_structure(self):
        """Verify expected CAIN directory structure."""
        required_dirs = [
            self.split_dir / "site_context",      # Road, vegetation, water
            self.split_dir / "planning_guidance",  # Land value, land use
            self.split_dir / "neighboring_footprints",
            self.split_dir / "mask",
        ]

        if self.stage == "footprint":
            required_dirs.append(self.split_dir / "footprint_target")
        elif self.stage == "height":
            required_dirs.append(self.split_dir / "height_target")

        for dir_path in required_dirs:
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")

        # Get list of sample IDs
        context_dir = self.split_dir / "site_context"
        self.sample_ids = sorted(
            [f.stem for f in context_dir.glob("*") if f.suffix.lower() in [".jpg", ".png", ".tif"]]
        )

        if not self.sample_ids:
            raise ValueError(f"No samples found in {context_dir}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]

        # Load all input channels
        site_context = self._load_channel(
            self.split_dir / "site_context" / f"{sample_id}.png"
        )  # (256, 256, 1)

        planning_guidance = self._load_channel(
            self.split_dir / "planning_guidance" / f"{sample_id}.png"
        )  # (256, 256, C) - multiple land use types

        neighboring_footprints = self._load_channel(
            self.split_dir / "neighboring_footprints" / f"{sample_id}.png"
        )  # (256, 256, 1)

        mask = self._load_channel(
            self.split_dir / "mask" / f"{sample_id}.png"
        )  # (256, 256, 1) - binary mask

        # Load target based on stage
        if self.stage == "footprint":
            target = self._load_channel(
                self.split_dir / "footprint_target" / f"{sample_id}.png"
            )
            target_name = "footprint"
        else:  # height
            target = self._load_channel(
                self.split_dir / "height_target" / f"{sample_id}.png"
            )
            target_name = "height"

        # Concatenate conditional inputs
        # Shape: (256, 256, C) where C = 1 (context) + C_guidance + 1 (footprints) + 1 (mask)
        conditional_inputs = np.concatenate(
            [site_context, planning_guidance, neighboring_footprints, mask],
            axis=2
        )

        # Apply augmentation (consistent between inputs and target)
        if self.augment and self.split == "train":
            augmented = self._apply_augmentation(conditional_inputs, target)
            conditional_tensor = augmented["image"]
            target_tensor = augmented["image_target"]
        else:
            # Validation/test: normalize only
            augmented = self._apply_normalization(conditional_inputs, target)
            conditional_tensor = augmented["image"]
            target_tensor = augmented["image_target"]

        return {
            "conditional_inputs": conditional_tensor,  # (C, 256, 256)
            target_name: target_tensor,                 # (1 or 3, 256, 256)
            "sample_id": sample_id,
            "stage": self.stage,
        }

    def _load_channel(self, path: Path) -> np.ndarray:
        """Load single or multi-channel image."""
        img = Image.open(path)
        if img.mode == "L":
            arr = np.array(img)
            arr = np.expand_dims(arr, axis=2)  # (H, W, 1)
        elif img.mode == "RGB":
            arr = np.array(img)  # (H, W, 3)
        else:
            img = img.convert("RGB")
            arr = np.array(img)

        # Ensure size
        if arr.shape[:2] != (self.image_size, self.image_size):
            img_pil = Image.fromarray(arr.squeeze() if arr.ndim == 3 and arr.shape[2] == 1 else arr)
            img_pil = img_pil.resize((self.image_size, self.image_size), Image.LANCZOS)
            arr = np.array(img_pil)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=2)

        return arr

    def _apply_augmentation(
        self,
        conditional: np.ndarray,
        target: np.ndarray
    ) -> dict:
        """Apply spatial augmentation to maintain consistency."""
        # For CAIN, spatial consistency is critical
        # Augmentation should preserve geometric relationships

        transform = A.Compose(
            [
                # Conservative spatial transforms
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, p=0.3, border_mode=1),

                # No color augmentation for planning data
                # (it's categorical/semantic, not photos)
                A.Normalize(
                    mean=[0.485] * conditional.shape[2],
                    std=[0.229] * conditional.shape[2],
                    max_pixel_value=255.0,
                ) if self.normalize else A.NoOp(),
                ToTensorV2(),
            ],
            additional_targets={"image_target": "image"},
        )

        augmented = transform(image=conditional, image_target=target)
        return augmented

    def _apply_normalization(
        self,
        conditional: np.ndarray,
        target: np.ndarray
    ) -> dict:
        """Apply normalization without spatial augmentation."""
        transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.485] * conditional.shape[2],
                    std=[0.229] * conditional.shape[2],
                    max_pixel_value=255.0,
                ) if self.normalize else A.NoOp(),
                ToTensorV2(),
            ],
            additional_targets={"image_target": "image"},
        )

        normalized = transform(image=conditional, image_target=target)
        return normalized


class CAINProgressiveDataset:
    """
    Wrapper for progressive two-stage training of CAIN-GAN.
    Manages both Footprint and Height datasets simultaneously.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        self.footprint_dataset = CAINDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            stage="footprint",
            augment=(split == "train"),
        )

        self.height_dataset = CAINDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            stage="height",
            augment=(split == "train"),
        )

        self.footprint_loader = torch.utils.data.DataLoader(
            self.footprint_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

        self.height_loader = torch.utils.data.DataLoader(
            self.height_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )

    def get_footprint_loader(self):
        """Get dataloader for footprint construction stage."""
        return self.footprint_loader

    def get_height_loader(self):
        """Get dataloader for height completion stage."""
        return self.height_loader


def create_cain_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Create CAIN-specific dataloaders for both stages.

    Returns:
        (footprint_train, footprint_val, height_train, height_val)
    """
    # Footprint stage loaders
    footprint_train_dataset = CAINDataset(
        data_root=data_root,
        split="train",
        stage="footprint",
        augment=True,
    )
    footprint_val_dataset = CAINDataset(
        data_root=data_root,
        split="val",
        stage="footprint",
        augment=False,
    )

    footprint_train_loader = torch.utils.data.DataLoader(
        footprint_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    footprint_val_loader = torch.utils.data.DataLoader(
        footprint_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Height stage loaders
    height_train_dataset = CAINDataset(
        data_root=data_root,
        split="train",
        stage="height",
        augment=True,
    )
    height_val_dataset = CAINDataset(
        data_root=data_root,
        split="val",
        stage="height",
        augment=False,
    )

    height_train_loader = torch.utils.data.DataLoader(
        height_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    height_val_loader = torch.utils.data.DataLoader(
        height_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        footprint_train_loader,
        footprint_val_loader,
        height_train_loader,
        height_val_loader,
    )


if __name__ == "__main__":
    print("CAIN-GAN Dataset Module")
    print("=" * 70)
    print("\nExpected directory structure:")
    print("""
    data_root/
    ├── train/
    │   ├── site_context/              (256×256, roads/vegetation/water)
    │   ├── planning_guidance/         (256×256, land use/value encoded)
    │   ├── neighboring_footprints/    (256×256, existing buildings)
    │   ├── mask/                      (256×256, binary: 0=design, 1=context)
    │   ├── footprint_target/          (256×256, output footprint)
    │   └── height_target/             (256×256, output heights)
    ├── val/
    │   └── (same structure)
    └── test/
        └── (same structure, height_target optional)
    """)

    print("\n[1] Single-stage training:")
    print("    dataset = CAINDataset(data_root, stage='footprint')")
    print("    dataset = CAINDataset(data_root, stage='height')")

    print("\n[2] Two-stage progressive training:")
    print("    fp_train, fp_val, h_train, h_val = create_cain_dataloaders(data_root)")

    print("\n[3] Batch structure:")
    print("    batch['conditional_inputs']: (B, C, 256, 256)")
    print("    batch['footprint']: (B, 1, 256, 256)")
    print("    batch['height']: (B, 1, 256, 256)")
