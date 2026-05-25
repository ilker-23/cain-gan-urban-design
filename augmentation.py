"""
Advanced augmentation strategies for satellite image to landscape design translation.
Implements domain-specific augmentations for urban landscape GAN training.
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Callable, Optional
import random


class SatelliteToLandscapeAugmentation:
    """
    Specialized augmentation pipeline for satellite-to-design translation.
    Maintains spatial consistency between input (A) and target (B) images.
    """

    def __init__(
        self,
        image_size: int = 256,
        augmentation_level: str = "moderate",
        apply_to_target: bool = False,
    ):
        """
        Args:
            image_size: Target image size
            augmentation_level: "light", "moderate", or "heavy"
            apply_to_target: Whether to apply color augmentation to target B images
        """
        self.image_size = image_size
        self.augmentation_level = augmentation_level
        self.apply_to_target = apply_to_target

        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()

    def _build_train_transform(self) -> A.Compose:
        """Build training augmentation pipeline."""

        spatial_transforms = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5, border_mode=1),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
        ]

        if self.augmentation_level == "heavy":
            spatial_transforms.extend(
                [
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                    A.GridDistortion(p=0.2),
                ]
            )

        # Input-specific augmentations (satellite imagery)
        input_augments = [
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MotionBlur(blur_limit=3, p=0.5),
                ],
                p=0.4,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                ],
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.4,
            ),
        ]

        if self.augmentation_level == "heavy":
            input_augments.extend(
                [
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.3),
                ]
            )

        transforms = (
            spatial_transforms
            + input_augments
            + [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        return A.Compose(
            transforms,
            additional_targets={"image_B": "image"},
        )

    def _build_val_transform(self) -> A.Compose:
        """Build validation/test augmentation pipeline (minimal augmentation)."""
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
            additional_targets={"image_B": "image"},
        )

    def __call__(self, imageA: np.ndarray, imageB: np.ndarray, is_train: bool = True) -> dict:
        """
        Apply augmentation to paired images.

        Args:
            imageA: Satellite image (H, W, 3)
            imageB: Landscape design (H, W, 3)
            is_train: Whether in training mode

        Returns:
            Dictionary with augmented images as tensors
        """
        transform = self.train_transform if is_train else self.val_transform

        augmented = transform(image=imageA, image_B=imageB)

        return {
            "A": augmented["image"],
            "B": augmented["image_B"],
        }


class CutMixAugmentation:
    """
    CutMix augmentation for paired images.
    Randomly mixes patches between different image pairs.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(self, imageA_batch: np.ndarray, imageB_batch: np.ndarray) -> tuple:
        """
        Apply CutMix to a batch of paired images.

        Args:
            imageA_batch: Batch of satellite images (B, H, W, 3)
            imageB_batch: Batch of landscape designs (B, H, W, 3)

        Returns:
            (augmented_A, augmented_B, lambda_value)
        """
        batch_size, height, width, _ = imageA_batch.shape

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)

        # Sample random index for mixing
        index = np.random.randint(0, batch_size)

        # Sample random box
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(height * cut_ratio)
        cut_w = int(width * cut_ratio)

        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)

        bbx1 = np.clip(cx - cut_w // 2, 0, width)
        bby1 = np.clip(cy - cut_h // 2, 0, height)
        bbx2 = np.clip(cx + cut_w // 2, 0, width)
        bby2 = np.clip(cy + cut_h // 2, 0, height)

        # Apply CutMix
        imageA_augmented = imageA_batch.copy()
        imageB_augmented = imageB_batch.copy()

        imageA_augmented[:, bby1:bby2, bbx1:bbx2, :] = imageA_batch[index, bby1:bby2, bbx1:bbx2, :]
        imageB_augmented[:, bby1:bby2, bbx1:bbx2, :] = imageB_batch[index, bby1:bby2, bbx1:bbx2, :]

        return imageA_augmented, imageB_augmented, lam


class MosaicAugmentation:
    """
    Mosaic augmentation: combines 4 images into one.
    Useful for learning diverse spatial layouts.
    """

    def __init__(self, image_size: int = 256):
        self.image_size = image_size
        self.mosaic_size = image_size * 2

    def __call__(self, images_A: list, images_B: list) -> tuple:
        """
        Args:
            images_A: List of 4 satellite images (H, W, 3)
            images_B: List of 4 corresponding landscape designs

        Returns:
            (mosaic_A, mosaic_B) both of size (mosaic_size, mosaic_size, 3)
        """
        if len(images_A) != 4 or len(images_B) != 4:
            raise ValueError("Mosaic augmentation requires exactly 4 images")

        mosaic_A = np.zeros((self.mosaic_size, self.mosaic_size, 3), dtype=np.uint8)
        mosaic_B = np.zeros((self.mosaic_size, self.mosaic_size, 3), dtype=np.uint8)

        positions = [(0, 0), (self.image_size, 0), (0, self.image_size), (self.image_size, self.image_size)]

        for i, (x, y) in enumerate(positions):
            img_A = images_A[i]
            img_B = images_B[i]

            # Resize to image_size
            if img_A.shape != (self.image_size, self.image_size, 3):
                img_A = cv2.resize(img_A, (self.image_size, self.image_size))
            if img_B.shape != (self.image_size, self.image_size, 3):
                img_B = cv2.resize(img_B, (self.image_size, self.image_size))

            mosaic_A[y : y + self.image_size, x : x + self.image_size] = img_A
            mosaic_B[y : y + self.image_size, x : x + self.image_size] = img_B

        return mosaic_A, mosaic_B


class RandAugment:
    """
    RandAugment: randomly selects and applies augmentation operations.
    Implementation for paired image translation.
    """

    def __init__(self, num_ops: int = 2, magnitude: int = 9):
        """
        Args:
            num_ops: Number of augmentation operations to apply
            magnitude: Magnitude of augmentation (0-30)
        """
        self.num_ops = num_ops
        self.magnitude = min(magnitude, 30) / 30.0

        self.augmentation_list = [
            self._rotate,
            self._shear,
            self._translate_x,
            self._translate_y,
            self._brightness,
            self._contrast,
            self._saturation,
        ]

    def _rotate(self, img: np.ndarray) -> np.ndarray:
        angle = int(30 * self.magnitude)
        return A.Rotate(limit=angle, border_mode=1)(image=img)["image"]

    def _shear(self, img: np.ndarray) -> np.ndarray:
        magnitude = int(30 * self.magnitude)
        return A.Affine(shear=(-magnitude, magnitude), mode="reflect")(image=img)["image"]

    def _translate_x(self, img: np.ndarray) -> np.ndarray:
        shift = int(img.shape[1] * 0.2 * self.magnitude)
        return A.Affine(translate_percent={"x": (shift, shift)}, mode="reflect")(image=img)["image"]

    def _translate_y(self, img: np.ndarray) -> np.ndarray:
        shift = int(img.shape[0] * 0.2 * self.magnitude)
        return A.Affine(translate_percent={"y": (shift, shift)}, mode="reflect")(image=img)["image"]

    def _brightness(self, img: np.ndarray) -> np.ndarray:
        delta = 0.3 * self.magnitude
        return A.RandomBrightness(limit=delta, p=1.0)(image=img)["image"]

    def _contrast(self, img: np.ndarray) -> np.ndarray:
        delta = 0.3 * self.magnitude
        return A.RandomContrast(limit=delta, p=1.0)(image=img)["image"]

    def _saturation(self, img: np.ndarray) -> np.ndarray:
        delta = int(30 * self.magnitude)
        return A.HueSaturationValue(sat_shift_limit=delta, p=1.0)(image=img)["image"]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentation operations."""
        ops = random.sample(self.augmentation_list, min(self.num_ops, len(self.augmentation_list)))

        for op in ops:
            if random.random() < 0.5:
                img = op(img)

        return img


# Preset augmentation configurations for different scenarios
def get_augmentation_pipeline(
    pipeline_type: str = "cain",
    image_size: int = 256,
    augmentation_level: str = "moderate",
) -> callable:
    """
    Get predefined augmentation pipeline.

    Args:
        pipeline_type: "cain" (for CAIN-GAN style), "pix2pix", "cyclegan", "custom"
        image_size: Image size
        augmentation_level: "light", "moderate", "heavy"

    Returns:
        Augmentation callable
    """

    if pipeline_type == "cain":
        # CAIN-GAN specific augmentation
        return SatelliteToLandscapeAugmentation(
            image_size=image_size,
            augmentation_level=augmentation_level,
        )

    elif pipeline_type == "pix2pix":
        # Standard pix2pix augmentation
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=10, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ],
            additional_targets={"image_B": "image"},
        )

    elif pipeline_type == "cyclegan":
        # CycleGAN style (unpaired)
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    else:
        raise ValueError(f"Unknown pipeline_type: {pipeline_type}")


if __name__ == "__main__":
    print("Augmentation Strategies Module")
    print("=" * 50)
    print("\nAvailable augmentation classes:")
    print("- SatelliteToLandscapeAugmentation: Main pipeline")
    print("- CutMixAugmentation: CutMix mixing strategy")
    print("- MosaicAugmentation: Mosaic combination")
    print("- RandAugment: Random operation selection")
