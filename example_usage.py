"""
Example usage of Urban Landscape Dataset and Augmentation modules.
Demonstrates proper setup for CAIN-GAN training pipeline.
"""

import torch
from torch.utils.data import DataLoader
from dataset import UrbanLandscapeDataset, create_dataloaders
from augmentation import SatelliteToLandscapeAugmentation, get_augmentation_pipeline
from pathlib import Path


def example_basic_loading():
    """Example 1: Basic dataset loading with paired images."""
    print("=" * 60)
    print("Example 1: Basic Dataset Loading")
    print("=" * 60)

    # Directory structure expected:
    # data/
    #   ├── train/
    #   │   ├── A/  (satellite images)
    #   │   └── B/  (landscape designs)
    #   ├── val/
    #   │   ├── A/
    #   │   └── B/
    #   └── test/
    #       ├── A/
    #       └── B/

    data_root = "/path/to/dataset"  # Update this path

    # Create dataset
    train_dataset = UrbanLandscapeDataset(
        data_root=data_root,
        split="train",
        image_size=256,
        paired_format="separate",  # Use separate A/ and B/ folders
        augment=True,
        normalize=True,
    )

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # Iterate over batches
    for batch_idx, batch in enumerate(train_loader):
        imageA = batch["A"]  # Satellite images
        imageB = batch["B"]  # Landscape designs

        print(f"Batch {batch_idx}:")
        print(f"  Satellite images (A): {imageA.shape}")  # (B, 3, 256, 256)
        print(f"  Landscape designs (B): {imageB.shape}")  # (B, 3, 256, 256)
        print(f"  Paths A: {batch['A_path'][:2]}")
        print(f"  Paths B: {batch['B_path'][:2]}")

        if batch_idx == 0:
            break  # Just show first batch


def example_with_augmentation_config():
    """Example 2: Using custom augmentation configuration."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Augmentation Configuration")
    print("=" * 60)

    data_root = "/path/to/dataset"

    train_dataset = UrbanLandscapeDataset(
        data_root=data_root,
        split="train",
        image_size=256,
        paired_format="separate",
        augment=True,
        normalize=True,
    )

    val_dataset = UrbanLandscapeDataset(
        data_root=data_root,
        split="val",
        image_size=256,
        paired_format="separate",
        augment=False,  # No augmentation for validation
        normalize=True,
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    return train_loader, val_loader


def example_create_dataloaders_helper():
    """Example 3: Using the helper function to create all dataloaders."""
    print("\n" + "=" * 60)
    print("Example 3: Using create_dataloaders Helper")
    print("=" * 60)

    data_root = "/path/to/dataset"

    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=16,
        num_workers=4,
        paired=True,
        image_size=256,
        augment=True,
    )

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    if test_loader:
        print(f"Test loader batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader


def example_training_loop():
    """Example 4: Simple training loop structure."""
    print("\n" + "=" * 60)
    print("Example 4: Training Loop Structure")
    print("=" * 60)

    data_root = "/path/to/dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=16,
        num_workers=4,
        paired=True,
        image_size=256,
    )

    # Pseudo-training loop (structure only)
    num_epochs = 100

    for epoch in range(num_epochs):
        # Training phase
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            imageA = batch["A"].to(device)  # Satellite images
            imageB = batch["B"].to(device)  # Landscape designs

            # Forward pass through generator and discriminator
            # generated_B = generator(imageA)
            # g_loss = criterion(generated_B, imageB)
            # d_loss = adversarial_loss(discriminator, imageA, imageB, generated_B)

            # Backward pass and optimization
            # optimizer_g.zero_grad()
            # optimizer_d.zero_grad()
            # (loss calculation and backprop)
            # optimizer_g.step()
            # optimizer_d.step()

            # train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}: Loss={train_loss / (batch_idx + 1):.4f}")

        # Validation phase
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imageA = batch["A"].to(device)
                imageB = batch["B"].to(device)
                # val_loss += validation_step(imageA, imageB)

        print(f"Epoch {epoch} - Val Loss: {val_loss / len(val_loader):.4f}")


def example_augmentation_pipelines():
    """Example 5: Different augmentation pipeline options."""
    print("\n" + "=" * 60)
    print("Example 5: Augmentation Pipeline Options")
    print("=" * 60)

    # CAIN-GAN specific augmentation
    augmentation = get_augmentation_pipeline(
        pipeline_type="cain",
        image_size=256,
        augmentation_level="moderate",
    )
    print("CAIN-GAN augmentation pipeline created")

    # Pix2Pix style
    augmentation_pix2pix = get_augmentation_pipeline(
        pipeline_type="pix2pix",
        image_size=256,
    )
    print("Pix2Pix augmentation pipeline created")

    # CycleGAN style (for unpaired)
    augmentation_cyclegan = get_augmentation_pipeline(
        pipeline_type="cyclegan",
        image_size=256,
    )
    print("CycleGAN augmentation pipeline created")


def example_data_statistics():
    """Example 6: Compute dataset statistics."""
    print("\n" + "=" * 60)
    print("Example 6: Dataset Statistics")
    print("=" * 60)

    data_root = "/path/to/dataset"

    train_dataset = UrbanLandscapeDataset(
        data_root=data_root,
        split="train",
        image_size=256,
        augment=False,
        normalize=False,  # Don't normalize to get actual pixel values
    )

    print(f"Total training samples: {len(train_dataset)}")

    # Compute mean and std for normalization
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    num_samples = 0

    loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)

    print("Computing statistics...")
    for batch in loader:
        imageA = batch["A"] / 255.0  # Normalize to [0, 1]
        imageB = batch["B"] / 255.0

        # Compute statistics
        batch_size = imageA.shape[0]
        for i in range(3):
            channel_sum[i] += imageA[:, i, :, :].sum()
            channel_sum_sq[i] += (imageA[:, i, :, :] ** 2).sum()

        num_samples += batch_size * 256 * 256

    mean = channel_sum / num_samples
    std = torch.sqrt(channel_sum_sq / num_samples - mean**2)

    print(f"\nNormalization statistics (ImageA):")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")


def print_batch_info(batch):
    """Utility function to print batch information."""
    print("\nBatch Information:")
    print(f"  Image A shape: {batch['A'].shape}")  # (B, 3, H, W)
    print(f"  Image B shape: {batch['B'].shape}")
    print(f"  Image A dtype: {batch['A'].dtype}")
    print(f"  Image A range: [{batch['A'].min():.4f}, {batch['A'].max():.4f}]")
    print(f"  Image B range: [{batch['B'].min():.4f}, {batch['B'].max():.4f}]")


if __name__ == "__main__":
    # Uncomment examples to run:

    # example_basic_loading()
    # example_with_augmentation_config()
    # example_create_dataloaders_helper()
    # example_training_loop()
    example_augmentation_pipelines()
    # example_data_statistics()

    print("\n" + "=" * 60)
    print("Dataset and Augmentation Module Ready!")
    print("=" * 60)
