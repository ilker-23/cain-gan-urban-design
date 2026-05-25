"""
CAIN-GAN Training Script
Two-stage progressive learning: Footprint → Height
Based on "Automated site planning using CAIN-GAN model" (Automation in Construction, 2024)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Tuple, Dict
import argparse
from tqdm import tqdm

from cain_dataset import CAINDataset, create_cain_dataloaders
from cain_architecture import (
    CAINGeneratorFootprint,
    CAINDiscriminatorFootprint,
    CAINGeneratorHeight,
    CAINDiscriminatorHeight,
)


class CAINTrainer:
    """
    Trainer for CAIN-GAN two-stage model.
    """

    def __init__(
        self,
        data_root: str,
        checkpoint_dir: str = "./checkpoints",
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 4,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_rec: float = 100.0,  # L1 reconstruction loss weight
        lambda_adv: float = 1.0,    # Adversarial loss weight
    ):
        self.data_root = data_root
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv

        # Create dataloaders
        print(f"Loading datasets from {data_root}...")
        (
            self.footprint_train_loader,
            self.footprint_val_loader,
            self.height_train_loader,
            self.height_val_loader,
        ) = create_cain_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Loss functions
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()

    def setup_footprint_stage(self):
        """Setup generators and discriminators for footprint construction."""
        print("\n" + "=" * 70)
        print("STAGE 1: FOOTPRINT CONSTRUCTION")
        print("=" * 70)

        self.generator_f = CAINGeneratorFootprint(
            conditional_channels=5,
            output_channels=1,
            ngf=64,
        ).to(self.device)

        self.discriminator_f = CAINDiscriminatorFootprint(
            input_channels=1,
            ndf=64,
        ).to(self.device)

        self.optimizer_g_f = optim.Adam(
            self.generator_f.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        self.optimizer_d_f = optim.Adam(
            self.discriminator_f.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        # Count parameters
        g_params = sum(p.numel() for p in self.generator_f.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator_f.parameters() if p.requires_grad)
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")

    def setup_height_stage(self):
        """Setup generators and discriminators for height completion."""
        print("\n" + "=" * 70)
        print("STAGE 2: HEIGHT COMPLETION")
        print("=" * 70)

        self.generator_h = CAINGeneratorHeight(
            conditional_channels=5,
            output_channels=1,
            ngf=64,
        ).to(self.device)

        self.discriminator_h = CAINDiscriminatorHeight(
            input_channels=1,
            ndf=64,
        ).to(self.device)

        self.optimizer_g_h = optim.Adam(
            self.generator_h.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        self.optimizer_d_h = optim.Adam(
            self.discriminator_h.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )

        g_params = sum(p.numel() for p in self.generator_h.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator_h.parameters() if p.requires_grad)
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")

    def train_footprint_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch for footprint construction stage."""
        self.generator_f.train()
        self.discriminator_f.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.footprint_train_loader, desc=f"Footprint Epoch {epoch+1}")

        for batch in pbar:
            conditional = batch["conditional_inputs"].to(self.device)
            target_footprint = batch["footprint"].to(self.device)

            batch_size = conditional.shape[0]

            # Real and fake labels
            real_label = torch.ones(batch_size, 1, 1, 1, device=self.device)
            fake_label = torch.zeros(batch_size, 1, 1, 1, device=self.device)

            # ============ GENERATOR ============
            self.optimizer_g_f.zero_grad()

            # Generate footprints
            generated_footprint = self.generator_f(conditional)

            # Adversarial loss
            d_output = self.discriminator_f(generated_footprint)
            loss_adv = self.criterion_gan(d_output, real_label)

            # Reconstruction loss (L1)
            loss_rec = self.criterion_l1(generated_footprint, target_footprint)

            # Total generator loss
            loss_g = self.lambda_adv * loss_adv + self.lambda_rec * loss_rec
            loss_g.backward()
            self.optimizer_g_f.step()

            # ============ DISCRIMINATOR ============
            self.optimizer_d_f.zero_grad()

            # Real footprints
            d_real = self.discriminator_f(target_footprint)
            loss_d_real = self.criterion_gan(d_real, real_label)

            # Fake footprints (detach to avoid backprop through generator)
            d_fake = self.discriminator_f(generated_footprint.detach())
            loss_d_fake = self.criterion_gan(d_fake, fake_label)

            # Total discriminator loss
            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            self.optimizer_d_f.step()

            # Accumulate losses
            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            num_batches += 1

            pbar.set_postfix({
                'G_loss': loss_g.item(),
                'D_loss': loss_d.item(),
            })

        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches

        return {"G_loss": avg_g_loss, "D_loss": avg_d_loss}

    def train_height_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch for height completion stage."""
        self.generator_h.train()
        self.discriminator_h.train()

        total_g_loss = 0.0
        total_d_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.height_train_loader, desc=f"Height Epoch {epoch+1}")

        for batch in pbar:
            conditional = batch["conditional_inputs"].to(self.device)
            target_height = batch["height"].to(self.device)

            batch_size = conditional.shape[0]
            real_label = torch.ones(batch_size, 1, 1, 1, device=self.device)
            fake_label = torch.zeros(batch_size, 1, 1, 1, device=self.device)

            # ============ GENERATOR ============
            self.optimizer_g_h.zero_grad()

            generated_height = self.generator_h(conditional)

            d_output = self.discriminator_h(generated_height)
            loss_adv = self.criterion_gan(d_output, real_label)

            loss_rec = self.criterion_l1(generated_height, target_height)
            loss_g = self.lambda_adv * loss_adv + self.lambda_rec * loss_rec
            loss_g.backward()
            self.optimizer_g_h.step()

            # ============ DISCRIMINATOR ============
            self.optimizer_d_h.zero_grad()

            d_real = self.discriminator_h(target_height)
            loss_d_real = self.criterion_gan(d_real, real_label)

            d_fake = self.discriminator_h(generated_height.detach())
            loss_d_fake = self.criterion_gan(d_fake, fake_label)

            loss_d = (loss_d_real + loss_d_fake) / 2
            loss_d.backward()
            self.optimizer_d_h.step()

            total_g_loss += loss_g.item()
            total_d_loss += loss_d.item()
            num_batches += 1

            pbar.set_postfix({
                'G_loss': loss_g.item(),
                'D_loss': loss_d.item(),
            })

        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches

        return {"G_loss": avg_g_loss, "D_loss": avg_d_loss}

    @torch.no_grad()
    def validate_footprint(self) -> Dict[str, float]:
        """Validate footprint stage."""
        self.generator_f.eval()

        total_l1_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.footprint_val_loader, desc="Validating footprint"):
            conditional = batch["conditional_inputs"].to(self.device)
            target = batch["footprint"].to(self.device)

            output = self.generator_f(conditional)
            loss = self.criterion_l1(output, target)

            total_l1_loss += loss.item()
            num_batches += 1

        return {"val_L1_loss": total_l1_loss / num_batches}

    @torch.no_grad()
    def validate_height(self) -> Dict[str, float]:
        """Validate height stage."""
        self.generator_h.eval()

        total_l1_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.height_val_loader, desc="Validating height"):
            conditional = batch["conditional_inputs"].to(self.device)
            target = batch["height"].to(self.device)

            output = self.generator_h(conditional)
            loss = self.criterion_l1(output, target)

            total_l1_loss += loss.item()
            num_batches += 1

        return {"val_L1_loss": total_l1_loss / num_batches}

    def save_checkpoint(self, stage: str, epoch: int):
        """Save model checkpoints."""
        checkpoint = {
            "epoch": epoch,
            "stage": stage,
        }

        if stage == "footprint":
            checkpoint["generator"] = self.generator_f.state_dict()
            checkpoint["discriminator"] = self.discriminator_f.state_dict()
            path = self.checkpoint_dir / f"footprint_epoch_{epoch}.pth"
        else:
            checkpoint["generator"] = self.generator_h.state_dict()
            checkpoint["discriminator"] = self.discriminator_h.state_dict()
            path = self.checkpoint_dir / f"height_epoch_{epoch}.pth"

        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    def train(
        self,
        num_epochs_footprint: int = 50,
        num_epochs_height: int = 50,
        save_interval: int = 10,
    ):
        """Train CAIN-GAN with two-stage progression."""
        # Stage 1: Footprint construction
        self.setup_footprint_stage()

        print("\nTraining Footprint Construction Stage...")
        for epoch in range(num_epochs_footprint):
            train_loss = self.train_footprint_epoch(epoch)
            val_loss = self.validate_footprint()

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint("footprint", epoch)

            print(f"Epoch {epoch+1}/{num_epochs_footprint} - "
                  f"Train G: {train_loss['G_loss']:.4f}, "
                  f"Train D: {train_loss['D_loss']:.4f}, "
                  f"Val L1: {val_loss['val_L1_loss']:.4f}")

        # Stage 2: Height completion
        self.setup_height_stage()

        print("\nTraining Height Completion Stage...")
        for epoch in range(num_epochs_height):
            train_loss = self.train_height_epoch(epoch)
            val_loss = self.validate_height()

            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint("height", epoch)

            print(f"Epoch {epoch+1}/{num_epochs_height} - "
                  f"Train G: {train_loss['G_loss']:.4f}, "
                  f"Train D: {train_loss['D_loss']:.4f}, "
                  f"Val L1: {val_loss['val_L1_loss']:.4f}")

        print("\n✓ Training complete!")


def main():
    parser = argparse.ArgumentParser(description="CAIN-GAN Training")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--epochs_footprint", type=int, default=50)
    parser.add_argument("--epochs_height", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=10)

    args = parser.parse_args()

    trainer = CAINTrainer(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
    )

    trainer.train(
        num_epochs_footprint=args.epochs_footprint,
        num_epochs_height=args.epochs_height,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
