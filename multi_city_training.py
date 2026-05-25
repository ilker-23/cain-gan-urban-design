"""
Multi-City Seismic-CAIN-GAN Training
=====================================
Elazığ + İstanbul üzerinde sismik farkındalıklı iki-aşamalı eğitim.

Eğitim stratejisi:
  Aşama 1: Footprint Construction (her iki şehir karışık)
    + terrain consistency loss
  Aşama 2: Height Completion
    + seismic-aware loss

Opsiyonel: Cross-city fine-tuning
  Step A: Elazığ'da eğit
  Step B: İstanbul'da fine-tune et
  Step C: Her iki şehirde test et
"""

import argparse
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from multi_city_dataset import (
    create_multi_city_dataloaders,
    TurkishCityDataset,
    CITY_INDEX,
)
from cain_architecture import (
    CAINDiscriminatorFootprint,
    CAINDiscriminatorHeight,
)
from seismic_extension import (
    SeismicCAINGenerator,
    composite_cain_loss,
)


class MultiCityCAINTrainer:
    """
    Elazığ + İstanbul için Seismic-CAIN-GAN eğitici.
    """

    def __init__(
        self,
        data_root: str,
        checkpoint_dir: str = "./checkpoints",
        cities: list = None,
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 4,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        # Loss ağırlıkları
        lambda_rec: float = 100.0,
        lambda_adv: float = 1.0,
        lambda_seismic: float = 10.0,
        lambda_terrain: float = 5.0,
        # Mimari
        ngf: int = 64,
        ndf: int = 64,
        use_seismic: bool = True,
        use_topography: bool = True,
        use_city_embedding: bool = True,
    ):
        self.data_root = data_root
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.cities = cities or ["elazig", "istanbul"]
        self.device = torch.device(device)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

        self.lambda_rec = lambda_rec
        self.lambda_adv = lambda_adv
        self.lambda_seismic = lambda_seismic
        self.lambda_terrain = lambda_terrain

        self.ngf = ngf
        self.ndf = ndf
        self.use_seismic = use_seismic
        self.use_topography = use_topography
        self.use_city_embedding = use_city_embedding

        # Conditional kanal sayısı (probe için 1 örnek)
        probe = TurkishCityDataset(
            data_root=data_root,
            city=self.cities[0],
            split="train",
            stage="footprint",
            use_seismic=use_seismic,
            use_topography=use_topography,
            use_city_embedding=use_city_embedding,
            augment=False,
        )
        self.conditional_channels = probe.num_conditional_channels
        print(f"📊 Conditional kanal sayısı: {self.conditional_channels}")

    # --------------------------------------------------------
    # Stage setup
    # --------------------------------------------------------
    def setup_stage(self, stage: str):
        """Generator + Discriminator + Optimizer kurulumu."""
        print(f"\n{'=' * 70}\nAşama: {stage.upper()}\n{'=' * 70}")

        generator = SeismicCAINGenerator(
            conditional_channels=self.conditional_channels,
            output_channels=1,
            ngf=self.ngf,
            use_city_norm=self.use_city_embedding,
            num_cities=len(self.cities),
        ).to(self.device)

        DiscriminatorCls = (
            CAINDiscriminatorFootprint if stage == "footprint" else CAINDiscriminatorHeight
        )
        discriminator = DiscriminatorCls(input_channels=1, ndf=self.ndf).to(self.device)

        opt_g = optim.Adam(generator.parameters(), lr=self.learning_rate,
                            betas=(self.beta1, self.beta2))
        opt_d = optim.Adam(discriminator.parameters(), lr=self.learning_rate,
                            betas=(self.beta1, self.beta2))

        g_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        print(f"  Generator parametreleri:     {g_params:,}")
        print(f"  Discriminator parametreleri: {d_params:,}")

        return generator, discriminator, opt_g, opt_d

    # --------------------------------------------------------
    # Tek aşama eğitimi (footprint veya height)
    # --------------------------------------------------------
    def train_stage(
        self,
        stage: str,
        num_epochs: int,
        save_interval: int = 5,
    ):
        # Dataloaders
        loaders = create_multi_city_dataloaders(
            data_root=self.data_root,
            cities=self.cities,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            stage=stage,
            use_seismic=self.use_seismic,
            use_topography=self.use_topography,
            use_city_embedding=self.use_city_embedding,
        )

        if "train" not in loaders:
            print(f"⚠️ {stage} aşaması için train verisi bulunamadı, atlanıyor.")
            return None

        train_loader = loaders["train"]
        val_loader = loaders.get("val")

        generator, discriminator, opt_g, opt_d = self.setup_stage(stage)

        target_key = "footprint" if stage == "footprint" else "height"
        # Sismik kanal indeksi (kanal sırası: site(1)+planning(3)+footprints(1)+mask(1)=6, sonra seismic)
        seismic_channel_idx = 6 if self.use_seismic else None
        dem_channel_idx = (7 if self.use_seismic else 6) if self.use_topography else None

        for epoch in range(num_epochs):
            generator.train()
            discriminator.train()

            running = {"L_total": 0.0, "D_loss": 0.0, "count": 0}
            pbar = tqdm(train_loader, desc=f"[{stage}] Epoch {epoch+1}/{num_epochs}")

            for batch in pbar:
                conditional = batch["conditional_inputs"].to(self.device)
                target = batch[target_key].to(self.device)
                city_idx = batch["city_index"].to(self.device)

                # Sismik ve DEM kanallarını ayır
                seismic_map = (
                    conditional[:, seismic_channel_idx:seismic_channel_idx + 1]
                    if seismic_channel_idx is not None else None
                )
                dem = (
                    conditional[:, dem_channel_idx:dem_channel_idx + 1]
                    if dem_channel_idx is not None else None
                )

                B = conditional.shape[0]
                real_lbl = torch.ones(B, 1, 1, 1, device=self.device)
                fake_lbl = torch.zeros(B, 1, 1, 1, device=self.device)

                # ========== Generator ==========
                opt_g.zero_grad()
                pred = generator(conditional, seismic_map=seismic_map, city_index=city_idx)
                d_pred = discriminator(pred)

                loss_g, comps = composite_cain_loss(
                    predicted=pred,
                    target=target,
                    discriminator_output=d_pred,
                    real_label=real_lbl,
                    seismic_map=seismic_map,
                    dem=dem,
                    output_type=stage,
                    lambda_rec=self.lambda_rec,
                    lambda_adv=self.lambda_adv,
                    lambda_seismic=self.lambda_seismic,
                    lambda_terrain=self.lambda_terrain,
                )
                loss_g.backward()
                opt_g.step()

                # ========== Discriminator ==========
                opt_d.zero_grad()
                d_real = discriminator(target)
                d_fake = discriminator(pred.detach())
                loss_d = (nn.functional.binary_cross_entropy(d_real, real_lbl) +
                          nn.functional.binary_cross_entropy(d_fake, fake_lbl)) / 2
                loss_d.backward()
                opt_d.step()

                # Logging
                running["L_total"] += loss_g.item()
                running["D_loss"] += loss_d.item()
                running["count"] += 1

                pbar.set_postfix({
                    "G": f"{loss_g.item():.3f}",
                    "D": f"{loss_d.item():.3f}",
                    **{k: f"{v:.3f}" for k, v in comps.items() if k.startswith("L_") and k != "L_total"},
                })

            avg_g = running["L_total"] / max(running["count"], 1)
            avg_d = running["D_loss"] / max(running["count"], 1)

            # Validation (multi-city)
            if val_loader is not None:
                val_metrics = self._validate(generator, val_loader, target_key,
                                              seismic_channel_idx, dem_channel_idx)
                print(f"  Epoch {epoch+1}: train_G={avg_g:.4f} train_D={avg_d:.4f} "
                      f"val_L1={val_metrics['val_L1']:.4f}")

                # Şehir-bazlı değerlendirme
                for city in self.cities:
                    key = f"val_{city}"
                    if key in loaders:
                        city_metrics = self._validate(generator, loaders[key], target_key,
                                                       seismic_channel_idx, dem_channel_idx)
                        print(f"    {city}: L1={city_metrics['val_L1']:.4f}")

            # Checkpoint
            if (epoch + 1) % save_interval == 0:
                ckpt_path = self.checkpoint_dir / f"{stage}_epoch_{epoch+1}.pth"
                torch.save({
                    "epoch": epoch + 1,
                    "stage": stage,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "config": {
                        "cities": self.cities,
                        "conditional_channels": self.conditional_channels,
                        "use_seismic": self.use_seismic,
                        "use_topography": self.use_topography,
                    },
                }, ckpt_path)
                print(f"  💾 Checkpoint: {ckpt_path.name}")

        return generator

    @torch.no_grad()
    def _validate(self, generator, loader, target_key, seismic_idx, dem_idx) -> Dict:
        generator.eval()
        l1 = nn.L1Loss()
        total, n = 0.0, 0

        for batch in loader:
            conditional = batch["conditional_inputs"].to(self.device)
            target = batch[target_key].to(self.device)
            city_idx = batch["city_index"].to(self.device)

            seismic_map = (
                conditional[:, seismic_idx:seismic_idx + 1] if seismic_idx is not None else None
            )

            pred = generator(conditional, seismic_map=seismic_map, city_index=city_idx)
            total += l1(pred, target).item()
            n += 1

        generator.train()
        return {"val_L1": total / max(n, 1)}

    # --------------------------------------------------------
    # Tam pipeline
    # --------------------------------------------------------
    def train(
        self,
        num_epochs_footprint: int = 50,
        num_epochs_height: int = 50,
        save_interval: int = 5,
    ):
        print("\n" + "=" * 70)
        print(f"🏙️ MULTI-CITY SEISMIC-CAIN-GAN")
        print(f"   Şehirler: {self.cities}")
        print(f"   Cihaz:    {self.device}")
        print(f"   Sismik:   {self.use_seismic} | Topografi: {self.use_topography}")
        print("=" * 70)

        self.train_stage("footprint", num_epochs_footprint, save_interval)
        self.train_stage("height", num_epochs_height, save_interval)

        print("\n🎉 Eğitim tamamlandı!")


def main():
    parser = argparse.ArgumentParser(description="Multi-City Seismic-CAIN-GAN Training")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--cities", nargs="+", default=["elazig", "istanbul"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--epochs_footprint", type=int, default=50)
    parser.add_argument("--epochs_height", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--no_seismic", action="store_true", help="Sismik kanalı devre dışı bırak")
    parser.add_argument("--no_topography", action="store_true", help="DEM kanalını devre dışı bırak")
    parser.add_argument("--no_city_embedding", action="store_true", help="Şehir embedding'i devre dışı bırak")

    args = parser.parse_args()

    trainer = MultiCityCAINTrainer(
        data_root=args.data_root,
        checkpoint_dir=args.checkpoint_dir,
        cities=args.cities,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        use_seismic=not args.no_seismic,
        use_topography=not args.no_topography,
        use_city_embedding=not args.no_city_embedding,
    )

    trainer.train(
        num_epochs_footprint=args.epochs_footprint,
        num_epochs_height=args.epochs_height,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
