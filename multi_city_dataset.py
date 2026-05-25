"""
Multi-City CAIN-GAN Dataset (Elazığ + İstanbul)
================================================
Türk şehirleri için seismic-aware ve topography-aware veri yükleyici.

Yeni kanallar (NYC CAIN-GAN'a ek olarak):
- Seismic risk (AFAD PGA değerleri, normalize edilmiş)
- DEM (Digital Elevation Model — eğim/yükseklik)
- City embedding (one-hot: 0=Elazığ, 1=İstanbul)

Veri kaynakları:
- Elazığ: OSM + Elazığ Belediyesi + AFAD + MTA
- İstanbul: İBB Açık Veri Portalı + OSM + AFAD + KAF
"""

from pathlib import Path
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Şehir kimlik kodları (one-hot encoding için)
CITY_INDEX = {
    "elazig": 0,
    "istanbul": 1,
}
NUM_CITIES = len(CITY_INDEX)


# Türk imar planı kategorileri (NYC'nin 4 sınıfından farklı)
TURKISH_LAND_USE = {
    "konut": 0,           # Residential
    "ticaret": 1,         # Commercial
    "sanayi": 2,          # Industrial
    "karma": 3,           # Mixed-use
    "yesil_alan": 4,      # Green space
    "kentsel_donusum": 5, # Urban transformation zone
}
NUM_LAND_USE = len(TURKISH_LAND_USE)


class TurkishCityDataset(Dataset):
    """
    Türk şehirleri için CAIN-GAN dataset.

    Conditional inputs (10+ kanal):
      [0]     Site context (yollar, vejetasyon, su)
      [1-6]   Planning guidance (Türk imar kategorileri, one-hot)
      [7]     Neighboring footprints (mevcut binalar)
      [8]     Mask (0=tasarım alanı, 1=kontekst)
      [9]     Seismic risk (AFAD PGA, normalize [0,1])
      [10]    DEM (yükseklik/eğim, normalize [0,1])
      [11-12] City embedding (one-hot: Elazığ/İstanbul)

    Outputs:
      footprint: (1, 256, 256) — bina yer planı
      height:    (1, 256, 256) — bina yükseklikleri
    """

    def __init__(
        self,
        data_root: str,
        city: str,                          # "elazig" veya "istanbul"
        split: str = "train",
        image_size: int = 256,
        stage: str = "footprint",           # "footprint" veya "height"
        use_seismic: bool = True,
        use_topography: bool = True,
        use_city_embedding: bool = True,
        augment: bool = True,
        normalize: bool = True,
    ):
        if city not in CITY_INDEX:
            raise ValueError(f"city '{city}' geçersiz. Kullan: {list(CITY_INDEX.keys())}")
        if stage not in ["footprint", "height"]:
            raise ValueError(f"stage '{stage}' geçersiz. Kullan: footprint/height")

        self.data_root = Path(data_root)
        self.city = city
        self.split = split
        self.image_size = image_size
        self.stage = stage
        self.use_seismic = use_seismic
        self.use_topography = use_topography
        self.use_city_embedding = use_city_embedding
        self.augment = augment
        self.normalize = normalize

        # Şehir + split dizini
        self.city_dir = self.data_root / city / split
        self._verify_structure()
        self._collect_samples()

    def _verify_structure(self):
        """Dizin yapısını doğrula."""
        required = ["site_context", "planning_guidance", "neighboring_footprints", "mask"]
        if self.use_seismic:
            required.append("seismic")
        if self.use_topography:
            required.append("dem")
        if self.stage == "footprint":
            required.append("footprint_target")
        else:
            required.append("height_target")

        missing = []
        for sub in required:
            if not (self.city_dir / sub).exists():
                missing.append(str(self.city_dir / sub))

        if missing:
            raise ValueError(
                f"Eksik dizinler:\n  " + "\n  ".join(missing) +
                f"\n\nBeklenen yapı:\n  {self.city_dir}/{{site_context,planning_guidance,...}}"
            )

    def _collect_samples(self):
        """Sample ID'leri topla."""
        context_dir = self.city_dir / "site_context"
        self.sample_ids = sorted([
            f.stem for f in context_dir.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        ])
        if not self.sample_ids:
            raise ValueError(f"Hiç örnek bulunamadı: {context_dir}")

    def _load_image(self, path: Path, mode: str = "L") -> np.ndarray:
        """Görüntüyü yükle ve 256×256'ya getir."""
        img = Image.open(path)
        if mode == "L" and img.mode != "L":
            img = img.convert("L")
        elif mode == "RGB" and img.mode != "RGB":
            img = img.convert("RGB")

        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)

        arr = np.array(img)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=2)
        return arr

    def _city_embedding_channel(self) -> np.ndarray:
        """Şehir kimliği için sabit dolgu kanalı (one-hot expanded)."""
        city_idx = CITY_INDEX[self.city]
        channels = []
        for i in range(NUM_CITIES):
            fill_value = 255 if i == city_idx else 0
            channel = np.full((self.image_size, self.image_size, 1), fill_value, dtype=np.uint8)
            channels.append(channel)
        return np.concatenate(channels, axis=2)  # (H, W, NUM_CITIES)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]

        # Ana kanallar
        site_context = self._load_image(
            self.city_dir / "site_context" / f"{sample_id}.png", mode="L"
        )
        planning = self._load_image(
            self.city_dir / "planning_guidance" / f"{sample_id}.png", mode="RGB"
        )
        footprints = self._load_image(
            self.city_dir / "neighboring_footprints" / f"{sample_id}.png", mode="L"
        )
        mask = self._load_image(
            self.city_dir / "mask" / f"{sample_id}.png", mode="L"
        )

        channels = [site_context, planning, footprints, mask]

        # Opsiyonel kanallar
        if self.use_seismic:
            seismic = self._load_image(
                self.city_dir / "seismic" / f"{sample_id}.png", mode="L"
            )
            channels.append(seismic)

        if self.use_topography:
            dem = self._load_image(
                self.city_dir / "dem" / f"{sample_id}.png", mode="L"
            )
            channels.append(dem)

        if self.use_city_embedding:
            channels.append(self._city_embedding_channel())

        # Birleştir
        conditional = np.concatenate(channels, axis=2)

        # Target
        target_dir = "footprint_target" if self.stage == "footprint" else "height_target"
        target = self._load_image(
            self.city_dir / target_dir / f"{sample_id}.png", mode="L"
        )

        # Augmentation (paired spatial consistency)
        transform = self._build_transform(conditional.shape[2])
        augmented = transform(image=conditional, image_target=target)

        target_key = "footprint" if self.stage == "footprint" else "height"

        return {
            "conditional_inputs": augmented["image"],   # (C, 256, 256)
            target_key: augmented["image_target"],       # (1, 256, 256)
            "sample_id": sample_id,
            "city": self.city,
            "city_index": CITY_INDEX[self.city],
            "stage": self.stage,
        }

    def _build_transform(self, num_channels: int) -> A.Compose:
        """Augmentation pipeline (spatial-only, semantik veri için renk yok)."""
        transforms = []

        if self.augment and self.split == "train":
            transforms.extend([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=10, p=0.3, border_mode=1),
            ])

        if self.normalize:
            transforms.append(
                A.Normalize(
                    mean=[0.5] * num_channels,
                    std=[0.5] * num_channels,
                    max_pixel_value=255.0,
                )
            )

        transforms.append(ToTensorV2())

        return A.Compose(
            transforms,
            additional_targets={"image_target": "image"},
        )

    @property
    def num_conditional_channels(self) -> int:
        """Toplam conditional kanal sayısı (hesaplama için)."""
        n = 1 + 3 + 1 + 1  # site + planning(RGB) + footprints + mask = 6
        if self.use_seismic:
            n += 1
        if self.use_topography:
            n += 1
        if self.use_city_embedding:
            n += NUM_CITIES
        return n


def create_multi_city_dataloaders(
    data_root: str,
    cities: List[str] = None,
    batch_size: int = 16,
    num_workers: int = 4,
    stage: str = "footprint",
    **dataset_kwargs,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Multi-city dataloader oluşturucu.

    Kullanım:
        loaders = create_multi_city_dataloaders(
            data_root="./data",
            cities=["elazig", "istanbul"],
            batch_size=16,
            stage="footprint",
        )
        # loaders["train"], loaders["val"], loaders["test"] (her biri ConcatDataset)
        # loaders["train_elazig"], loaders["train_istanbul"] (şehir-bazlı)
    """
    if cities is None:
        cities = ["elazig", "istanbul"]

    loaders = {}

    for split in ["train", "val", "test"]:
        city_datasets = []
        for city in cities:
            try:
                ds = TurkishCityDataset(
                    data_root=data_root,
                    city=city,
                    split=split,
                    stage=stage,
                    augment=(split == "train"),
                    **dataset_kwargs,
                )
                city_datasets.append(ds)

                # Şehir-bazlı loader (cross-city evaluation için)
                loaders[f"{split}_{city}"] = torch.utils.data.DataLoader(
                    ds,
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    num_workers=num_workers,
                    pin_memory=True,
                )
            except ValueError as e:
                print(f"⚠️  {city}/{split} atlandı: {e}")

        # Birleşik loader (multi-city training için)
        if city_datasets:
            combined = ConcatDataset(city_datasets) if len(city_datasets) > 1 else city_datasets[0]
            loaders[split] = torch.utils.data.DataLoader(
                combined,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                pin_memory=True,
            )

    return loaders


if __name__ == "__main__":
    print("Multi-City CAIN-GAN Dataset (Elazığ + İstanbul)")
    print("=" * 70)
    print("\nBeklenen dizin yapısı:")
    print("""
    data/
    ├── elazig/
    │   ├── train/
    │   │   ├── site_context/        (yollar/vejetasyon/su)
    │   │   ├── planning_guidance/   (Türk imar kategorileri, RGB)
    │   │   ├── neighboring_footprints/
    │   │   ├── mask/                (0=tasarım, 1=kontekst)
    │   │   ├── seismic/             (AFAD PGA, opsiyonel)
    │   │   ├── dem/                 (DEM, opsiyonel)
    │   │   ├── footprint_target/
    │   │   └── height_target/
    │   ├── val/
    │   └── test/
    │
    └── istanbul/
        ├── train/
        ├── val/
        └── test/
    """)

    print("\n[1] Tek şehir kullanımı:")
    print("    ds = TurkishCityDataset(data_root='./data', city='elazig')")

    print("\n[2] Multi-city kullanımı:")
    print("    loaders = create_multi_city_dataloaders(")
    print("        data_root='./data',")
    print("        cities=['elazig', 'istanbul'],")
    print("    )")

    print("\n[3] Kanal sayısı (varsayılan tüm uzantılarla):")
    print("    1 (site) + 3 (planning) + 1 (footprints) + 1 (mask)")
    print("    + 1 (seismic) + 1 (DEM) + 2 (city embedding) = 10 kanal")
