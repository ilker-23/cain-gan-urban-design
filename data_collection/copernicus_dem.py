"""
Copernicus DEM İndirici (30m global yükseklik)
================================================
ESA Copernicus DEM GLO-30 ürününü Türkiye için indirir.

Erişim yöntemleri:
1. AWS Open Data (Registry of Open Data on AWS) — herkese açık
   https://copernicus-dem-30m.s3.amazonaws.com/
2. Google Earth Engine (kayıt gerektirir)
3. Copernicus DataSpace Ecosystem (kayıt gerektirir)

Bu modül AWS Open Data S3 bucket'ını kullanır (kayıt gerekmez).

Kullanım:
    python -m data_collection.copernicus_dem --city elazig
"""

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple
import requests

from .config import CITY_BBOXES, DATA_DIRS


# AWS S3 base URL
AWS_DEM_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_name(lat: int, lon: int) -> str:
    """Copernicus DEM tile dosya adı.

    Tile'lar 1°x1° kare. Latitude N/S, Longitude E/W prefix.

    Örnek: 38°N, 39°E → Copernicus_DSM_COG_10_N38_00_E039_00_DEM
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    )


def required_tiles(bbox: Tuple[float, float, float, float]) -> List[Tuple[int, int]]:
    """BBox'ı kapsayan 1°×1° tile'ları listele."""
    minlon, minlat, maxlon, maxlat = bbox

    lon_start = math.floor(minlon)
    lon_end = math.floor(maxlon)
    lat_start = math.floor(minlat)
    lat_end = math.floor(maxlat)

    tiles = []
    for lat in range(lat_start, lat_end + 1):
        for lon in range(lon_start, lon_end + 1):
            tiles.append((lat, lon))
    return tiles


def download_tile(lat: int, lon: int, output_dir: Path) -> Optional[Path]:
    """Tek bir DEM tile indir."""
    name = tile_name(lat, lon)
    url = f"{AWS_DEM_BASE}/{name}/{name}.tif"

    output_path = output_dir / f"{name}.tif"
    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"  ⏩ Zaten var: {name}.tif ({size_mb:.2f} MB)")
        return output_path

    try:
        print(f"  📥 İndiriliyor: {name}.tif")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size_mb = output_path.stat().st_size / 1e6
        print(f"  ✅ {name}.tif ({size_mb:.2f} MB)")
        return output_path
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Hata: {e}")
        return None


def download_city(city: str, output_root: Optional[Path] = None) -> List[Path]:
    """Şehir için tüm gerekli DEM tile'larını indir."""
    if output_root is None:
        output_root = Path(DATA_DIRS["raw"])

    out_dir = output_root / city / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)

    bbox = CITY_BBOXES[city]["bbox"]
    tiles = required_tiles(bbox)

    print(f"\n🏔️ Copernicus DEM: {city.upper()}")
    print(f"   BBox: {bbox}")
    print(f"   Gerekli tile sayısı: {len(tiles)}")

    downloaded = []
    for lat, lon in tiles:
        path = download_tile(lat, lon, out_dir)
        if path:
            downloaded.append(path)

    print(f"\n✅ {len(downloaded)}/{len(tiles)} tile indirildi")
    return downloaded


def merge_tiles_to_png(
    tile_paths: List[Path],
    bbox: Tuple[float, float, float, float],
    output_path: Path,
    target_size: int = 2048,
) -> Path:
    """
    GeoTIFF tile'ları PNG olarak birleştir (rasterio'suz fallback).
    Rasterio yoksa, basit bir scale yapılır.

    Daha sonra patch_extractor.py 256x256 küçük parçalara böler.
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("⚠️ Pillow + numpy gerekli")
        return output_path

    minlon, minlat, maxlon, maxlat = bbox

    # Rasterio varsa kullan (daha doğru)
    try:
        import rasterio
        from rasterio.merge import merge
        from rasterio.windows import from_bounds

        sources = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = merge(sources)
        for s in sources:
            s.close()

        # Crop to bbox
        # mosaic shape: (1, H, W)
        data = mosaic[0]

        # Normalize to [0, 255]
        min_val = float(np.nanmin(data))
        max_val = float(np.nanmax(data))
        if max_val - min_val < 1e-6:
            normalized = np.zeros_like(data, dtype=np.uint8)
        else:
            normalized = ((data - min_val) / (max_val - min_val) * 255).astype("uint8")

        # Resize to target_size
        img = Image.fromarray(normalized)
        img = img.resize((target_size, target_size), Image.LANCZOS)
        img.save(output_path)

        meta_path = output_path.with_suffix(".meta.json")
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "source": "Copernicus DEM GLO-30",
                "bbox": list(bbox),
                "elevation_min_m": min_val,
                "elevation_max_m": max_val,
                "target_size": target_size,
            }, f, indent=2)

        print(f"  ✅ Merged: {output_path} ({output_path.stat().st_size / 1e6:.2f} MB)")
        return output_path

    except ImportError:
        print("  ⚠️ rasterio yok, basit merge yapılamıyor")
        print("     Lütfen: pip install rasterio")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Copernicus DEM İndirici")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"], required=True)
    parser.add_argument("--output", type=str, default=DATA_DIRS["raw"])
    parser.add_argument("--merge", action="store_true",
                        help="Tile'ları birleştirip tek PNG üret")
    parser.add_argument("--target_size", type=int, default=2048)
    args = parser.parse_args()

    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]

    for city in cities:
        tiles = download_city(city, Path(args.output))
        if args.merge and tiles:
            out_dir = Path(args.output) / city / "dem"
            merged = out_dir / "dem_merged.png"
            merge_tiles_to_png(
                tiles, CITY_BBOXES[city]["bbox"], merged, args.target_size
            )

    print("\n🎉 Tamamlandı")


if __name__ == "__main__":
    main()
