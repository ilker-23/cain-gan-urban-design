"""
Patch Extractor — Sliding Window ile 256×256 Patch Üretimi
============================================================
Bütün şehir verisinden CAIN-GAN eğitim patch'leri üretir.

Pipeline:
  1. Şehir BBox'unu meter cinsinden alt-bölgelere (256m × 256m) böl
  2. Her patch için tüm kanalları rasterize et
  3. Yeterli bina içeren patch'leri kaydet
  4. Train/val/test split yap

Kullanım:
    python -m data_collection.patch_extractor --city elazig
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import (
    CITY_BBOXES,
    DATA_DIRS,
    PATCH_CONFIG,
)
from .rasterize import (
    load_geojson,
    build_site_context,
    build_planning_guidance,
    build_buildings_mask,
    build_height_map,
    filter_features_by_bbox,
)

try:
    from PIL import Image
except ImportError:
    raise ImportError("Pillow gerekli: pip install Pillow")


def meters_to_degrees(meters: float, lat: float) -> Tuple[float, float]:
    """Metre cinsinden mesafeyi derece cinsine çevir (yaklaşık)."""
    # 1° lat ≈ 111,000 m
    dlat = meters / 111000.0
    # 1° lon ≈ 111,000 * cos(lat) m
    dlon = meters / (111000.0 * math.cos(math.radians(lat)))
    return dlon, dlat


def generate_patch_bboxes(
    city_bbox: Tuple[float, float, float, float],
    patch_size_m: int,
    stride_m: int,
) -> List[Tuple[float, float, float, float]]:
    """Şehir BBox'unu patch BBox'larına böl."""
    minlon, minlat, maxlon, maxlat = city_bbox
    center_lat = (minlat + maxlat) / 2

    dlon_patch, dlat_patch = meters_to_degrees(patch_size_m, center_lat)
    dlon_stride, dlat_stride = meters_to_degrees(stride_m, center_lat)

    patches = []
    lat = minlat
    while lat + dlat_patch <= maxlat:
        lon = minlon
        while lon + dlon_patch <= maxlon:
            patches.append((lon, lat, lon + dlon_patch, lat + dlat_patch))
            lon += dlon_stride
        lat += dlat_stride

    return patches


def count_buildings_in_bbox(
    building_features: List[Dict],
    bbox: Tuple[float, float, float, float],
) -> int:
    """BBox içinde kaç bina olduğunu say."""
    minlon, minlat, maxlon, maxlat = bbox
    count = 0
    for feat in building_features:
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") in ("Polygon", "MultiPolygon") and coords:
            # Polygon: coords[0] = exterior ring
            # MultiPolygon: coords[0][0] = first polygon's exterior
            ring = coords[0] if geom["type"] == "Polygon" else coords[0][0]
            cx = sum(c[0] for c in ring) / len(ring)
            cy = sum(c[1] for c in ring) / len(ring)
            if minlon <= cx <= maxlon and minlat <= cy <= maxlat:
                count += 1
    return count


def count_buildings_in_inner(
    building_features: List[Dict],
    patch_bbox: Tuple[float, float, float, float],
    inner_ratio: float = 0.4,
) -> int:
    """Sadece tasarım alanı (mask=0 bölgesi) içindeki bina sayısı.

    Bu kritik — model boş tasarım alanını öğrenemez.
    """
    minlon, minlat, maxlon, maxlat = patch_bbox
    dlon = maxlon - minlon
    dlat = maxlat - minlat
    inset_lon = dlon * (1 - inner_ratio) / 2
    inset_lat = dlat * (1 - inner_ratio) / 2
    inner_bbox = (
        minlon + inset_lon,
        minlat + inset_lat,
        maxlon - inset_lon,
        maxlat - inset_lat,
    )
    return count_buildings_in_bbox(building_features, inner_bbox)


def create_design_mask(
    patch_bbox: Tuple[float, float, float, float],
    image_size: int = 256,
    inner_ratio: float = 0.4,
) -> np.ndarray:
    """Mask kanalı: merkez bölge tasarım alanı (0), çevre kontekst (255).

    Args:
        inner_ratio: tasarım alanı oranı (0-1)
    """
    mask = np.full((image_size, image_size), 255, dtype=np.uint8)
    inner = int(image_size * inner_ratio)
    offset = (image_size - inner) // 2
    mask[offset:offset + inner, offset:offset + inner] = 0
    return mask


def split_inside_outside_mask(
    full_buildings: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binaları maske bölgesine göre ikiye ayır.

    Returns:
        (neighboring_outside, target_inside)
    """
    inside = (mask == 0)
    outside = (mask == 255)

    neighboring = np.where(outside, full_buildings, 0).astype(np.uint8)
    target = np.where(inside, full_buildings, 0).astype(np.uint8)

    return neighboring, target


def extract_seismic_patch(
    seismic_full: Optional[np.ndarray],
    seismic_bbox: Optional[Tuple[float, float, float, float]],
    patch_bbox: Tuple[float, float, float, float],
    image_size: int = 256,
) -> np.ndarray:
    """Şehir-ölçekli sismik raster'dan patch çıkar."""
    if seismic_full is None or seismic_bbox is None:
        # Default orta risk
        return np.full((image_size, image_size), 128, dtype=np.uint8)

    H, W = seismic_full.shape[:2]
    sminlon, sminlat, smaxlon, smaxlat = seismic_bbox
    pminlon, pminlat, pmaxlon, pmaxlat = patch_bbox

    x0 = int((pminlon - sminlon) / (smaxlon - sminlon) * W)
    x1 = int((pmaxlon - sminlon) / (smaxlon - sminlon) * W)
    y0 = int((smaxlat - pmaxlat) / (smaxlat - sminlat) * H)
    y1 = int((smaxlat - pminlat) / (smaxlat - sminlat) * H)

    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)

    if x1 <= x0 or y1 <= y0:
        return np.full((image_size, image_size), 128, dtype=np.uint8)

    crop = seismic_full[y0:y1, x0:x1]
    img = Image.fromarray(crop).resize((image_size, image_size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def extract_dem_patch(
    dem_full: Optional[np.ndarray],
    dem_bbox: Optional[Tuple[float, float, float, float]],
    patch_bbox: Tuple[float, float, float, float],
    image_size: int = 256,
) -> np.ndarray:
    """DEM patch çıkar (sismik ile aynı mantık)."""
    return extract_seismic_patch(dem_full, dem_bbox, patch_bbox, image_size)


def save_patch(
    patch_id: str,
    output_root: Path,
    city: str,
    split: str,
    channels: Dict[str, np.ndarray],
):
    """Tüm kanalları ilgili dizinlere kaydet."""
    split_dir = output_root / city / split

    for ch_name, ch_data in channels.items():
        out_path = split_dir / ch_name / f"{patch_id}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if ch_data.ndim == 2:
            Image.fromarray(ch_data, mode="L").save(out_path)
        elif ch_data.ndim == 3 and ch_data.shape[2] == 3:
            Image.fromarray(ch_data, mode="RGB").save(out_path)


def process_city(
    city: str,
    raw_root: Path,
    output_root: Path,
    image_size: int = 256,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, int]:
    """Bir şehir için tüm pipeline'ı koştur."""

    print(f"\n{'=' * 70}")
    print(f"🏗️ PATCH PIPELINE: {city.upper()}")
    print('=' * 70)

    raw_city = raw_root / city

    # 1. GeoJSON'ları yükle
    print("\n📂 GeoJSON dosyaları yükleniyor...")
    buildings = load_geojson(raw_city / "buildings.geojson")

    # Microsoft buildings fallback
    if not buildings:
        ms_path = raw_city / "microsoft_buildings" / "buildings_microsoft.geojson"
        if ms_path.exists():
            buildings = load_geojson(ms_path)
            print(f"   Microsoft buildings kullanılıyor: {len(buildings)}")

    if not buildings:
        print("   ❌ HİÇ BİNA VERİSİ YOK! Önce osm_downloader.py çalıştırın.")
        return {}

    roads = load_geojson(raw_city / "roads.geojson")
    water = load_geojson(raw_city / "water.geojson")
    vegetation = load_geojson(raw_city / "vegetation.geojson")
    landuse = load_geojson(raw_city / "landuse.geojson")

    print(f"   binalar:    {len(buildings)}")
    print(f"   yollar:     {len(roads)}")
    print(f"   su:         {len(water)}")
    print(f"   vejetasyon: {len(vegetation)}")
    print(f"   landuse:    {len(landuse)}")

    # 2. Sismik ve DEM raster'ları yükle
    print("\n🌋 Sismik ve DEM raster'ları yükleniyor...")
    seismic_full = None
    dem_full = None
    city_bbox = CITY_BBOXES[city]["bbox"]

    seismic_path = raw_city / "seismic" / "pga_475yr.png"
    if seismic_path.exists():
        seismic_full = np.array(Image.open(seismic_path).convert("L"))
        print(f"   ✅ Sismik: {seismic_full.shape}")
    else:
        print("   ⚠️ Sismik veri yok, default değer kullanılacak")

    dem_path = raw_city / "dem" / "dem_merged.png"
    if dem_path.exists():
        dem_full = np.array(Image.open(dem_path).convert("L"))
        print(f"   ✅ DEM: {dem_full.shape}")
    else:
        print("   ⚠️ DEM yok, default değer kullanılacak")

    # 3. Patch BBox'larını üret
    print("\n📐 Patch BBox'ları üretiliyor...")
    patch_bboxes = generate_patch_bboxes(
        city_bbox,
        PATCH_CONFIG["size_meters"],
        PATCH_CONFIG["stride_meters"],
    )
    print(f"   Toplam aday patch: {len(patch_bboxes)}")

    # 4. Bina sayısı filtresi
    #
    # SIKI FİLTRELER:
    # - Patch'te toplam ≥ min_buildings (yoğun urban alan)
    # - Tasarım alanında (mask=0) ≥ min_inside_buildings (model öğrenebilsin)
    min_buildings = PATCH_CONFIG["min_buildings_per_patch"]
    min_inside = PATCH_CONFIG.get("min_inside_buildings", 2)

    valid_patches = []
    n_rejected_total = 0
    n_rejected_inside = 0

    for i, pbbox in enumerate(patch_bboxes):
        if i % 100 == 0:
            print(f"   Tarama: {i}/{len(patch_bboxes)}")
        n_total = count_buildings_in_bbox(buildings, pbbox)
        if n_total < min_buildings:
            n_rejected_total += 1
            continue
        n_inside = count_buildings_in_inner(buildings, pbbox, inner_ratio=0.4)
        if n_inside < min_inside:
            n_rejected_inside += 1
            continue
        valid_patches.append((i, pbbox, n_total))

    print(f"\n   📊 Filtreleme sonucu:")
    print(f"      Toplam aday:     {len(patch_bboxes)}")
    print(f"      Az bina (red):    {n_rejected_total}")
    print(f"      Boş merkez (red): {n_rejected_inside}")
    print(f"      ✅ Geçerli:       {len(valid_patches)}")

    if not valid_patches:
        print("\n   ❌ HİÇ GEÇERLİ PATCH YOK!")
        print("   Olası sebepler:")
        print("     1. BBox çok geniş (kırsal alan)")
        print("     2. min_buildings_per_patch çok yüksek")
        print("     3. OSM'de bu bölgenin verisi eksik")
        print("\n   Çözüm: config.py'da CITY_BBOXES[city]['bbox']'i daha dar yapın")
        print("          veya PATCH_CONFIG['min_buildings_per_patch']'i düşürün")
        return {}

    # 5. Split
    random.seed(seed)
    random.shuffle(valid_patches)
    n_total = len(valid_patches)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    splits = {
        "train": valid_patches[:n_train],
        "val": valid_patches[n_train:n_train + n_val],
        "test": valid_patches[n_train + n_val:],
    }

    print("\n📊 Split:")
    for sp, patches in splits.items():
        print(f"   {sp}: {len(patches)} patch")

    # 6. Patch işleme
    print("\n🎨 Patch'ler işleniyor...")
    stats = {"train": 0, "val": 0, "test": 0}

    for split, patches in splits.items():
        for idx, (orig_idx, pbbox, n_bldg) in enumerate(patches):
            if idx % 50 == 0:
                print(f"   {split}: {idx}/{len(patches)}")

            patch_id = f"{city}_{split}_{orig_idx:06d}"

            # Komşu binaları geniş alandan al (border'da kesilmeyi önlemek için)
            buffer_dlon, buffer_dlat = meters_to_degrees(
                PATCH_CONFIG["buffer_meters"],
                (pbbox[1] + pbbox[3]) / 2,
            )
            buffer_bbox = (
                pbbox[0] - buffer_dlon, pbbox[1] - buffer_dlat,
                pbbox[2] + buffer_dlon, pbbox[3] + buffer_dlat,
            )

            local_buildings = filter_features_by_bbox(buildings, buffer_bbox)
            local_roads = filter_features_by_bbox(roads, buffer_bbox)
            local_water = filter_features_by_bbox(water, buffer_bbox)
            local_veg = filter_features_by_bbox(vegetation, buffer_bbox)
            local_landuse = filter_features_by_bbox(landuse, buffer_bbox)

            # Kanalları üret
            site = build_site_context(local_roads, local_water, local_veg,
                                       pbbox, (image_size, image_size))
            # ⚡ Geofabrik landuse zayıfsa building-based inference devreye girer
            planning = build_planning_guidance(
                local_landuse, pbbox, (image_size, image_size),
                building_features=local_buildings,
            )
            all_buildings = build_buildings_mask(local_buildings, pbbox, (image_size, image_size))
            mask = create_design_mask(pbbox, image_size)

            neighboring, footprint_target = split_inside_outside_mask(all_buildings, mask)

            height_full = build_height_map(local_buildings, pbbox, (image_size, image_size))
            _, height_target = split_inside_outside_mask(height_full, mask)

            seismic_patch = extract_seismic_patch(seismic_full, city_bbox, pbbox, image_size)
            dem_patch = extract_dem_patch(dem_full, city_bbox, pbbox, image_size)

            channels = {
                "site_context": site,
                "planning_guidance": planning,
                "neighboring_footprints": neighboring,
                "mask": mask,
                "seismic": seismic_patch,
                "dem": dem_patch,
                "footprint_target": footprint_target,
                "height_target": height_target,
            }

            save_patch(patch_id, output_root, city, split, channels)
            stats[split] += 1

    print(f"\n✅ {city.upper()} pipeline tamamlandı:")
    for sp, count in stats.items():
        print(f"   {sp}: {count} patch")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Patch Extractor")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"], required=True)
    parser.add_argument("--raw_root", type=str, default=DATA_DIRS["raw"])
    parser.add_argument("--output", type=str, default=DATA_DIRS["patches"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]
    all_stats = {}
    for city in cities:
        all_stats[city] = process_city(
            city,
            Path(args.raw_root),
            Path(args.output),
            args.image_size,
            seed=args.seed,
        )

    print("\n\n" + "=" * 70)
    print("📊 ÖZET")
    print("=" * 70)
    for city, stats in all_stats.items():
        print(f"\n{city}:")
        for sp, count in stats.items():
            print(f"  {sp}: {count}")


if __name__ == "__main__":
    main()
