"""
Veri Kalitesi Teşhis Aracı
============================
Patch üretmeden ÖNCE şehir verisinin yoğunluğunu ve dağılımını gösterir.

Üç görsel üretir:
  1. Bina yoğunluğu ısı haritası (BBox üzerinde)
  2. Aday patch'lerin valid/invalid durumu
  3. Rastgele 8 patch'in tüm kanalları (üretim öncesi quality check)

Kullanım:
    python -m data_collection.diagnose_data --city elazig
    python -m data_collection.diagnose_data --city elazig --post  # patch sonrası
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import CITY_BBOXES, DATA_DIRS, PATCH_CONFIG


def load_features(city: str, raw_root: Path, layer: str) -> List[Dict]:
    path = raw_root / city / f"{layer}.geojson"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f).get("features", [])


def building_centroid(feat: Dict) -> Tuple[float, float]:
    geom = feat.get("geometry", {})
    coords = geom.get("coordinates", [])
    if not coords:
        return (0.0, 0.0)
    if geom.get("type") == "Polygon":
        ring = coords[0]
    elif geom.get("type") == "MultiPolygon":
        ring = coords[0][0]
    else:
        return (0.0, 0.0)
    cx = sum(c[0] for c in ring) / len(ring)
    cy = sum(c[1] for c in ring) / len(ring)
    return cx, cy


def density_heatmap(
    city: str,
    raw_root: Path,
    bins: int = 64,
) -> np.ndarray:
    """Şehir BBox'unda bina yoğunluğu ısı haritası."""
    buildings = load_features(city, raw_root, "buildings")
    bbox = CITY_BBOXES[city]["bbox"]
    minlon, minlat, maxlon, maxlat = bbox

    heat = np.zeros((bins, bins), dtype=np.int32)

    for feat in buildings:
        cx, cy = building_centroid(feat)
        if not (minlon <= cx <= maxlon and minlat <= cy <= maxlat):
            continue
        j = min(int((cx - minlon) / (maxlon - minlon) * bins), bins - 1)
        i = min(int((maxlat - cy) / (maxlat - minlat) * bins), bins - 1)
        heat[i, j] += 1

    return heat


def diagnose_pre_patching(city: str, raw_root: Path,
                            output_dir: Path) -> Dict:
    """Patch üretmeden önce veri kalitesini incele."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️ matplotlib gerekli")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🔍 ÖN TEŞHIS: {city.upper()}")
    print('=' * 70)

    # 1. Katman feature sayıları
    stats = {}
    for layer in ["buildings", "roads", "water", "vegetation", "landuse"]:
        feats = load_features(city, raw_root, layer)
        stats[layer] = len(feats)

    print("\n📊 Katman özetleri:")
    for layer, n in stats.items():
        flag = "✅" if n > 0 else "❌"
        print(f"  {flag} {layer:12s}: {n:7d} feature")

    if stats["buildings"] == 0:
        print("\n❌ Bina yok — Geofabrik çalıştırılmamış olabilir")
        return stats

    # 2. Yoğunluk ısı haritası
    print("\n🌡️ Bina yoğunluğu hesaplanıyor...")
    heat = density_heatmap(city, raw_root, bins=64)
    bbox = CITY_BBOXES[city]["bbox"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    im = axes[0].imshow(heat, cmap='hot', extent=bbox, aspect='auto', origin='upper')
    axes[0].set_title(f'{city.upper()} — Bina Yoğunluğu (64×64 bin)\n'
                       f'Toplam: {stats["buildings"]} bina')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im, ax=axes[0], label='Bina sayısı')

    # 3. Patch valid/invalid haritası
    from .patch_extractor import (
        generate_patch_bboxes, count_buildings_in_bbox,
        count_buildings_in_inner,
    )

    buildings = load_features(city, raw_root, "buildings")
    patch_bboxes = generate_patch_bboxes(
        bbox, PATCH_CONFIG["size_meters"], PATCH_CONFIG["stride_meters"]
    )

    print(f"\n📐 Patch BBox üretimi: {len(patch_bboxes)} aday")
    print(f"   min_buildings_per_patch: {PATCH_CONFIG['min_buildings_per_patch']}")
    print(f"   min_inside_buildings:    {PATCH_CONFIG.get('min_inside_buildings', 2)}")

    valid_count = 0
    valid_centroids = []
    invalid_centroids = []

    for pbbox in patch_bboxes:
        n_total = count_buildings_in_bbox(buildings, pbbox)
        cx = (pbbox[0] + pbbox[2]) / 2
        cy = (pbbox[1] + pbbox[3]) / 2

        if n_total < PATCH_CONFIG["min_buildings_per_patch"]:
            invalid_centroids.append((cx, cy))
            continue
        n_inside = count_buildings_in_inner(buildings, pbbox, 0.4)
        if n_inside < PATCH_CONFIG.get("min_inside_buildings", 2):
            invalid_centroids.append((cx, cy))
            continue
        valid_centroids.append((cx, cy))
        valid_count += 1

    print(f"\n   ✅ Geçerli patch: {valid_count}/{len(patch_bboxes)} "
          f"({100 * valid_count / len(patch_bboxes):.1f}%)")

    if invalid_centroids:
        ic = np.array(invalid_centroids)
        axes[1].scatter(ic[:, 0], ic[:, 1], c='red', s=4, alpha=0.4, label='Reddedildi')
    if valid_centroids:
        vc = np.array(valid_centroids)
        axes[1].scatter(vc[:, 0], vc[:, 1], c='green', s=4, alpha=0.7, label='Geçerli')

    axes[1].set_title(f'Patch Geçerliliği — {valid_count} geçerli / {len(patch_bboxes)} aday')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_xlim(bbox[0], bbox[2])
    axes[1].set_ylim(bbox[1], bbox[3])
    axes[1].legend()
    axes[1].set_aspect('auto')

    plt.tight_layout()
    out_path = output_dir / f"diagnose_{city}_pre.png"
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n💾 Kaydedildi: {out_path}")

    return {
        **stats,
        "n_candidate_patches": len(patch_bboxes),
        "n_valid_patches": valid_count,
        "valid_ratio": valid_count / max(len(patch_bboxes), 1),
    }


def diagnose_post_patching(city: str, data_root: Path,
                             output_dir: Path,
                             n_samples: int = 8) -> None:
    """Patch üretildikten sonra rastgele örnekleri görselleştir."""
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("⚠️ matplotlib + Pillow gerekli")
        return

    train_dir = data_root / city / "train"
    if not (train_dir / "site_context").exists():
        print(f"❌ Patch'ler yok: {train_dir}")
        return

    samples = sorted((train_dir / "site_context").glob("*.png"))
    if not samples:
        print(f"❌ {train_dir}/site_context boş")
        return

    print(f"\n🎨 {len(samples)} patch'ten {n_samples} örnek görselleştiriliyor...")

    random.seed(42)
    chosen = random.sample(samples, min(n_samples, len(samples)))

    channels = ['site_context', 'planning_guidance', 'neighboring_footprints',
                'mask', 'seismic', 'dem', 'footprint_target', 'height_target']

    fig, axes = plt.subplots(n_samples, len(channels), figsize=(28, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, sample_path in enumerate(chosen):
        sid = sample_path.stem
        for col, ch in enumerate(channels):
            ch_path = train_dir / ch / f"{sid}.png"
            if ch_path.exists():
                img = Image.open(ch_path)
                if ch == 'planning_guidance':
                    img = img.convert('RGB')
                    axes[row, col].imshow(img)
                else:
                    cmap = {'seismic': 'hot', 'dem': 'terrain'}.get(ch, 'gray')
                    axes[row, col].imshow(img, cmap=cmap)
                axes[row, col].set_title(f'{ch}', fontsize=9)
            axes[row, col].axis('off')

        # Sample ID en sola yaz
        axes[row, 0].set_ylabel(sid, fontsize=8, rotation=0,
                                  labelpad=40, ha='right', va='center')

    plt.suptitle(f'{city.upper()} — {n_samples} Rastgele Patch (Quality Check)',
                  fontsize=14, y=1.001)
    plt.tight_layout()

    out_path = output_dir / f"diagnose_{city}_post_samples.png"
    plt.savefig(out_path, dpi=80, bbox_inches='tight')
    plt.close()
    print(f"💾 Kaydedildi: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Veri Kalitesi Teşhisi")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"],
                        required=True)
    parser.add_argument("--raw_root", type=str, default=DATA_DIRS["raw"])
    parser.add_argument("--data_root", type=str, default=DATA_DIRS["patches"])
    parser.add_argument("--output", type=str, default="diagnostics")
    parser.add_argument("--post", action="store_true",
                        help="Patch üretildikten sonra örnekleri göster")
    parser.add_argument("--pre", action="store_true",
                        help="Patch üretmeden önce kalite haritası göster")
    args = parser.parse_args()

    out_dir = Path(args.output)
    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]

    # Varsayılan: hem pre hem post
    run_pre = args.pre or not args.post
    run_post = args.post or (not args.pre and not args.post)

    for city in cities:
        if run_pre:
            diagnose_pre_patching(city, Path(args.raw_root), out_dir)
        if run_post:
            diagnose_post_patching(city, Path(args.data_root), out_dir)


if __name__ == "__main__":
    main()
