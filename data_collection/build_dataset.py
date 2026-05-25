"""
End-to-End Dataset Builder
============================
Tüm veri toplama pipeline'ını tek komutla çalıştırır.

Aşamalar:
  1. OSM verisi indir (osm_downloader)
  2. Microsoft buildings indir (microsoft_buildings) [opsiyonel]
  3. AFAD sismik verisi (afad_seismic)
  4. Copernicus DEM (copernicus_dem)
  5. İBB Açık Veri [istanbul için opsiyonel]
  6. Patch'leri üret (patch_extractor)

Kullanım:
    # Tam pipeline (yaklaşık 1-2 saat, internet bağlantısına göre)
    python -m data_collection.build_dataset --city both

    # Sadece Elazığ, sadece OSM + patches
    python -m data_collection.build_dataset --city elazig --steps osm,patches

    # Sismik + DEM atla, hızlıca test et
    python -m data_collection.build_dataset --city elazig --skip seismic,dem
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List

from .config import (
    CITY_BBOXES,
    ELAZIG_DISTRICTS,
    ISTANBUL_DISTRICTS,
    DATA_DIRS,
)


ALL_STEPS = ["geofabrik", "osm", "microsoft", "ibb", "seismic", "dem", "patches"]


def run_step_geofabrik(cities: List[str], raw_root: Path):
    """Geofabrik bulk indir (önerilen, Overpass yerine)."""
    from .geofabrik_downloader import pipeline as geofabrik_pipeline
    return geofabrik_pipeline(cities, output_root=raw_root)


def run_step_osm(city: str, raw_root: Path, districts: List[str] = None):
    """Overpass fallback (Geofabrik başarısızsa)."""
    from .osm_downloader import download_city
    return download_city(city, raw_root, districts=districts)


def run_step_microsoft(city: str, raw_root: Path):
    from .microsoft_buildings import download_city
    return download_city(city, raw_root)


def run_step_ibb(city: str, raw_root: Path):
    if city != "istanbul":
        print(f"⏩ İBB sadece İstanbul için, '{city}' atlanıyor")
        return
    from .ibb_downloader import download_layer, LAYER_KEYWORDS
    output = raw_root / "istanbul" / "ibb"
    for layer in ["zoning", "microzoning", "buildings"]:
        if layer in LAYER_KEYWORDS:
            download_layer(layer, output, max_datasets=2)


def run_step_seismic(city: str, raw_root: Path):
    from .afad_seismic import download_seismic
    return download_seismic(city, raw_root)


def run_step_dem(city: str, raw_root: Path):
    from .copernicus_dem import download_city, merge_tiles_to_png
    tiles = download_city(city, raw_root)
    if tiles:
        out_dir = raw_root / city / "dem"
        merge_tiles_to_png(
            tiles, CITY_BBOXES[city]["bbox"],
            out_dir / "dem_merged.png",
            target_size=2048,
        )


def run_step_patches(city: str, raw_root: Path, output_root: Path):
    from .patch_extractor import process_city
    return process_city(city, raw_root, output_root)


def run_pipeline(
    city: str,
    steps: List[str],
    raw_root: Path,
    output_root: Path,
    use_districts: bool = False,
):
    """Tek şehir için tüm pipeline."""
    print("\n" + "█" * 70)
    print(f"  ŞEHIR: {city.upper()}")
    print(f"  Adımlar: {', '.join(steps)}")
    print("█" * 70)

    districts = None
    if use_districts:
        districts = ISTANBUL_DISTRICTS if city == "istanbul" else ELAZIG_DISTRICTS

    timings = {}

    for step in steps:
        t0 = time.time()
        print(f"\n\n>>> ADIM: {step.upper()}")

        try:
            if step == "geofabrik":
                # Geofabrik tek seferde tüm şehirler için işliyor
                # (Türkiye extract'i ortak)
                continue  # main() level'da işlenecek
            elif step == "osm":
                run_step_osm(city, raw_root, districts)
            elif step == "microsoft":
                run_step_microsoft(city, raw_root)
            elif step == "ibb":
                run_step_ibb(city, raw_root)
            elif step == "seismic":
                run_step_seismic(city, raw_root)
            elif step == "dem":
                run_step_dem(city, raw_root)
            elif step == "patches":
                run_step_patches(city, raw_root, output_root)
            else:
                print(f"  ⚠️ Bilinmeyen adım: {step}")
        except Exception as e:
            print(f"\n  ❌ ADIM BAŞARISIZ: {step}")
            print(f"     {type(e).__name__}: {e}")
            print("  ↪ Sonraki adıma devam ediliyor")

        dt = time.time() - t0
        timings[step] = dt
        print(f"\n  ⏱️ {step} süresi: {dt:.1f}s")

    return timings


def main():
    parser = argparse.ArgumentParser(description="End-to-End Dataset Builder")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"], required=True)
    parser.add_argument("--steps", type=str, default=None,
                        help=f"Virgülle ayrılmış: {','.join(ALL_STEPS)}")
    parser.add_argument("--skip", type=str, default=None,
                        help="Atlanacak adımlar")
    parser.add_argument("--raw_root", type=str, default=DATA_DIRS["raw"])
    parser.add_argument("--output", type=str, default=DATA_DIRS["patches"])
    parser.add_argument("--use_districts", action="store_true",
                        help="İlçe-bazlı OSM indirme (daha yavaş, daha hassas)")

    args = parser.parse_args()

    if args.steps:
        steps = args.steps.split(",")
    else:
        # Varsayılan: geofabrik öncelikli, osm fallback yok
        steps = ["geofabrik", "microsoft", "ibb", "seismic", "dem", "patches"]

    if args.skip:
        skip = set(args.skip.split(","))
        steps = [s for s in steps if s not in skip]

    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]

    raw_root = Path(args.raw_root)
    output_root = Path(args.output)

    # 1) Geofabrik bulk indirme (ortak, tüm şehirler için tek seferde)
    if "geofabrik" in steps:
        print("\n" + "█" * 70)
        print("  ADIM 0: GEOFABRIK BULK İNDİRME")
        print("█" * 70)
        try:
            run_step_geofabrik(cities, raw_root)
        except Exception as e:
            print(f"❌ Geofabrik başarısız: {e}")
            print("   Overpass fallback'e geçiliyor...")
            steps = ["osm" if s == "geofabrik" else s for s in steps]

    total_timings = {}
    for city in cities:
        timings = run_pipeline(
            city, steps, raw_root, output_root, args.use_districts
        )
        total_timings[city] = timings

    print("\n\n" + "=" * 70)
    print("📊 TOPLAM ÖZET")
    print("=" * 70)
    for city, timings in total_timings.items():
        print(f"\n{city.upper()}:")
        for step, dt in timings.items():
            print(f"  {step:12s}: {dt:.1f}s")
        total = sum(timings.values())
        print(f"  {'TOPLAM':12s}: {total:.1f}s ({total/60:.1f} dakika)")

    # Dataset boyutu raporu
    print("\n\n📁 ÜRETİLEN DATASET:")
    for city in cities:
        for split in ["train", "val", "test"]:
            site_ctx_dir = output_root / city / split / "site_context"
            if site_ctx_dir.exists():
                n = len(list(site_ctx_dir.glob("*.png")))
                print(f"  {city}/{split}: {n} patch")

    print("\n🎉 Pipeline tamamlandı!")
    print(f"\nSonraki adım:")
    print(f"  python multi_city_training.py --data_root {output_root}")


if __name__ == "__main__":
    main()
