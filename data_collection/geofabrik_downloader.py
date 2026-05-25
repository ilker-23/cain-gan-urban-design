"""
Geofabrik Türkiye OSM Bulk İndirici
=====================================
Overpass API yerine Geofabrik'in günlük güncellenen Türkiye OSM extract'i.

NEDEN GEOFABRİK?
- ✅ Tek dosya, tek indirme (~600 MB)
- ✅ Günlük güncellenir
- ✅ Rate-limit yok, 406 yok
- ✅ Standart shapefile/PBF formatı
- ✅ Bilimsel araştırmada altın standart

Kaynak: https://download.geofabrik.de/europe/turkey.html

Sunulan dosyalar:
  • turkey-latest-free.shp.zip  ← BİZ BUNU KULLANIYORUZ (~570 MB)
    İçindekiler:
      - gis_osm_buildings_a_free_1.shp    → BİNALAR
      - gis_osm_roads_free_1.shp          → YOLLAR
      - gis_osm_water_a_free_1.shp        → SU (polygon)
      - gis_osm_waterways_free_1.shp      → SU (akarsu)
      - gis_osm_natural_a_free_1.shp      → DOĞAL ALANLAR (orman, vs.)
      - gis_osm_landuse_a_free_1.shp      → ARAZİ KULLANIMI
      - gis_osm_pois_a_free_1.shp         → ÖNEMLİ NOKTALAR
      - gis_osm_traffic_a_free_1.shp      → TRAFİK
      - vd.

Kullanım:
    # Tek seferde tüm Türkiye indir (~600 MB, ~5-10 dk)
    python -m data_collection.geofabrik_downloader

    # Bir şehir için filtrele
    python -m data_collection.geofabrik_downloader --extract elazig
    python -m data_collection.geofabrik_downloader --extract istanbul
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

from .config import CITY_BBOXES, DATA_DIRS


GEOFABRIK_URL = "https://download.geofabrik.de/europe/turkey-latest-free.shp.zip"
GEOFABRIK_PBF_URL = "https://download.geofabrik.de/europe/turkey-latest.osm.pbf"

# Geofabrik shapefile → CAIN-GAN layer eşleştirmesi
SHP_LAYER_MAP = {
    "buildings": ["gis_osm_buildings_a_free_1.shp"],
    "roads":     ["gis_osm_roads_free_1.shp"],
    "water":     ["gis_osm_water_a_free_1.shp",
                  "gis_osm_waterways_free_1.shp"],
    "vegetation": ["gis_osm_natural_a_free_1.shp"],
    "landuse":   ["gis_osm_landuse_a_free_1.shp"],
}


# =======================================================================
# 1. DOWNLOAD
# =======================================================================

def download_turkey_extract(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Geofabrik Türkiye shapefile bundle'ını indir."""
    if output_dir is None:
        output_dir = Path(DATA_DIRS["raw"]) / "_geofabrik"

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "turkey-latest-free.shp.zip"

    if zip_path.exists() and not force:
        size_mb = zip_path.stat().st_size / 1e6
        print(f"⏩ Zaten var: {zip_path.name} ({size_mb:.1f} MB)")
        return zip_path

    print(f"\n📥 Geofabrik Türkiye OSM extract indiriliyor")
    print(f"   URL: {GEOFABRIK_URL}")
    print(f"   Hedef: {zip_path}")
    print(f"   Boyut: ~600 MB (5-10 dakika sürebilir)\n")

    try:
        response = requests.get(GEOFABRIK_URL, stream=True, timeout=600)
        response.raise_for_status()

        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        last_pct = -1

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = int(downloaded / total_size * 100)
                        if pct != last_pct and pct % 5 == 0:
                            mb_done = downloaded / 1e6
                            mb_total = total_size / 1e6
                            print(f"   {pct:3d}% — {mb_done:.0f}/{mb_total:.0f} MB")
                            last_pct = pct

        size_mb = zip_path.stat().st_size / 1e6
        print(f"\n✅ İndirme tamam: {size_mb:.1f} MB")
        return zip_path

    except requests.exceptions.RequestException as e:
        print(f"\n❌ İndirme hatası: {e}")
        print(f"\n💡 Manuel indirme:")
        print(f"   1. {GEOFABRIK_URL} adresini tarayıcıda aç")
        print(f"   2. ZIP'i {zip_path} konumuna kaydet")
        raise


def extract_zip(zip_path: Path, output_dir: Optional[Path] = None) -> Path:
    """ZIP'i çıkar."""
    if output_dir is None:
        output_dir = zip_path.parent / "shapefiles"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    expected_marker = output_dir / "gis_osm_buildings_a_free_1.shp"
    if expected_marker.exists():
        print(f"⏩ Zaten çıkarılmış: {output_dir}")
        return output_dir

    print(f"\n📦 ZIP çıkarılıyor: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        print(f"   İçindeki dosya sayısı: {len(members)}")
        zf.extractall(output_dir)

    files = list(output_dir.glob("*"))
    print(f"✅ Çıkarıldı: {len(files)} dosya — {output_dir}")
    return output_dir


# =======================================================================
# 2. CITY EXTRACTION
# =======================================================================

def shapefile_to_geojson_by_bbox(
    shp_path: Path,
    bbox: Tuple[float, float, float, float],
) -> Dict:
    """Shapefile'ı BBox ile filtreleyip GeoJSON döndür.

    geopandas kullanır (en hızlı), yoksa fiona fallback.
    """
    minlon, minlat, maxlon, maxlat = bbox

    # ---- geopandas (önerilen) ----
    try:
        import geopandas as gpd
        gdf = gpd.read_file(shp_path, bbox=bbox)
        if gdf.empty:
            return {"type": "FeatureCollection", "features": []}

        # WGS84'e çevir (Geofabrik zaten EPSG:4326)
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        return json.loads(gdf.to_json())

    except ImportError:
        pass

    # ---- fiona fallback ----
    try:
        import fiona
        from shapely.geometry import shape, box
        from shapely.prepared import prep

        bbox_shape = prep(box(*bbox))
        features = []

        with fiona.open(shp_path) as src:
            for rec in src:
                geom = rec.get("geometry")
                if not geom:
                    continue
                shp = shape(geom)
                if bbox_shape.intersects(shp):
                    features.append({
                        "type": "Feature",
                        "geometry": geom,
                        "properties": dict(rec.get("properties", {})),
                    })

        return {"type": "FeatureCollection", "features": features}

    except ImportError:
        raise ImportError(
            "geopandas veya fiona gerekli.\n"
            "Yükleyin: pip install geopandas\n"
            "veya:     pip install fiona shapely"
        )


def extract_city(
    city: str,
    shapefile_dir: Path,
    output_root: Optional[Path] = None,
) -> Dict[str, Path]:
    """Bir şehir için tüm katmanları filtrele ve GeoJSON kaydet."""
    if output_root is None:
        output_root = Path(DATA_DIRS["raw"])

    bbox = CITY_BBOXES[city]["bbox"]
    city_dir = output_root / city
    city_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"🌍 ŞEHIR: {city.upper()} (Geofabrik)")
    print(f"   BBox: {bbox}")
    print('=' * 70)

    results = {}

    for layer, shp_files in SHP_LAYER_MAP.items():
        print(f"\n📋 Katman: {layer}")

        output_path = city_dir / f"{layer}.geojson"
        if output_path.exists():
            n_feat = _count_features(output_path)
            print(f"   ⏩ Zaten var: {output_path.name} ({n_feat} feature)")
            results[layer] = output_path
            continue

        # Bu layer için tüm shapefile'ları birleştir
        all_features = []
        for shp_name in shp_files:
            shp_path = shapefile_dir / shp_name
            if not shp_path.exists():
                print(f"   ⚠️  Shapefile yok: {shp_name}")
                continue

            print(f"   → İşleniyor: {shp_name}")
            try:
                geojson = shapefile_to_geojson_by_bbox(shp_path, bbox)
                n = len(geojson.get("features", []))
                all_features.extend(geojson["features"])
                print(f"     {n} feature (BBox içinde)")
            except Exception as e:
                print(f"     ❌ Hata: {e}")

        # Kaydet
        geojson = {"type": "FeatureCollection", "features": all_features}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)

        size_mb = output_path.stat().st_size / 1e6
        print(f"   ✅ Kaydedildi: {output_path.name} ({len(all_features)} feature, {size_mb:.2f} MB)")
        results[layer] = output_path

    return results


def _count_features(geojson_path: Path) -> int:
    """GeoJSON içindeki feature sayısını oku."""
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("features", []))
    except Exception:
        return 0


# =======================================================================
# 3. END-TO-END
# =======================================================================

def pipeline(
    cities: List[str],
    output_root: Optional[Path] = None,
    force_download: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """Tam pipeline: indir → çıkar → şehir filtrele."""

    print("\n" + "█" * 70)
    print("  GEOFABRIK TÜRKİYE PIPELINE")
    print(f"  Şehirler: {cities}")
    print("█" * 70)

    # 1. Türkiye extract indir
    geofabrik_dir = Path(output_root or DATA_DIRS["raw"]) / "_geofabrik"
    zip_path = download_turkey_extract(geofabrik_dir, force=force_download)

    # 2. ZIP çıkar
    shapefile_dir = extract_zip(zip_path)

    # 3. Her şehir için filtrele
    results = {}
    for city in cities:
        results[city] = extract_city(city, shapefile_dir, Path(output_root or DATA_DIRS["raw"]))

    print("\n\n" + "=" * 70)
    print("📊 ÖZET")
    print("=" * 70)
    for city, layers in results.items():
        print(f"\n{city.upper()}:")
        for layer, path in layers.items():
            n = _count_features(path)
            print(f"  {layer:12s}: {n:7d} feature")

    return results


def main():
    parser = argparse.ArgumentParser(description="Geofabrik Türkiye OSM Bulk")
    parser.add_argument("--download_only", action="store_true",
                        help="Sadece Türkiye extract'ini indir, şehir çıkarma yok")
    parser.add_argument("--extract", choices=["elazig", "istanbul", "both"],
                        default="both")
    parser.add_argument("--force", action="store_true",
                        help="ZIP varsa bile yeniden indir")
    parser.add_argument("--output", type=str, default=DATA_DIRS["raw"])
    args = parser.parse_args()

    out = Path(args.output)

    if args.download_only:
        download_turkey_extract(out / "_geofabrik", force=args.force)
        return

    cities = ["elazig", "istanbul"] if args.extract == "both" else [args.extract]
    pipeline(cities, output_root=out, force_download=args.force)

    print("\n🎉 Tamamlandı!")
    print(f"\nSıradaki adım:")
    print(f"  python -m data_collection.patch_extractor --city {args.extract}")


if __name__ == "__main__":
    main()
