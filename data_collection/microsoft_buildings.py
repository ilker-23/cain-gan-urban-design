"""
Microsoft Global Building Footprints İndirici
==============================================
Microsoft'un yayınladığı 1.4 milyar bina footprint'inden Türkiye verilerini çeker.

Kaynak: https://github.com/microsoft/GlobalMLBuildingFootprints

Avantajlar:
- OSM'den çok daha kapsamlı (özellikle Anadolu için)
- Otomatik üretilmiş (Bing Maps + AI)
- Yükseklik bilgisi YOK (sadece footprint)

Kullanım:
    python -m data_collection.microsoft_buildings --city elazig
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import requests
import gzip
import io

from .config import CITY_BBOXES, DATA_DIRS


# Microsoft datasets-edge yapısı:
# https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv
DATASET_LINKS_URL = (
    "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"
)


def fetch_dataset_links() -> List[Dict]:
    """Microsoft dataset link CSV'sini al."""
    print("📋 Microsoft Global Building Footprints meta-veri alınıyor...")
    response = requests.get(DATASET_LINKS_URL, timeout=60)
    response.raise_for_status()

    lines = response.text.strip().split("\n")
    header = lines[0].split(",")
    records = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) >= len(header):
            records.append(dict(zip(header, parts)))
    return records


def filter_turkey_tiles(links: List[Dict]) -> List[Dict]:
    """Türkiye'yi içeren tile'ları filtrele."""
    turkey_links = [
        link for link in links
        if link.get("Location", "").strip().lower() in ["turkey", "türkiye"]
    ]
    return turkey_links


def bbox_intersects(
    tile_bbox: tuple,
    target_bbox: tuple,
) -> bool:
    """İki bounding box kesişiyor mu?"""
    tminlon, tminlat, tmaxlon, tmaxlat = tile_bbox
    minlon, minlat, maxlon, maxlat = target_bbox
    return not (tmaxlon < minlon or tminlon > maxlon or
                tmaxlat < minlat or tminlat > maxlat)


def download_and_filter_tile(
    url: str,
    target_bbox: tuple,
    output_path: Path,
) -> int:
    """Bir tile'ı indir ve bbox'a göre filtrele."""
    minlon, minlat, maxlon, maxlat = target_bbox

    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    features = []
    # GZIP'li GeoJSON Lines formatı
    content = response.content
    if url.endswith(".gz") or response.headers.get("Content-Encoding") == "gzip":
        try:
            content = gzip.decompress(content)
        except OSError:
            pass

    text = content.decode("utf-8", errors="ignore")

    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            feature = json.loads(line)
            geom = feature.get("geometry", {})
            if geom.get("type") != "Polygon":
                continue
            coords = geom["coordinates"][0]

            # Bina merkez noktası
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)

            if minlon <= cx <= maxlon and minlat <= cy <= maxlat:
                features.append(feature)
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

    if features:
        geojson = {"type": "FeatureCollection", "features": features}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)

    return len(features)


def download_city(city: str, output_root: Optional[Path] = None) -> Path:
    """Şehir için Microsoft binalarını indir."""
    if output_root is None:
        output_root = Path(DATA_DIRS["raw"])

    city_dir = output_root / city / "microsoft_buildings"
    city_dir.mkdir(parents=True, exist_ok=True)

    output_path = city_dir / "buildings_microsoft.geojson"
    if output_path.exists():
        print(f"⏩ Zaten var: {output_path}")
        return output_path

    bbox = CITY_BBOXES[city]["bbox"]
    print(f"\n📥 Microsoft Buildings: {city.upper()}")
    print(f"   BBox: {bbox}")

    try:
        links = fetch_dataset_links()
        turkey_links = filter_turkey_tiles(links)
        print(f"   Türkiye tile sayısı: {len(turkey_links)}")
    except Exception as e:
        print(f"\n⚠️  Microsoft API erişilemiyor: {e}")
        print("   💡 Manuel indirme:")
        print("      https://github.com/microsoft/GlobalMLBuildingFootprints")
        print("      → Releases → turkey.geojsonl.gz")
        print(f"      Çıktı: {output_path}")
        return output_path

    total_features = 0
    all_features = []

    for i, link in enumerate(turkey_links):
        url = link.get("Url") or link.get("URL") or link.get("url")
        if not url:
            continue

        print(f"   [{i+1}/{len(turkey_links)}] Tile indiriliyor...")
        try:
            tmp_path = city_dir / f"_tmp_tile_{i}.geojson"
            count = download_and_filter_tile(url, bbox, tmp_path)
            if count > 0:
                with open(tmp_path) as f:
                    data = json.load(f)
                all_features.extend(data["features"])
                tmp_path.unlink()
                print(f"     → {count} bina (bbox içinde)")
            total_features += count
        except Exception as e:
            print(f"     ⚠️ Hata: {e}")

    if all_features:
        geojson = {"type": "FeatureCollection", "features": all_features}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False)
        print(f"\n✅ Toplam {total_features} bina kaydedildi: {output_path.name}")
    else:
        print("\n⚠️ Hiç bina bulunamadı. Manuel indirmeyi deneyin.")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Microsoft Global Building Footprints")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"], required=True)
    parser.add_argument("--output", type=str, default=DATA_DIRS["raw"])
    args = parser.parse_args()

    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]
    for city in cities:
        download_city(city, Path(args.output))

    print("\n🎉 Tamamlandı!")


if __name__ == "__main__":
    main()
