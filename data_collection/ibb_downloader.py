"""
İBB Açık Veri Portalı İndirici
================================
İstanbul için kapsamlı CBS verileri (imar, yol, yeşil alan, mikro-bölgeleme).

Portal: https://data.ibb.gov.tr/

İBB API'si CKAN tabanlı:
  - Dataset listesi: /api/3/action/package_list
  - Dataset detayı: /api/3/action/package_show?id=<dataset_id>
  - Kaynak indirme: doğrudan URL

Bu modül popüler CBS dataset'lerini otomatik tespit edip indirir.

Kullanım:
    python -m data_collection.ibb_downloader --layer all
    python -m data_collection.ibb_downloader --layer zoning
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import requests

from .config import DATA_DIRS


IBB_API_BASE = "https://data.ibb.gov.tr/api/3/action"

# Aranılan dataset anahtar kelimeleri (CKAN search için)
LAYER_KEYWORDS = {
    "buildings": ["bina", "yapi", "yapı"],
    "zoning": ["imar", "plan", "1/1000", "1/5000"],
    "roads": ["yol", "ulasim", "ulaşım", "kara yolu"],
    "green": ["yesil alan", "yeşil alan", "park", "agac", "ağaç"],
    "microzoning": ["mikro bölgeleme", "zemin", "deprem", "afet"],
    "districts": ["ilçe", "mahalle", "idari"],
    "landuse": ["arazi kullanım", "kullanim", "kullanım"],
}


def search_datasets(query: str, limit: int = 20) -> List[Dict]:
    """CKAN search ile dataset bul."""
    url = f"{IBB_API_BASE}/package_search"
    params = {"q": query, "rows": limit}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        return []
    return data["result"]["results"]


def get_dataset_resources(dataset_id: str) -> List[Dict]:
    """Bir dataset'in tüm kaynaklarını listele."""
    url = f"{IBB_API_BASE}/package_show"
    params = {"id": dataset_id}
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        return []
    return data["result"].get("resources", [])


def is_geospatial_resource(resource: Dict) -> bool:
    """Coğrafi formatta bir kaynak mı?"""
    fmt = resource.get("format", "").lower()
    return fmt in ["geojson", "shapefile", "shp", "kml", "gpkg", "wfs", "wms"]


def download_resource(resource: Dict, output_path: Path) -> bool:
    """Bir kaynağı indir."""
    url = resource.get("url")
    if not url:
        return False

    try:
        print(f"     📥 {resource.get('name', '?')} ({resource.get('format', '?')})")
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size_mb = output_path.stat().st_size / 1e6
        print(f"     ✅ {output_path.name} ({size_mb:.2f} MB)")
        return True
    except requests.exceptions.RequestException as e:
        print(f"     ⚠️ Hata: {e}")
        return False


def download_layer(
    layer: str,
    output_root: Optional[Path] = None,
    max_datasets: int = 5,
) -> Dict[str, List[str]]:
    """Bir katman için tüm matching dataset'leri indir."""
    if output_root is None:
        output_root = Path(DATA_DIRS["raw"]) / "istanbul" / "ibb"

    output_root.mkdir(parents=True, exist_ok=True)

    results = {"downloaded": [], "skipped": [], "failed": []}

    keywords = LAYER_KEYWORDS.get(layer, [layer])

    seen_datasets = set()

    for kw in keywords:
        print(f"\n🔍 Arama: '{kw}'")
        try:
            datasets = search_datasets(kw)
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ API hatası: {e}")
            continue

        for ds in datasets[:max_datasets]:
            ds_id = ds.get("id") or ds.get("name")
            ds_name = ds.get("title", ds_id)

            if ds_id in seen_datasets:
                continue
            seen_datasets.add(ds_id)

            print(f"\n  📦 {ds_name}")

            resources = ds.get("resources", []) or get_dataset_resources(ds_id)
            geo_resources = [r for r in resources if is_geospatial_resource(r)]

            if not geo_resources:
                print("     ⏩ Coğrafi format yok, atlanıyor")
                results["skipped"].append(ds_name)
                continue

            for res in geo_resources:
                fmt = res.get("format", "bin").lower()
                name = res.get("name", "resource").replace("/", "_").replace(" ", "_")
                out_path = output_root / layer / f"{ds_id}__{name}.{fmt}"

                if out_path.exists():
                    print(f"     ⏩ Zaten var: {out_path.name}")
                    continue

                if download_resource(res, out_path):
                    results["downloaded"].append(str(out_path))
                else:
                    results["failed"].append(str(out_path))

    return results


def main():
    parser = argparse.ArgumentParser(description="İBB Açık Veri İndirici")
    parser.add_argument(
        "--layer",
        choices=list(LAYER_KEYWORDS.keys()) + ["all"],
        default="all",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max_datasets", type=int, default=3,
                        help="Her keyword için maks dataset sayısı")
    args = parser.parse_args()

    layers = list(LAYER_KEYWORDS.keys()) if args.layer == "all" else [args.layer]
    out = Path(args.output) if args.output else None

    summary = {}
    for layer in layers:
        print(f"\n{'=' * 70}")
        print(f"📊 KATMAN: {layer.upper()}")
        print('=' * 70)
        summary[layer] = download_layer(layer, out, args.max_datasets)

    print("\n\n" + "=" * 70)
    print("📋 ÖZET")
    print("=" * 70)
    for layer, res in summary.items():
        print(f"\n{layer}:")
        print(f"  ✅ İndirildi: {len(res['downloaded'])}")
        print(f"  ⏩ Atlandı:   {len(res['skipped'])}")
        print(f"  ❌ Hatalı:    {len(res['failed'])}")

    print("\n💡 Manuel indirme: https://data.ibb.gov.tr/")


if __name__ == "__main__":
    main()
