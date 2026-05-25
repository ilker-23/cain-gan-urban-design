"""
OpenStreetMap Veri İndirici (Overpass API)
============================================
Elazığ + İstanbul için bina, yol, vejetasyon, su verilerini indirir.

Kullanım:
    python -m data_collection.osm_downloader --city elazig
    python -m data_collection.osm_downloader --city istanbul --district Kadıköy

Çıktı: data_collection/raw/{city}/{layer}.geojson
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict
import requests

from .config import CITY_BBOXES, ELAZIG_DISTRICTS, ISTANBUL_DISTRICTS, DATA_DIRS


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


# Layer-spesifik Overpass sorguları
LAYER_QUERIES = {
    "buildings": """
        (
          way["building"]({area});
          relation["building"]({area});
        );
        out geom;
    """,
    "roads": """
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|service|unclassified)$"]({area});
        );
        out geom;
    """,
    "water": """
        (
          way["natural"="water"]({area});
          way["waterway"~"^(river|stream|canal)$"]({area});
          relation["natural"="water"]({area});
        );
        out geom;
    """,
    "vegetation": """
        (
          way["landuse"~"^(forest|grass|meadow|orchard|farmland)$"]({area});
          way["natural"~"^(wood|scrub|grassland)$"]({area});
          way["leisure"~"^(park|garden|nature_reserve)$"]({area});
          relation["landuse"="forest"]({area});
        );
        out geom;
    """,
    "landuse": """
        (
          way["landuse"~"^(residential|commercial|industrial|retail|mixed)$"]({area});
          relation["landuse"~"^(residential|commercial|industrial|retail|mixed)$"]({area});
        );
        out geom;
    """,
}


def build_area_filter(city: str, district: Optional[str] = None) -> str:
    """Overpass area filtresi oluştur.

    Önce bounding box kullan (hızlı), gerekirse area-based query'e geç.
    """
    if district:
        # İlçe-bazlı (admin_level=6) — daha kesin ama yavaş
        return f'area["name"="{district}"]["boundary"="administrative"]'
    else:
        # BBox-bazlı — hızlı
        bbox = CITY_BBOXES[city]["bbox"]
        minlon, minlat, maxlon, maxlat = bbox
        return f"{minlat},{minlon},{maxlat},{maxlon}"


def build_overpass_query(city: str, layer: str, district: Optional[str] = None) -> str:
    """Tam Overpass QL sorgusu üret."""
    template = LAYER_QUERIES[layer]

    if district:
        area_filter = f'area["name"="{district}"]["boundary"="administrative"]'
        query = f"""
        [out:json][timeout:300];
        {area_filter}->.searchArea;
        {template.replace('{area}', 'area.searchArea')}
        """
    else:
        bbox_str = build_area_filter(city)
        query = f"""
        [out:json][timeout:300];
        {template.replace('{area}', bbox_str)}
        """

    return query


def query_overpass(query: str, max_retries: int = 3) -> Dict:
    """Overpass API'ye sorgu gönder, retry mekanizmasıyla."""
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(max_retries):
            try:
                print(f"  → Endpoint: {endpoint.split('/')[2]} (deneme {attempt + 1})")
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    timeout=600,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"  ⚠️ Hata: {e}")
                if attempt < max_retries - 1:
                    wait = 10 * (attempt + 1)
                    print(f"  ⏳ {wait}s bekleniyor...")
                    time.sleep(wait)
        print(f"  ❌ {endpoint} başarısız, sıradakine geçiliyor")

    raise RuntimeError("Tüm Overpass endpoint'leri başarısız")


def osm_json_to_geojson(osm_data: Dict) -> Dict:
    """OSM JSON çıktısını GeoJSON FeatureCollection'a dönüştür."""
    features = []

    for element in osm_data.get("elements", []):
        elem_type = element.get("type")

        if elem_type == "way" and "geometry" in element:
            coords = [(node["lon"], node["lat"]) for node in element["geometry"]]
            if len(coords) < 2:
                continue

            # Polygon mu LineString mı?
            tags = element.get("tags", {})
            is_polygon = (
                "building" in tags
                or tags.get("area") == "yes"
                or "landuse" in tags
                or tags.get("natural") == "water"
                or tags.get("leisure") in ["park", "garden"]
            )

            if is_polygon and coords[0] != coords[-1]:
                coords.append(coords[0])  # poligonu kapat

            geometry = {
                "type": "Polygon" if is_polygon else "LineString",
                "coordinates": [coords] if is_polygon else coords,
            }

            features.append({
                "type": "Feature",
                "id": f"way/{element['id']}",
                "geometry": geometry,
                "properties": tags,
            })

        elif elem_type == "relation" and "members" in element:
            # Multipolygon işleme — basitleştirilmiş
            outer_ways = []
            for member in element["members"]:
                if member.get("type") == "way" and member.get("role") == "outer":
                    if "geometry" in member:
                        outer_ways.append([(n["lon"], n["lat"]) for n in member["geometry"]])

            if outer_ways:
                tags = element.get("tags", {})
                for way_coords in outer_ways:
                    if len(way_coords) >= 3:
                        if way_coords[0] != way_coords[-1]:
                            way_coords.append(way_coords[0])
                        features.append({
                            "type": "Feature",
                            "id": f"relation/{element['id']}",
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [way_coords],
                            },
                            "properties": tags,
                        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def download_layer(
    city: str,
    layer: str,
    output_dir: Path,
    district: Optional[str] = None,
) -> Path:
    """Tek bir katmanı indir ve kaydet."""
    suffix = f"_{district}" if district else ""
    output_path = output_dir / f"{layer}{suffix}.geojson"

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1e6
        print(f"  ⏩ Zaten var: {output_path.name} ({size_mb:.2f} MB)")
        return output_path

    print(f"\n📥 İndiriliyor: {city}/{layer}{suffix}")
    query = build_overpass_query(city, layer, district)
    osm_data = query_overpass(query)

    print(f"  → {len(osm_data.get('elements', []))} element bulundu")

    geojson = osm_json_to_geojson(osm_data)
    print(f"  → {len(geojson['features'])} feature dönüştürüldü")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False)

    size_mb = output_path.stat().st_size / 1e6
    print(f"  ✅ Kaydedildi: {output_path.name} ({size_mb:.2f} MB)")
    return output_path


def download_city(city: str, output_root: Optional[Path] = None,
                    districts: Optional[list] = None) -> Dict[str, list]:
    """Bir şehir için tüm katmanları indir."""
    if output_root is None:
        output_root = Path(DATA_DIRS["raw"])

    city_dir = output_root / city
    city_dir.mkdir(parents=True, exist_ok=True)

    results = {layer: [] for layer in LAYER_QUERIES}

    layers = list(LAYER_QUERIES.keys())

    if districts is None:
        # Tek seferde tüm şehir
        print(f"\n{'=' * 70}")
        print(f"🌍 ŞEHIR: {city.upper()} (BBox tabanlı)")
        print('=' * 70)
        for layer in layers:
            path = download_layer(city, layer, city_dir)
            results[layer].append(str(path))
            time.sleep(2)  # Overpass'a saygı
    else:
        for district in districts:
            print(f"\n{'=' * 70}")
            print(f"🌍 ŞEHIR: {city.upper()} — İlçe: {district}")
            print('=' * 70)
            for layer in layers:
                try:
                    path = download_layer(city, layer, city_dir, district=district)
                    results[layer].append(str(path))
                    time.sleep(2)
                except Exception as e:
                    print(f"  ❌ Hata ({district}/{layer}): {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="OSM Veri İndirici")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"], required=True)
    parser.add_argument("--district", type=str, default=None,
                        help="Belirli bir ilçe (örn: Kadıköy)")
    parser.add_argument("--all_districts", action="store_true",
                        help="Önceden tanımlı tüm ilçeleri indir")
    parser.add_argument("--output", type=str, default=DATA_DIRS["raw"])

    args = parser.parse_args()

    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]

    for city in cities:
        if args.district:
            districts = [args.district]
        elif args.all_districts:
            districts = ISTANBUL_DISTRICTS if city == "istanbul" else ELAZIG_DISTRICTS
        else:
            districts = None  # tüm şehir BBox

        download_city(city, Path(args.output), districts=districts)

    print("\n🎉 İndirme tamamlandı!")


if __name__ == "__main__":
    main()
