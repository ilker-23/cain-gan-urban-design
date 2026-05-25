"""
Vector → Raster Dönüşümü
==========================
GeoJSON katmanlarını numpy arrays'e dönüştürür.

Bağımlılıklar: geopandas, shapely, rasterio
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np


def load_geojson(path: Path) -> List[Dict]:
    """GeoJSON dosyasını yükle ve feature listesi döndür."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("features", [])


def coords_to_pixel(
    lon: float,
    lat: float,
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Coğrafi koordinatları piksel koordinatına çevir.

    Args:
        lon, lat: WGS84
        bbox: (minlon, minlat, maxlon, maxlat)
        image_size: (width, height)

    Returns:
        (x, y) — sol-üst orijin
    """
    minlon, minlat, maxlon, maxlat = bbox
    width, height = image_size

    x = int((lon - minlon) / (maxlon - minlon) * width)
    y = int((maxlat - lat) / (maxlat - minlat) * height)  # y ters

    return x, y


def rasterize_polygons(
    features: List[Dict],
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int] = (256, 256),
    fill_value: int = 255,
    background: int = 0,
    line_width: int = 0,
) -> np.ndarray:
    """Poligon feature'larını binary mask'a rasterize et.

    Args:
        features: GeoJSON feature listesi
        bbox: (minlon, minlat, maxlon, maxlat)
        image_size: (W, H)
        fill_value: Poligon iç değeri
        background: Arka plan değeri
        line_width: 0 → dolu poligon, >0 → outline only

    Returns:
        np.array (H, W) uint8
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        raise ImportError("Pillow gerekli: pip install Pillow")

    W, H = image_size
    img = Image.new("L", (W, H), color=background)
    draw = ImageDraw.Draw(img)

    for feature in features:
        geom = feature.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon":
            rings = coords  # exterior + holes
            for ring in rings[:1]:  # sadece dış ring
                pixel_coords = [
                    coords_to_pixel(lon, lat, bbox, image_size)
                    for lon, lat in ring
                ]
                if len(pixel_coords) >= 3:
                    if line_width > 0:
                        draw.line(pixel_coords + [pixel_coords[0]],
                                  fill=fill_value, width=line_width)
                    else:
                        draw.polygon(pixel_coords, fill=fill_value)

        elif gtype == "LineString":
            pixel_coords = [
                coords_to_pixel(lon, lat, bbox, image_size)
                for lon, lat in coords
            ]
            if len(pixel_coords) >= 2:
                draw.line(pixel_coords, fill=fill_value, width=max(line_width, 2))

        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly[:1]:
                    pixel_coords = [
                        coords_to_pixel(lon, lat, bbox, image_size)
                        for lon, lat in ring
                    ]
                    if len(pixel_coords) >= 3:
                        draw.polygon(pixel_coords, fill=fill_value)

    return np.array(img, dtype=np.uint8)


def build_site_context(
    roads_features: List[Dict],
    water_features: List[Dict],
    vegetation_features: List[Dict],
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Site context kanalını oluştur.

    Encoding (grayscale):
      0   = arka plan
      64  = yol
      128 = vejetasyon
      192 = su
    """
    W, H = image_size
    canvas = np.zeros((H, W), dtype=np.uint8)

    # Vejetasyon (en alt katman)
    veg_mask = rasterize_polygons(
        vegetation_features, bbox, image_size, fill_value=128
    )
    canvas[veg_mask > 0] = 128

    # Su (vejetasyondan üstün)
    water_mask = rasterize_polygons(
        water_features, bbox, image_size, fill_value=192
    )
    canvas[water_mask > 0] = 192

    # Yollar (en üst, line)
    road_mask = rasterize_polygons(
        roads_features, bbox, image_size, fill_value=64, line_width=2
    )
    canvas[road_mask > 0] = 64

    return canvas


def build_planning_guidance(
    landuse_features: List[Dict],
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Planning guidance RGB kanalı.

    Encoding:
      R = residential (konut)
      G = commercial (ticaret)
      B = industrial (sanayi)
    """
    W, H = image_size
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    # Kategori-bazlı filtreleme
    categories = {
        "residential": [],
        "commercial": [],
        "industrial": [],
    }

    for feat in landuse_features:
        landuse = feat.get("properties", {}).get("landuse", "").lower()
        if landuse == "residential":
            categories["residential"].append(feat)
        elif landuse in ["commercial", "retail"]:
            categories["commercial"].append(feat)
        elif landuse == "industrial":
            categories["industrial"].append(feat)

    # Her kategori için ayrı kanal
    if categories["residential"]:
        mask = rasterize_polygons(categories["residential"], bbox, image_size, 255)
        rgb[:, :, 0] = mask
    if categories["commercial"]:
        mask = rasterize_polygons(categories["commercial"], bbox, image_size, 255)
        rgb[:, :, 1] = mask
    if categories["industrial"]:
        mask = rasterize_polygons(categories["industrial"], bbox, image_size, 255)
        rgb[:, :, 2] = mask

    return rgb


def build_buildings_mask(
    building_features: List[Dict],
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """Tüm binaları binary mask olarak rasterize et."""
    return rasterize_polygons(building_features, bbox, image_size, fill_value=255)


def build_height_map(
    building_features: List[Dict],
    bbox: Tuple[float, float, float, float],
    image_size: Tuple[int, int] = (256, 256),
    max_height_m: float = 100.0,
) -> np.ndarray:
    """Bina yükseklikleri haritası.

    OSM'den 'height' veya 'building:levels' özelliği okunur.
    Yoksa varsayılan kat sayısı kullanılır.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return np.zeros(image_size, dtype=np.uint8)

    W, H = image_size
    img = Image.new("L", (W, H), color=0)
    draw = ImageDraw.Draw(img)

    for feat in building_features:
        props = feat.get("properties", {})

        # Yükseklik tahmini
        height_m = None

        # 1) Direct height tag
        if "height" in props:
            try:
                height_m = float(str(props["height"]).replace("m", "").strip())
            except ValueError:
                pass

        # 2) building:levels
        if height_m is None and "building:levels" in props:
            try:
                levels = float(props["building:levels"])
                height_m = levels * 3.0  # her kat ~3m varsayım
            except ValueError:
                pass

        # 3) Default
        if height_m is None:
            height_m = 9.0  # ~3 kat varsayım

        height_normalized = min(255, int(height_m / max_height_m * 255))

        geom = feat.get("geometry", {})
        if geom.get("type") == "Polygon":
            rings = geom["coordinates"]
            for ring in rings[:1]:
                pixel_coords = [
                    coords_to_pixel(lon, lat, bbox, image_size)
                    for lon, lat in ring
                ]
                if len(pixel_coords) >= 3:
                    draw.polygon(pixel_coords, fill=height_normalized)

    return np.array(img, dtype=np.uint8)


def filter_features_by_bbox(
    features: List[Dict],
    bbox: Tuple[float, float, float, float],
) -> List[Dict]:
    """BBox dışındaki feature'ları filtrele (hızlandırma için)."""
    minlon, minlat, maxlon, maxlat = bbox
    result = []
    for feat in features:
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])

        if gtype == "Polygon" and coords:
            ring = coords[0]
            cx = sum(c[0] for c in ring) / len(ring)
            cy = sum(c[1] for c in ring) / len(ring)
        elif gtype == "LineString" and coords:
            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)
        else:
            continue

        if minlon <= cx <= maxlon and minlat <= cy <= maxlat:
            result.append(feat)

    return result
