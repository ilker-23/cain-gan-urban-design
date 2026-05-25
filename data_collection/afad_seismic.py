"""
AFAD Türkiye Deprem Tehlike Haritası İndirici
==============================================
AFAD'dan PGA (Peak Ground Acceleration) ve fay hatları verileri.

Resmi portal: https://tdth.afad.gov.tr/

AFAD API endpoint'i halka kapalı; resmi olarak şu seçenekler var:
1. PDF raporlar (manuel)
2. SHP/GeoJSON export (akademik talep)
3. Üçüncü taraf API'ler (GitHub mirror'lar)

Bu modül:
- TR Tehlike Haritası (statik raster) sunucusundan PNG/TIF çeker (mümkünse)
- Aksi takdirde, kullanıcıya manuel indirme rehberi verir
- Coğrafi indeksleme yapar

Kullanım:
    python -m data_collection.afad_seismic --city elazig
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import requests

from .config import CITY_BBOXES, DATA_DIRS


# AFAD/TDTH halka açık servisler (zaman zaman değişiyor)
AFAD_RESOURCES = {
    # OGC WMS endpoint (akademik kullanım için resmi olmayan)
    "wms_endpoint": "https://tdth.afad.gov.tr/TDTH/api/services",
    # Statik raster (PGA 475 yıl, %10 aşılma olasılığı)
    "pga_475yr_url": (
        "https://tdth.afad.gov.tr/TDTH/api/raster/getPgaRaster"
    ),
}

# Bilinen büyük fay hatları (manuel hard-coded — AFAD harita verisinden)
KNOWN_FAULTS = {
    "elazig": [
        # Doğu Anadolu Fay Zonu (yaklaşık koordinatlar)
        {"name": "DAFZ_Pertek", "lat1": 38.83, "lon1": 39.32, "lat2": 38.78, "lon2": 39.55},
        {"name": "DAFZ_Palu", "lat1": 38.69, "lon1": 39.92, "lat2": 38.62, "lon2": 40.10},
        {"name": "DAFZ_Sivrice", "lat1": 38.45, "lon1": 39.30, "lat2": 38.38, "lon2": 39.55},
    ],
    "istanbul": [
        # Kuzey Anadolu Fayı (Marmara segment)
        {"name": "KAF_Marmara_W", "lat1": 40.78, "lon1": 28.10, "lat2": 40.82, "lon2": 28.70},
        {"name": "KAF_Marmara_E", "lat1": 40.82, "lon1": 28.70, "lat2": 40.85, "lon2": 29.50},
    ],
}


def estimate_pga_from_distance(lat: float, lon: float, city: str) -> float:
    """Fay hattına en yakın mesafeden basit PGA tahmini.

    Bu, AFAD verisi alınamadığında **placeholder**'dir.
    Gerçek araştırmada AFAD'ın resmi PGA haritası kullanılmalıdır.

    Returns:
        PGA tahmini (g cinsinden, 0-1 normalize edilmiş)
    """
    faults = KNOWN_FAULTS.get(city, [])
    if not faults:
        return 0.4  # default orta risk

    min_dist_deg = float("inf")
    for fault in faults:
        # Çizgi parçasına nokta-mesafe (basit, dünya yuvarlağı ihmal edildi)
        x1, y1 = fault["lon1"], fault["lat1"]
        x2, y2 = fault["lon2"], fault["lat2"]
        x0, y0 = lon, lat

        # Çizgiye en yakın nokta projection
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            d = ((x0 - x1) ** 2 + (y0 - y1) ** 2) ** 0.5
        else:
            t = max(0, min(1, ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy)))
            px, py = x1 + t * dx, y1 + t * dy
            d = ((x0 - px) ** 2 + (y0 - py) ** 2) ** 0.5

        min_dist_deg = min(min_dist_deg, d)

    # 1 derece ≈ 111 km. Sabel-Boore zayıflama benzeri sönüm:
    dist_km = min_dist_deg * 111.0

    # Basit attenuation: yakında 0.6g, 50km'de 0.2g
    pga = max(0.1, 0.7 - 0.01 * dist_km)
    return min(pga, 0.8)


def generate_pga_geotiff_proxy(city: str, output_path: Path,
                                 resolution: int = 1000,
                                 smooth_sigma: float = 8.0) -> Path:
    """
    AFAD verisi alınamadığında proxy PGA raster üret.
    Bu **yedek** çözümdür; resmi araştırma için AFAD'dan resmi veri alınmalıdır.

    Args:
        city: 'elazig' veya 'istanbul'
        output_path: çıktı GeoTIFF yolu
        resolution: raster boyutu (resolution x resolution)
        smooth_sigma: Gaussian blur sigma (daha doğal görünüm için)
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter
    except ImportError:
        print("⚠️ numpy + Pillow gerekli")
        return output_path

    bbox = CITY_BBOXES[city]["bbox"]
    minlon, minlat, maxlon, maxlat = bbox

    # Grid oluştur
    lons = np.linspace(minlon, maxlon, resolution)
    lats = np.linspace(maxlat, minlat, resolution)

    pga_grid = np.zeros((resolution, resolution), dtype=np.float32)

    print(f"  → PGA grid hesaplanıyor ({resolution}×{resolution})...")
    # Vektörize: meshgrid ile daha hızlı + smooth
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    flat_lats = lat_mesh.flatten()
    flat_lons = lon_mesh.flatten()

    # Çok kalabalık olduğu için batch hesaplama yerine inner-loop;
    # ama gaussian blur ile son halini yumuşatıyoruz.
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            pga_grid[i, j] = estimate_pga_from_distance(lat, lon, city)

    # Normalize → uint8
    pga_uint8 = (pga_grid * 255 / 0.8).clip(0, 255).astype("uint8")

    # GAUSSIAN BLUR ile yumuşat (merdiven görünümünü gider)
    img = Image.fromarray(pga_uint8)
    if smooth_sigma > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=smooth_sigma))
        print(f"  → Gaussian blur uygulandı (sigma={smooth_sigma})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    # Meta-veri kaydet
    meta = {
        "source": "proxy_estimate",
        "bbox": list(bbox),
        "resolution": resolution,
        "note": "Bu PGA haritası fay-mesafesi tabanlı bir proxy'dir. "
                "Resmi AFAD verisi için https://tdth.afad.gov.tr/ kullanın.",
    }
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return output_path


def try_afad_api(city: str, output_path: Path) -> bool:
    """AFAD resmi API'sini dene."""
    bbox = CITY_BBOXES[city]["bbox"]

    try:
        # Bu endpoint sıklıkla değişiyor — fallback amaçlı
        response = requests.get(
            AFAD_RESOURCES["pga_475yr_url"],
            params={
                "minLon": bbox[0],
                "minLat": bbox[1],
                "maxLon": bbox[2],
                "maxLat": bbox[3],
            },
            timeout=60,
        )

        if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("image"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
    except requests.exceptions.RequestException as e:
        print(f"  ⚠️ AFAD API erişilemiyor: {e}")
    return False


def download_seismic(city: str, output_root: Optional[Path] = None) -> Path:
    """Şehir için sismik veriyi indir veya proxy üret."""
    if output_root is None:
        output_root = Path(DATA_DIRS["raw"])

    out_dir = output_root / city / "seismic"
    out_dir.mkdir(parents=True, exist_ok=True)

    pga_path = out_dir / "pga_475yr.png"

    print(f"\n🌋 Sismik veri: {city.upper()}")
    print(f"   Hedef: {pga_path}")

    if pga_path.exists():
        print("   ⏩ Zaten var")
        return pga_path

    # 1) Resmi API
    print("   → AFAD resmi API deneniyor...")
    if try_afad_api(city, pga_path):
        print("   ✅ AFAD'dan başarıyla indirildi")
        return pga_path

    # 2) Proxy
    print("   → Proxy PGA grid üretiliyor (fay-mesafesi)...")
    print("   ⚠️  Bu proxy SCI dergisinde yetmez!")
    print("   📌 Resmi veri için:")
    print("      1. https://tdth.afad.gov.tr/ adresinde haritayı görüntüleyin")
    print("      2. Akademik talep için: bilgi@afad.gov.tr")
    print(f"      3. Resmi raster'i {pga_path} konumuna koyun")

    generate_pga_geotiff_proxy(city, pga_path)
    print(f"   ✅ Proxy oluşturuldu: {pga_path}")
    return pga_path


def main():
    parser = argparse.ArgumentParser(description="AFAD Sismik İndirici")
    parser.add_argument("--city", choices=["elazig", "istanbul", "both"], required=True)
    parser.add_argument("--output", type=str, default=DATA_DIRS["raw"])
    args = parser.parse_args()

    cities = ["elazig", "istanbul"] if args.city == "both" else [args.city]
    for city in cities:
        download_seismic(city, Path(args.output))

    print("\n🎉 Tamamlandı")


if __name__ == "__main__":
    main()
