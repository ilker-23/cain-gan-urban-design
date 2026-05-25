"""
CAIN-GAN Veri Toplama Pipeline
================================
Elazığ + İstanbul için gerçek veri toplama ve preprocessing.

Modüller:
- osm_downloader:        OpenStreetMap üzerinden bina/yol/vejetasyon
- microsoft_buildings:   Microsoft Global Building Footprints
- ibb_downloader:        İBB Açık Veri Portalı
- afad_seismic:          AFAD Türkiye Deprem Tehlike Haritası
- copernicus_dem:        Copernicus DEM (30m)
- rasterize:             Vector → Raster dönüşümü
- patch_extractor:       256x256 patch'lere bölme
- build_dataset:         End-to-end orchestration
"""

from .config import (
    CITY_BBOXES,
    ISTANBUL_DISTRICTS,
    ELAZIG_DISTRICTS,
    DATA_DIRS,
)

__all__ = [
    "CITY_BBOXES",
    "ISTANBUL_DISTRICTS",
    "ELAZIG_DISTRICTS",
    "DATA_DIRS",
]
