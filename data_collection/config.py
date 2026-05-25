"""
Veri toplama konfigürasyonu.
Şehir sınırları, ilçe listeleri ve dizin yapıları.
"""

# Şehir bounding box'ları (minlon, minlat, maxlon, maxlat) — WGS84
#
# ⚠️ ÖNEMLİ: BBox'lar **kentsel çekirdek**e daraltıldı.
# Rural alanları dahil etmek dataset'i sulandırır (sparse patches).
# Geniş BBox isteyenler için 'bbox_extended' alanı eklendi.
CITY_BBOXES = {
    "elazig": {
        # KENT MERKEZİ — yoğun yerleşim
        # ~12 km × 8 km = 96 km² urban core
        "bbox": (39.17, 38.65, 39.29, 38.72),
        # Tüm il (referans için, kullanılmıyor)
        "bbox_extended": (39.05, 38.58, 39.40, 38.78),
        "center": (39.22, 38.685),
        "name_tr": "Elazığ",
        "admin_level": "4",
        "epsg_local": 32637,
    },
    "istanbul": {
        # Seçili 5 ilçe (Beyoğlu, Kadıköy, Beşiktaş, Üsküdar, Sarıyer)
        # Kuzey-güney ekseni, Boğaz çevresi — morfolojik çeşitlilik
        # ~25 km × 15 km kentsel çekirdek
        "bbox": (28.94, 40.97, 29.10, 41.10),
        # Tüm Avrupa+Asya yakası (kullanılmıyor)
        "bbox_extended": (28.50, 40.80, 29.45, 41.30),
        "center": (29.02, 41.04),
        "name_tr": "İstanbul",
        "admin_level": "4",
        "epsg_local": 32635,
    },
}

# Elazığ ilçeleri (merkez + çevre)
ELAZIG_DISTRICTS = [
    "Merkez",
    "Harput",  # Tarihi alan
]

# İstanbul ilçeleri — morfolojik çeşitlilik için seçilmiş
# Tüm İstanbul = 1.5M+ bina → hesaplama olanaksız, örneklem alıyoruz
ISTANBUL_DISTRICTS = [
    "Beyoğlu",     # Tarihi merkez, Bizans-Osmanlı
    "Kadıköy",     # Modern, yüksek yoğunluk
    "Beşiktaş",    # Karma, yüksek değer
    "Üsküdar",     # Tarihi + modern karışım
    "Sarıyer",     # Düşük yoğunluk, ormanlık
]

# Veri dizin yapısı
DATA_DIRS = {
    "raw": "data_collection/raw",
    "processed": "data_collection/processed",
    "patches": "data",  # CAIN-GAN training dataset hedefi
}

# OSM tag tanımları
OSM_TAGS = {
    "buildings": ["building"],
    "roads": ["highway"],
    "water": ["natural=water", "waterway"],
    "vegetation": [
        "landuse=forest",
        "landuse=grass",
        "leisure=park",
        "natural=wood",
    ],
    "landuse": [
        "landuse=residential",
        "landuse=commercial",
        "landuse=industrial",
        "landuse=retail",
    ],
}

# Patch parametreleri
#
# ⚠️ min_buildings_per_patch SIKILAŞTIRILDI (3 → 15):
# Önceki değer kırsal/seyrek alanları kabul ediyordu → patch'lerde 2-5 bina
# kalıyordu. 15+ bina filtreleyince yoğun kentsel alanlar elde ediliyor.
PATCH_CONFIG = {
    "size_meters": 256,
    "image_size": 256,
    "stride_meters": 128,           # %50 overlap (daha çok patch)
    "buffer_meters": 100,
    "min_buildings_per_patch": 15,  # Sıkı filtre — sadece yoğun alanlar
    "min_inside_buildings": 2,      # Tasarım alanında en az 2 bina olmalı
    "max_water_ratio": 0.5,         # Yarıdan fazlası su → atla
}

# Türk imar planı kategorileri (RGB encoding)
LAND_USE_ENCODING = {
    "konut":           (255, 0, 0),    # Kırmızı — Residential
    "ticaret":         (0, 255, 0),    # Yeşil — Commercial
    "sanayi":          (0, 0, 255),    # Mavi — Industrial
    "karma":           (255, 255, 0),  # Sarı — Mixed
    "yesil_alan":      (0, 255, 255),  # Cyan — Green
    "kentsel_donusum": (255, 0, 255),  # Magenta — Transformation
}

# Site context encoding (grayscale)
SITE_CONTEXT_ENCODING = {
    "background":  0,
    "road":        64,
    "vegetation":  128,
    "water":       192,
}
