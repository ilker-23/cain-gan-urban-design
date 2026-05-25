"""
Veri toplama konfigürasyonu.
Şehir sınırları, ilçe listeleri ve dizin yapıları.
"""

# Şehir bounding box'ları (minlon, minlat, maxlon, maxlat) — WGS84
CITY_BBOXES = {
    "elazig": {
        "bbox": (39.05, 38.58, 39.40, 38.78),
        "center": (39.22, 38.68),
        "name_tr": "Elazığ",
        "admin_level": "4",
        "epsg_local": 32637,  # UTM Zone 37N
    },
    "istanbul": {
        "bbox": (28.50, 40.80, 29.45, 41.30),  # Avrupa + Asya yakası
        "center": (28.97, 41.01),
        "name_tr": "İstanbul",
        "admin_level": "4",
        "epsg_local": 32635,  # UTM Zone 35N
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
PATCH_CONFIG = {
    "size_meters": 256,        # Her patch ~256m × 256m kapsar
    "image_size": 256,         # 256×256 piksel → 1m/piksel çözünürlük
    "stride_meters": 192,      # %25 overlap (data augmentation)
    "buffer_meters": 100,      # Komşu bina çekme buffer'ı
    "min_buildings_per_patch": 3,  # Bu sayıdan az binası olan patch'i atla
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
