# 🇹🇷 Türk Şehirleri Veri Toplama Rehberi

> **Elazığ + İstanbul** için CAIN-GAN veri hazırlama dokümantasyonu.
> Bu rehber, SCI dergi standartlarında veri toplama ve preprocessing içindir.

---

## 📋 İçindekiler

1. [Genel Strateji](#strateji)
2. [Veri Kaynakları](#kaynaklar)
3. [Elazığ Veri Toplama](#elazig)
4. [İstanbul Veri Toplama](#istanbul)
5. [Sismik ve Topografik Veriler](#seismic)
6. [Preprocessing Pipeline](#preprocess)
7. [Kalite Kontrol](#kalite)
8. [Etik ve Lisans](#etik)

---

## 🎯 Genel Strateji <a id="strateji"></a>

### Hedef Veri Hacmi

| Şehir | Eğitim örneği | Mahalle/Ada sayısı | Bina sayısı |
|-------|---------------|-------------------|--------------|
| Elazığ | 3.000 – 5.000 | ~600 ada | ~50.000 |
| İstanbul | 10.000 – 20.000 | ~3.000 ada | ~500.000+ |
| **Toplam** | **13.000 – 25.000** | **~3.600 ada** | **~550.000+** |

> 📌 **Önemli:** NYC çalışmasında 37.590 Census Block kullanılmıştı.
> Bizim "ada" seviyemiz daha incedir → örnek başına çeşitlilik daha yüksek.

### Veri Birimi: **Ada** (Block)

Türkiye'de kullanılacak doğal birim:
- **Çok büyük:** Mahalle (~200 bina) → reddet
- **✅ Doğru:** Ada (~10-30 bina) → kullan
- **Çok küçük:** Parsel (~1-3 bina) → reddet

---

## 🗂️ Veri Kaynakları <a id="kaynaklar"></a>

### Ücretsiz / Açık Kaynaklar

| Kaynak | İçerik | Format | URL |
|--------|--------|--------|-----|
| **OpenStreetMap** | Bina footprint, yollar, alanlar | GeoJSON, Shapefile | https://www.openstreetmap.org |
| **Microsoft Building Footprints** | Otomatik üretilmiş binalar | GeoJSON | https://github.com/microsoft/GlobalMLBuildingFootprints |
| **Google Open Buildings** | Bina yükseklikleri | CSV, GeoTIFF | https://sites.research.google/open-buildings/ |
| **Copernicus DEM** | 30m yükseklik haritası | GeoTIFF | https://spacedata.copernicus.eu |
| **SRTM DEM** | 30m yükseklik | GeoTIFF | https://earthexplorer.usgs.gov |
| **AFAD Türkiye Deprem Tehlike Haritası** | PGA değerleri | PDF/GeoTIFF | https://tdth.afad.gov.tr/ |
| **Sentinel-2** | 10m çözünürlüklü uydu | GeoTIFF | https://scihub.copernicus.eu/ |

### Türkiye Resmi Kaynakları

| Kurum | Veri | Erişim |
|-------|------|--------|
| **İBB Açık Veri Portalı** | İstanbul tüm CBS verileri | ✅ Açık — https://data.ibb.gov.tr/ |
| **Elazığ Belediyesi** | İmar planı, ada/parsel | 🟡 Talep gerekli |
| **Tapu ve Kadastro GM** | Parsel verileri | 🟡 Belediye aracılığıyla |
| **MTA (Maden Tetkik Arama)** | Jeolojik haritalar | ✅ Açık — http://yerbilimleri.mta.gov.tr |
| **AFAD** | Deprem fay hatları, PGA | ✅ Açık |
| **MGM** | Meteorolojik veriler | ✅ Açık |
| **TUCBS** (Ulusal CBS) | Standart vektörler | 🟡 Akademik kullanım için talep |

---

## 🏔️ Elazığ Veri Toplama <a id="elazig"></a>

### Bağlam
- **Konum:** 38.68°N, 39.22°E
- **Nüfus:** ~600.000 (2024)
- **Yükseklik:** 1067m (plato şehir)
- **Deprem riski:** 1. derece (Doğu Anadolu Fay Zonu)
- **Son büyük deprem:** 24 Ocak 2020, Mw 6.8

### Adım 1: OSM Verisi İndir

```python
# Overpass API ile Elazığ sınırları içinde binalar
import requests

overpass_url = "http://overpass-api.de/api/interpreter"
query = """
[out:json][timeout:300];
area["name"="Elazığ"]["admin_level"="4"]->.searchArea;
(
  way["building"](area.searchArea);
  way["highway"](area.searchArea);
  way["natural"="water"](area.searchArea);
  way["landuse"="forest"](area.searchArea);
  way["leisure"="park"](area.searchArea);
);
out geom;
"""
response = requests.post(overpass_url, data={"data": query})
elazig_osm = response.json()
```

veya **QGIS** üzerinden:
```
QGIS → Vector → QuickOSM → Türkiye → Elazığ → building, highway, natural, landuse
```

### Adım 2: Belediye Verisi

```
Adım 1: Elazığ Belediyesi'ne resmi yazı ile başvur
        (Akademik kullanım, etik kurul onayı ile)
Adım 2: İmar planı PDF/CAD dosyalarını al
Adım 3: QGIS'te georeferans yap
Adım 4: Polygon olarak sayısallaştır (manuel, ~2-3 hafta iş)
```

> 📞 İletişim: imar@elazig.bel.tr (genel adres, doğrulanmalı)

### Adım 3: Bina Yükseklikleri

İki yöntem:

**A) Google Open Buildings (otomatik, hızlı)**
```python
# Open Buildings CSV indir
url = "https://sites.research.google/open-buildings/download/"
# Bölge seç: V3 → Turkey → Elazig
# CSV'de "confidence", "area", "height" kolonları
```

**B) Sentinel-1 SAR + Sentinel-2 (yüksek doğruluk)**
```python
# Google Earth Engine ile
import ee
ee.Initialize()
elazig = ee.Geometry.Rectangle([39.1, 38.6, 39.35, 38.75])
sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(elazig)
# Yükseklik tahmini için stereo görüntü işleme gerekir
```

### Adım 4: Sismik Veri

```
1. https://tdth.afad.gov.tr/ adresine git
2. Elazığ koordinatlarını gir (38.68, 39.22)
3. PGA (Peak Ground Acceleration) haritası indir
4. GeoTIFF olarak kaydet
5. 256x256 patch'lere böl
```

---

## 🌉 İstanbul Veri Toplama <a id="istanbul"></a>

### Bağlam
- **Konum:** 41.01°N, 28.97°E
- **Nüfus:** ~16 milyon
- **Topografi:** Karma (kıyı + tepe)
- **Deprem riski:** Kuzey Anadolu Fayı (KAF) — beklenen büyük deprem
- **Karmaşıklık:** Bizans + Osmanlı + modern

### Adım 1: İBB Açık Veri Portalı

İstanbul için **en güçlü kaynak** İBB'dir:

```python
import requests
import pandas as pd

# Bina verisi
buildings_url = "https://data.ibb.gov.tr/dataset/binalar"

# İmar durumu
zoning_url = "https://data.ibb.gov.tr/dataset/imar-plani"

# Yeşil alanlar
green_url = "https://data.ibb.gov.tr/dataset/yesil-alanlar"

# Yol ağı
roads_url = "https://data.ibb.gov.tr/dataset/yol-agi"
```

Manuel indirme:
```
1. https://data.ibb.gov.tr/ → Arama: "bina"
2. Shapefile/GeoJSON formatında indir
3. İlçe bazlı filtre uygulanabilir (örn: Kadıköy, Beyoğlu, Üsküdar)
```

### Adım 2: İlçe Seçimi (Hesaplama Verimliliği)

Tüm İstanbul = ~1.500.000 bina → çok büyük.

**Önerilen örneklem (5 farklı ilçe, morfolojik çeşitlilik için):**

| İlçe | Karakter | Örnek bina |
|------|----------|------------|
| **Beyoğlu** | Tarihi merkez, Bizans-Osmanlı | ~30.000 |
| **Kadıköy** | Modern, yüksek yoğunluk | ~80.000 |
| **Beşiktaş** | Karma, yüksek değer | ~40.000 |
| **Üsküdar** | Tarihi + modern karışım | ~70.000 |
| **Sarıyer** | Düşük yoğunluk, ormanlık | ~40.000 |
| **Toplam** | | **~260.000** |

Bu, 10-20K eğitim örneği üretir (her bina ~50 ada başına).

### Adım 3: Sismik Bölgeleme

İstanbul KAF'a yakın → tüm şehir 1. derece sismik bölge.
Ancak yerel zemin koşulları çeşitli:

```python
# İBB Mikro-bölgeleme haritası
microzoning_url = "https://data.ibb.gov.tr/dataset/zemin-etudleri"
# 5 sınıf: ZA, ZB, ZC, ZD, ZE (en kötü)
```

---

## 🌍 Sismik ve Topografik Veriler <a id="seismic"></a>

### Sismik Risk Kanalı

**Format:** Grayscale PNG (256×256, [0,255])
**Anlamı:** Piksel değeri normalize edilmiş PGA
- 0   → çok düşük risk
- 255 → çok yüksek risk (>0.5g)

```python
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from PIL import Image

def crop_seismic_patch(geotiff_path, bbox, output_path):
    """AFAD PGA GeoTIFF'inden 256x256 patch çıkar."""
    with rasterio.open(geotiff_path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        data = src.read(1, window=window)

        # PGA değerleri 0-1g aralığında → [0,255]'e normalize et
        normalized = np.clip(data / 1.0 * 255, 0, 255).astype(np.uint8)

        # 256x256'ya resize
        img = Image.fromarray(normalized).resize((256, 256), Image.LANCZOS)
        img.save(output_path)
```

### Topografya Kanalı (DEM)

**Kaynak:** Copernicus DEM (30m çözünürlük)

```python
# Earth Engine ile indirme
import ee
ee.Initialize()

dem = ee.Image('COPERNICUS/DEM/GLO30').select('DEM')

# Elazığ için patch
elazig_bbox = ee.Geometry.Rectangle([39.1, 38.6, 39.35, 38.75])
clipped = dem.clip(elazig_bbox)

# Export to Drive
task = ee.batch.Export.image.toDrive(
    image=clipped,
    description='elazig_dem',
    scale=30,
    region=elazig_bbox,
    fileFormat='GeoTIFF',
)
task.start()
```

**Preprocessing:**
```python
# Yüksekliği normalize et
dem_array = ...  # GeoTIFF'ten yüklenmiş
elevation_min = dem_array.min()
elevation_max = dem_array.max()
normalized = (dem_array - elevation_min) / (elevation_max - elevation_min)
img = (normalized * 255).astype(np.uint8)
```

---

## 🔧 Preprocessing Pipeline <a id="preprocess"></a>

### Adım 1: Vektör → Raster Dönüşümü

```python
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

def rasterize_buildings(geodf, output_size=(256, 256), bounds=None):
    """GeoDataFrame'i 256x256 binary mask'a dönüştür."""
    if bounds is None:
        bounds = geodf.total_bounds  # (minx, miny, maxx, maxy)

    transform = rasterio.transform.from_bounds(*bounds, *output_size)
    shapes = [(geom, 1) for geom in geodf.geometry]
    raster = rasterize(shapes, out_shape=output_size, transform=transform, dtype=np.uint8)
    return raster * 255  # binary → [0, 255]
```

### Adım 2: Ada Bazlı Patch Çıkarımı

```python
def extract_block_patches(blocks_gdf, buildings_gdf, output_dir):
    """Her ada için tüm 256x256 patch'leri üret."""
    for idx, block in blocks_gdf.iterrows():
        # Ada bounding box (256m * 256m varsayım)
        bbox = block.geometry.bounds

        # Komşu binaları al
        nearby_buildings = buildings_gdf[buildings_gdf.intersects(block.geometry.buffer(100))]

        # Site context, planning, footprints, mask oluştur
        # ... (çoklu kanal oluşturma)

        sample_id = f"block_{idx:06d}"
        save_all_channels(sample_id, output_dir)
```

### Adım 3: Train/Val/Test Split

```python
import random

def stratified_split(sample_ids, train_ratio=0.7, val_ratio=0.15):
    """Stratified split (şehir başına eşit oran)."""
    random.seed(42)
    random.shuffle(sample_ids)

    n = len(sample_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": sample_ids[:n_train],
        "val": sample_ids[n_train:n_train + n_val],
        "test": sample_ids[n_train + n_val:],
    }
```

### Adım 4: Final Dizin Yapısı

```
data/
├── elazig/
│   ├── train/
│   │   ├── site_context/         ← roads/veg/water (256x256 grayscale)
│   │   ├── planning_guidance/    ← imar (256x256 RGB, one-hot)
│   │   ├── neighboring_footprints/
│   │   ├── mask/                 ← 0=tasarım, 255=kontekst
│   │   ├── seismic/              ← AFAD PGA
│   │   ├── dem/                  ← Copernicus
│   │   ├── footprint_target/
│   │   └── height_target/
│   ├── val/
│   └── test/
└── istanbul/
    ├── train/
    └── ...
```

---

## ✅ Kalite Kontrol <a id="kalite"></a>

### Veri Kalite Checklist

- [ ] Tüm görüntüler **256×256** boyutunda
- [ ] Eşleşen filename'ler tüm dizinlerde mevcut
- [ ] Site context değerleri [0-3] aralığında (4 sınıf)
- [ ] Planning guidance RGB ve one-hot encoded
- [ ] Mask binary (0 veya 255)
- [ ] Sismik kanal **normalize edilmiş** ([0-255])
- [ ] DEM **normalize edilmiş** ([0-255])
- [ ] Train/val/test örtüşmesi **YOK**
- [ ] Şehir başına en az 3.000 eğitim örneği
- [ ] Her örnekte en az 5 komşu bina
- [ ] Etik onay alındı (akademik kurul)

### Veri Doğrulama Scripti

```python
from pathlib import Path
from PIL import Image
import numpy as np

def validate_dataset(data_root, city):
    """Veri seti bütünlük kontrolü."""
    base = Path(data_root) / city / "train"
    required = ["site_context", "planning_guidance", "mask",
                "neighboring_footprints", "footprint_target"]

    sample_ids = [f.stem for f in (base / required[0]).glob("*.png")]

    issues = []
    for sid in sample_ids:
        for sub in required:
            path = base / sub / f"{sid}.png"
            if not path.exists():
                issues.append(f"Eksik: {sub}/{sid}")
                continue

            img = Image.open(path)
            if img.size != (256, 256):
                issues.append(f"Yanlış boyut: {sub}/{sid} = {img.size}")

    print(f"📊 {city}: {len(sample_ids)} örnek, {len(issues)} sorun")
    return issues
```

---

## ⚖️ Etik ve Lisans <a id="etik"></a>

### Akademik Kullanım

1. **Etik kurul onayı:** Üniversitenizden alın (özellikle Belediye verisi için)
2. **Veri paylaşım protokolü:** Belediyelerle yazılı anlaşma
3. **Anonimleştirme:** Parsel sahibi bilgileri **asla** kullanılmamalı

### Veri Lisansları

| Kaynak | Lisans | Atıf |
|--------|--------|------|
| OpenStreetMap | ODbL | Zorunlu |
| Copernicus DEM | CC BY 4.0 | Zorunlu |
| İBB Açık Veri | Çeşitli | İBB belirtir |
| AFAD | Kamu | Zorunlu |
| Google Open Buildings | CC BY 4.0 | Zorunlu |

### Önerilen Atıf Şablonu (Makale İçin)

```
"Building footprints were obtained from OpenStreetMap contributors (OSM 2024)
and supplemented with Microsoft Global Building Footprints (Microsoft 2023).
Seismic hazard data was acquired from the Turkish Disaster and Emergency
Management Authority (AFAD 2024). Topographic data was derived from the
Copernicus DEM at 30m resolution (ESA 2024). Istanbul-specific zoning data
was provided by Istanbul Metropolitan Municipality Open Data Portal
(IBB 2024). Elazığ municipal data was provided under an academic research
agreement with Elazığ Municipality."
```

---

## 🎓 Önerilen İş Sırası

```
Hafta 1-2:  OSM + Microsoft + Google Open Buildings indirme (her iki şehir)
Hafta 3-4:  İBB Açık Veri Portalı'ndan İstanbul detaylı verileri
Hafta 5-6:  Elazığ Belediyesi başvurusu + manuel sayısallaştırma
Hafta 7:    AFAD sismik veriler + DEM çıkarma
Hafta 8-9:  Preprocessing pipeline çalıştırma
Hafta 10:   Kalite kontrol + train/val/test split
Hafta 11:   Pilot eğitim (synthetic ile validasyon)
Hafta 12:   Gerçek veri ile tam eğitim başlat
```

---

## 📞 İletişim ve Destek

- **OSM Türkiye:** https://www.openstreetmap.org/user/turkiye
- **İBB Açık Veri:** acikveri@ibb.istanbul
- **AFAD Veri:** https://www.afad.gov.tr/iletisim

---

**Son güncelleme:** 2026-05-25
**Versiyon:** 1.0 (Elazığ + İstanbul)
