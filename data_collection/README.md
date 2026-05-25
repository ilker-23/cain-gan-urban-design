# 🌍 Veri Toplama Pipeline

Elazığ + İstanbul için CAIN-GAN eğitim verisinin **end-to-end otomatik üretimi**.

---

## 🚀 Hızlı Başlangıç

### Tek komutla tam pipeline:

```bash
# Ek bağımlılıklar
pip install -r data_collection/requirements.txt

# Tam pipeline (yaklaşık 1-2 saat)
python -m data_collection.build_dataset --city both
```

Bu komut şunları yapar:
1. ✅ OSM'den bina/yol/su/vejetasyon indirir
2. ✅ Microsoft Building Footprints (Türkiye)
3. ✅ İBB Açık Veri (sadece İstanbul)
4. ✅ AFAD sismik veri (proxy / resmi)
5. ✅ Copernicus DEM (30m)
6. ✅ 256×256 patch'lere böler
7. ✅ `data/{elazig,istanbul}/{train,val,test}` dizinine kaydeder

---

## 📦 Modüller

| Modül | İşlev |
|-------|-------|
| `osm_downloader.py` | Overpass API ile OSM verisi |
| `microsoft_buildings.py` | Microsoft Global Building Footprints |
| `ibb_downloader.py` | İBB Açık Veri Portalı (İstanbul) |
| `afad_seismic.py` | AFAD sismik (PGA) — proxy fallback'li |
| `copernicus_dem.py` | Copernicus DEM 30m (AWS Open Data) |
| `rasterize.py` | Vector → Raster (256×256) |
| `patch_extractor.py` | Sliding window ile patch üretimi |
| `build_dataset.py` | End-to-end orchestration |
| `config.py` | Şehir BBox, ilçe listesi, encoding |

---

## 🎯 Tek Tek Çalıştırma

### 1. OSM indirme

```bash
# Tüm şehir (BBox tabanlı, hızlı)
python -m data_collection.osm_downloader --city elazig
python -m data_collection.osm_downloader --city istanbul

# İlçe-bazlı (yavaş ama hassas)
python -m data_collection.osm_downloader --city istanbul --all_districts
```

### 2. Microsoft Building Footprints

```bash
python -m data_collection.microsoft_buildings --city both
```

> 💡 API erişilemezse manuel indirin:
> https://github.com/microsoft/GlobalMLBuildingFootprints

### 3. AFAD Sismik Veri

```bash
python -m data_collection.afad_seismic --city both
```

> ⚠️ Resmi AFAD verisi alınamadığında proxy üretilir (fay-mesafesi tabanlı).
> SCI dergi makalesi için **resmi AFAD raster'i şart**:
> https://tdth.afad.gov.tr/ → akademik kullanım talebi

### 4. Copernicus DEM

```bash
python -m data_collection.copernicus_dem --city both --merge
```

AWS Open Data S3 bucket'ından ücretsiz indirilir (kayıt gerekmez).

### 5. İBB Açık Veri (sadece İstanbul)

```bash
python -m data_collection.ibb_downloader --layer all
```

### 6. Patch Üretimi

```bash
python -m data_collection.patch_extractor --city both
```

---

## 📂 Çıktı Dizin Yapısı

```
data_collection/
├── raw/
│   ├── elazig/
│   │   ├── buildings.geojson        ← OSM binalar
│   │   ├── roads.geojson
│   │   ├── water.geojson
│   │   ├── vegetation.geojson
│   │   ├── landuse.geojson
│   │   ├── microsoft_buildings/     ← Microsoft GBF
│   │   ├── seismic/
│   │   │   └── pga_475yr.png        ← AFAD PGA raster
│   │   └── dem/
│   │       ├── Copernicus_DSM_*.tif ← DEM tile'lar
│   │       └── dem_merged.png       ← Birleştirilmiş DEM
│   └── istanbul/
│       └── (aynı yapı + ibb/ klasörü)
│
data/                                ← CAIN-GAN training input!
├── elazig/
│   ├── train/
│   │   ├── site_context/            ← roads/veg/water
│   │   ├── planning_guidance/       ← landuse one-hot
│   │   ├── neighboring_footprints/
│   │   ├── mask/                    ← 0=tasarım, 255=kontekst
│   │   ├── seismic/
│   │   ├── dem/
│   │   ├── footprint_target/
│   │   └── height_target/
│   ├── val/
│   └── test/
└── istanbul/
    └── (aynı yapı)
```

---

## ⚙️ Konfigürasyon

`data_collection/config.py` dosyasında:

- **CITY_BBOXES**: Şehir sınırları (WGS84)
- **ISTANBUL_DISTRICTS**: İstanbul ilçe listesi (5 ilçe önerildi)
- **ELAZIG_DISTRICTS**: Elazığ ilçe listesi
- **PATCH_CONFIG**: Patch boyutu (256m), stride (192m), min bina sayısı (3)
- **LAND_USE_ENCODING**: Türk imar kategorileri (RGB)
- **SITE_CONTEXT_ENCODING**: Yol/su/vejetasyon değerleri

---

## 📊 Beklenen Veri Hacmi

Varsayılan konfigürasyonla:

| Şehir | Tahmini Patch Sayısı | Toplam Boyut |
|-------|---------------------|--------------|
| Elazığ | 1.500 – 3.000 | ~150 MB |
| İstanbul (5 ilçe) | 8.000 – 15.000 | ~750 MB |
| **Toplam** | **10K – 18K** | **~900 MB** |

Hesaplama:
- Elazığ ~25 km × 22 km = 550 km² → 256m patches ≈ 8400 aday
- Filtreden geçen (3+ bina): ~%30 = 2500 patch
- İstanbul daha yoğun → %50 geçer

---

## 🔍 Sorun Giderme

### `requests.exceptions.ConnectionError` (Overpass API)
Overpass yoğun olabilir. Birkaç dakika bekleyip tekrar deneyin.
Script otomatik olarak 3 farklı endpoint dener.

### `requests.exceptions.HTTPError: 403` (Microsoft Buildings)
S3 erişimi kısıtlı olabilir. Manuel indirin:
```bash
wget https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv
```

### `OSError: [Errno 28] No space left on device`
Raw veri 2-5 GB yer kaplayabilir. Yeterli disk alanı kontrolü yapın.

### Patch'ler boş geliyor
`PATCH_CONFIG["min_buildings_per_patch"]` değerini düşürün (örn: 1).

---

## 📜 Veri Lisansları ve Atıflar

⚠️ **SCI dergi makalesinde atıf yapılması ZORUNLUDUR:**

```
Building footprints: © OpenStreetMap contributors (ODbL)
Microsoft Building Footprints: © Microsoft (CC BY 4.0)
İBB Açık Veri: © Istanbul Metropolitan Municipality
AFAD: © Disaster and Emergency Management Authority, Turkey
Copernicus DEM: © European Space Agency (CC BY 4.0)
```

---

## 🎓 Akademik Kullanım

1. **Etik Kurul Onayı** alın (özellikle Belediye verisi için)
2. **Veri paylaşım protokolü** imzalayın
3. **Anonimleştirme**: parsel sahibi bilgileri **kullanılmaz**

Detay: ana `TURKISH_CITIES_GUIDE.md`
