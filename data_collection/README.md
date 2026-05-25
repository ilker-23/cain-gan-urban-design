# 🌍 Veri Toplama Pipeline

Elazığ + İstanbul için CAIN-GAN eğitim verisinin **end-to-end otomatik üretimi**.

---

## 🚀 Hızlı Başlangıç

### Tek komutla tam pipeline (ÖNERİLEN — Geofabrik tabanlı):

```bash
# Ek bağımlılıklar (geopandas KRİTİK)
pip install -r data_collection/requirements.txt

# Tam pipeline (~30-45 dakika)
python -m data_collection.build_dataset --city both
```

Bu komut şunları yapar:
1. ✅ **Geofabrik Türkiye OSM** bulk indirme (~600 MB, tek seferlik)
2. ✅ Şehir BBox ile filtreleme (binalar, yollar, su, vejetasyon, landuse)
3. ✅ AFAD sismik veri (proxy / resmi)
4. ✅ Copernicus DEM (30m, AWS Open Data)
5. ✅ İBB Açık Veri (sadece İstanbul, opsiyonel)
6. ✅ 256×256 patch'lere böler
7. ✅ `data/{elazig,istanbul}/{train,val,test}` dizinine kaydeder

---

## ⚡ Neden Geofabrik?

| Kriter | Overpass API | **Geofabrik** ✅ |
|--------|-------------|------------------|
| Rate-limit | Sıkı (406/429) | Yok |
| İstanbul (1.5M+ bina) | ❌ 406 hatası | ✅ Sorunsuz |
| Hız | Saatler | ~10 dakika |
| Güncellik | Anlık | Günlük |
| Bağımlılık | requests | geopandas |
| Güvenilirlik | Düşük | %100 |

Overpass API yoğun saatlerde "Not Acceptable" (HTTP 406) atıyor. Geofabrik aynı OSM verisini tek bir ZIP'te sunuyor.

---

## 📦 Modüller

| Modül | İşlev | Durum |
|-------|-------|-------|
| `geofabrik_downloader.py` | **Geofabrik Türkiye OSM bulk** ⭐ | Birincil |
| `osm_downloader.py` | Overpass API (fallback) | Yedek |
| `microsoft_buildings.py` | Microsoft Global Building Footprints | Opsiyonel |
| `ibb_downloader.py` | İBB Açık Veri Portalı (İstanbul) | Opsiyonel |
| `afad_seismic.py` | AFAD sismik (PGA) — proxy fallback'li | Önerilen |
| `copernicus_dem.py` | Copernicus DEM 30m (AWS Open Data) | Önerilen |
| `rasterize.py` | Vector → Raster (256×256) | Core |
| `patch_extractor.py` | Sliding window ile patch üretimi | Core |
| `build_dataset.py` | End-to-end orchestration | Core |
| `config.py` | Şehir BBox, ilçe listesi, encoding | Core |

---

## 🎯 Tek Tek Çalıştırma

### 1. Geofabrik (ÖNERİLEN)

```bash
# Türkiye extract'ini indir + Elazığ + İstanbul çıkar
python -m data_collection.geofabrik_downloader --extract both

# Sadece bir şehir
python -m data_collection.geofabrik_downloader --extract elazig

# Sadece indir (şehir çıkarmadan)
python -m data_collection.geofabrik_downloader --download_only
```

İndirilen ZIP varsayılan olarak `data_collection/raw/_geofabrik/turkey-latest-free.shp.zip` konumunda saklanır. Aynı komut tekrar çalıştırıldığında indirme atlanır.

### 2. OSM/Overpass (Yedek)

⚠️ **Overpass yoğun saatlerde 406 hatası verir.** Geofabrik başarısız olursa kullanın.

```bash
python -m data_collection.osm_downloader --city elazig
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
