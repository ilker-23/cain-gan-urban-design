# 🏙️ Seismic-CAIN-GAN: Multi-City Urban Site Planning (Elazığ + İstanbul)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/cain-gan-urban-design/blob/main/CAIN_GAN_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Sismik farkındalıklı, çok-şehirli kentsel site planlama için GAN tabanlı PyTorch implementasyonu**
> — *Automation in Construction* dergisindeki **CAIN-GAN** modeli temel alınarak Türkiye bağlamına genişletildi.

Bu repository, **Elazığ** ve **İstanbul** için uydu görüntülerinden bina yer planı ve yükseklik üretimi yapan, **sismik risk + topografya** farkındalıklı genişletilmiş bir CAIN-GAN uygulamasıdır. SCI Q1/Q2 dergi yayını ve doktora/yüksek lisans tezi için tasarlanmıştır.

---

## 🚀 Colab Notebook'ları

| Notebook | Amaç | Süre |
|----------|------|------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/cain-gan-urban-design/blob/main/Data_Collection_Colab.ipynb) **Data Collection** | Elazığ + İstanbul gerçek veri toplama | ~30-60 dk |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/cain-gan-urban-design/blob/main/CAIN_GAN_Colab.ipynb) **Training** | Seismic-CAIN-GAN eğitimi | ~2-6 saat |

**Önerilen sıra:**
1. **Data_Collection_Colab** → gerçek veriyi topla (OSM, Microsoft, AFAD, Copernicus, İBB)
2. **CAIN_GAN_Colab** → toplanan veri üzerinde eğitim başlat

---

## 🎯 Bu Proje Ne Sunuyor?

### Orijinal CAIN-GAN'a Göre Yenilikler

| Özellik | Orijinal CAIN-GAN (NYC) | Bu Proje (TR) |
|---------|-------------------------|---------------|
| **Şehir sayısı** | 1 (NYC) | 2 (Elazığ + İstanbul) |
| **Ölçek çeşitliliği** | Tek (megakent) | İki (orta + mega) |
| **Sismik farkındalık** | ❌ Yok | ✅ Var (AFAD PGA kanalı) |
| **Topografya** | ❌ Düz arazi varsayımı | ✅ DEM kanalı (Copernicus) |
| **Şehir-spesifik norm** | ❌ Yok | ✅ CityConditionalNorm |
| **Loss fonksiyonu** | L1 + Adv | L1 + Adv + Seismic + Terrain |
| **Conditional kanal** | 5 | 10+ |

### Akademik Katkılar

1. **Seismic-CAIN-GAN:** Sismik attention gate + sismik kayıp terimi
2. **Multi-scale validation:** Küçük şehir (Elazığ) ↔ megakent (İstanbul)
3. **Türkiye boşluğu:** GAN-tabanlı urban design literatüründe Türkiye case study'si yok
4. **Post-disaster context:** 2020 Sivrice ve 2023 Kahramanmaraş depremleri sonrası yeniden yapılanma

---

## 📂 Proje Yapısı

```
cain-gan-urban-design/
│
├── 🔧 Core (Orijinal CAIN-GAN)
│   ├── cain_dataset.py              # NYC-style dataset (referans)
│   ├── cain_architecture.py         # Generator, Discriminator, Attention
│   └── cain_training.py             # Tek şehir eğitim
│
├── 🇹🇷 Türk Şehirleri Uzantısı
│   ├── multi_city_dataset.py        # Elazığ + İstanbul dataset
│   ├── seismic_extension.py         # Sismik + topografi modülleri
│   └── multi_city_training.py       # Çok-şehirli sismik eğitim
│
├── 📚 Genel Modüller
│   ├── dataset.py                   # Generic Pix2Pix dataset
│   ├── augmentation.py              # Augmentation stratejileri
│   └── config_loader.py             # YAML config yöneticisi
│
├── ⚙️ Yapılandırma
│   ├── config.yaml                  # Eğitim parametreleri
│   └── requirements.txt             # Bağımlılıklar
│
├── 📓 Notebook
│   └── CAIN_GAN_Colab.ipynb         # Colab eğitim notebook'u
│
└── 📖 Dokümantasyon
    ├── README.md                       # Bu dosya
    ├── TURKISH_CITIES_GUIDE.md         # ⭐ Türk veri toplama rehberi
    ├── CAIN_GAN_IMPLEMENTATION.md      # Teknik mimari
    ├── QUICKSTART.md                   # Hızlı başlangıç
    └── PROJECT_SUMMARY.md              # Proje özeti
```

---

## 🏗️ Mimari

### Genişletilmiş Conditional Input (10+ kanal)

```
[0]      Site context        ← yollar, vejetasyon, su
[1-3]    Planning guidance   ← Türk imar kategorileri (RGB)
[4]      Neighboring footprints
[5]      Mask                ← 0=tasarım alanı, 1=kontekst
[6]      Seismic risk        ← AFAD PGA (NEW)
[7]      DEM                 ← Copernicus yükseklik (NEW)
[8-9]    City embedding      ← one-hot (Elazığ/İstanbul) (NEW)
```

### Seismic Attention Gate

```
Encoder → DualPathBottleneck → SeismicAttentionGate → CityNorm → Decoder
                                       ↑                ↑
                                  seismic_map      city_index
```

### Genişletilmiş Loss

```
L_total = λ_rec · L1 + λ_adv · L_adv
        + λ_seismic · L_seismic   (yüksek riskli bölgede yüksek bina cezası)
        + λ_terrain · L_terrain   (eğimli bölgelerde footprint cezası)
```

---

## 💻 Yerel Kurulum

```bash
# Repo'yu klonla
git clone https://github.com/ilker-23/cain-gan-urban-design.git
cd cain-gan-urban-design

# Sanal ortam (önerilen)
python -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## 📊 Veri Hazırlama

Detaylı rehber için: **[TURKISH_CITIES_GUIDE.md](TURKISH_CITIES_GUIDE.md)**

### Veri Kaynakları Özeti

| Şehir | Ana Kaynak | İmar | Sismik | DEM |
|-------|-----------|------|--------|-----|
| **Elazığ** | OSM + Microsoft Buildings | Elazığ Belediyesi | AFAD | Copernicus |
| **İstanbul** | İBB Açık Veri Portalı | İBB | AFAD + İBB Mikro-bölgeleme | Copernicus |

### Dizin Yapısı

```
data/
├── elazig/
│   ├── train/
│   │   ├── site_context/         # roads, vegetation, water
│   │   ├── planning_guidance/    # imar planı kategorileri
│   │   ├── neighboring_footprints/
│   │   ├── mask/
│   │   ├── seismic/              # AFAD PGA
│   │   ├── dem/                  # Copernicus DEM
│   │   ├── footprint_target/
│   │   └── height_target/
│   ├── val/
│   └── test/
└── istanbul/
    └── (aynı yapı)
```

---

## 🎓 Kullanım

### 1. Multi-City Dataset

```python
from multi_city_dataset import create_multi_city_dataloaders

loaders = create_multi_city_dataloaders(
    data_root="./data",
    cities=["elazig", "istanbul"],
    batch_size=16,
    stage="footprint",
    use_seismic=True,
    use_topography=True,
    use_city_embedding=True,
)

# Birleşik loader (çok-şehirli)
for batch in loaders["train"]:
    conditional = batch["conditional_inputs"]  # (B, 10, 256, 256)
    target = batch["footprint"]                 # (B, 1, 256, 256)
    city_idx = batch["city_index"]              # (B,)
```

### 2. Sismik-CAIN-GAN Eğitimi

```python
from multi_city_training import MultiCityCAINTrainer

trainer = MultiCityCAINTrainer(
    data_root="./data",
    cities=["elazig", "istanbul"],
    batch_size=16,
    use_seismic=True,
    use_topography=True,
    lambda_seismic=10.0,
    lambda_terrain=5.0,
)

trainer.train(
    num_epochs_footprint=50,
    num_epochs_height=50,
)
```

### 3. CLI ile Eğitim

```bash
python multi_city_training.py \
    --data_root ./data \
    --cities elazig istanbul \
    --batch_size 16 \
    --epochs_footprint 50 \
    --epochs_height 50 \
    --learning_rate 0.0002
```

---

## 📈 Hedef Yayın Stratejisi

### SCI Makale

**Önerilen başlık:**
> *"Seismic-aware urban site planning with multi-scale CAIN-GAN: A comparative study of Istanbul and Elazığ, Turkey"*

**Hedef dergiler:**

| Dergi | IF | Kabul Olasılığı |
|-------|-----|------------------|
| **Automation in Construction** (orijinal CAIN-GAN dergisi) | 10.3 | %50-60 |
| **Sustainable Cities and Society** | 11.7 | %55-65 |
| **Building and Environment** | 7.1 | %50-60 |
| **Computers, Environment and Urban Systems** | 7.1 | %45-55 |
| **Cities** | 6.7 | %40-50 |
| **Buildings (MDPI)** — düşük risk | 3.8 | %75+ |

### Tez Yapısı (Önerilen)

```
Bölüm 1: Giriş ve Motivasyon
Bölüm 2: Literatür İncelemesi (CAIN-GAN + Türk şehirleri)
Bölüm 3: Metodoloji (Seismic-CAIN-GAN mimarisi)
Bölüm 4: Vaka 1 — Elazığ (Anadolu şehri)
Bölüm 5: Vaka 2 — İstanbul (megakent)
Bölüm 6: Karşılaştırmalı Analiz ve Cross-City Transfer
Bölüm 7: Tartışma ve Sınırlılıklar
Bölüm 8: Sonuç ve Gelecek Çalışmalar
```

---

## 🐛 Sorun Giderme

### OOM Hatası
```bash
python multi_city_training.py --batch_size 8
```

### Hesaplama yavaş
```bash
python multi_city_training.py --num_workers 2
```

### Sismik kanal yoksa
```bash
python multi_city_training.py --no_seismic --no_topography
```

---

## 📚 Dokümantasyon

| Dosya | İçerik |
|-------|--------|
| **[TURKISH_CITIES_GUIDE.md](TURKISH_CITIES_GUIDE.md)** | ⭐ Türk şehirleri veri toplama rehberi |
| [CAIN_GAN_IMPLEMENTATION.md](CAIN_GAN_IMPLEMENTATION.md) | Mimari teknik detayları |
| [QUICKSTART.md](QUICKSTART.md) | Hızlı başlangıç |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Proje istatistikleri |

---

## 📖 Atıf (Citation)

Bu projeyi araştırmanızda kullanırsanız hem **orijinal makaleyi** hem bu repo'yu atıf yapın:

### Orijinal CAIN-GAN Makalesi
```bibtex
@article{jiang2024automated,
  title={Automated site planning using CAIN-GAN model},
  author={Jiang, Feifeng and Ma, Jun and Webster, Christopher John and Wang, Wei and Cheng, Jack C.P.},
  journal={Automation in Construction},
  volume={159},
  pages={105286},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.autcon.2024.105286}
}
```

### Bu Repository
```bibtex
@misc{karabulut2026seismiccain,
  author = {Karabulut, Ilker},
  title = {Seismic-CAIN-GAN: Multi-City Urban Site Planning for Turkey},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/ilker-23/cain-gan-urban-design}
}
```

### Veri Kaynakları (Mutlaka Atıf Yapın)
- **OpenStreetMap** contributors (2024) — ODbL
- **Microsoft Global Building Footprints** (2023)
- **AFAD** Türkiye Deprem Tehlike Haritası (2024)
- **Copernicus DEM** GLO-30 — ESA (2024)
- **İBB Açık Veri Portalı** (2024)

---

## ⚖️ Lisans

MIT License — [LICENSE](LICENSE) dosyasına bakın.

> ⚠️ Orijinal CAIN-GAN makalesinin telif hakları yazarlara ve Elsevier'a aittir.
> Bu repo akademik araştırma amaçlı bir re-implementation + uzantıdır.

---

## 🌟 Star History

Bu projeyi beğendiyseniz ⭐ vermeyi unutmayın!

---

<div align="center">

**🎓 Türkiye kentsel planlama araştırması için yapıldı**
Made with ❤️ for urban planning research in Turkey

</div>
