# 🏙️ CAIN-GAN: Automated Urban Site Planning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/cain-gan-urban-design/blob/main/CAIN_GAN_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **GAN-Based Automated Urban Site Planning** — PyTorch implementation inspired by the paper *"Automated site planning using CAIN-GAN model"* (Jiang et al., *Automation in Construction*, Elsevier 2024)

Bu repo, **uydu görüntülerinden kentsel peyzaj/site tasarımları** üretmek için CAIN-GAN (Context-Aware Image-to-Image Network) mimarisinin tam PyTorch implementasyonunu içerir. SCI makale ve tez projeleri için tasarlanmıştır.

---

## 🚀 Hızlı Başlangıç — Google Colab

**En kolay yol:** Colab'da tek tıkla çalıştırın 👇

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ilker-23/cain-gan-urban-design/blob/main/CAIN_GAN_Colab.ipynb)

**Adımlar:**
1. Yukarıdaki Colab butonuna tıklayın
2. `Runtime → Change runtime type → GPU` seçin (T4 veya A100)
3. `Runtime → Run all` ile tüm hücreleri çalıştırın

---

## 📂 Proje Yapısı

```
cain-gan-urban-design/
│
├── 🔧 Core CAIN-GAN
│   ├── cain_dataset.py           # Multi-channel veri yükleyici
│   ├── cain_architecture.py      # Model mimarisi (Generator, Discriminator)
│   └── cain_training.py          # İki-aşamalı eğitim
│
├── 📚 Modular Framework
│   ├── dataset.py                # Pix2Pix-style dataset
│   ├── augmentation.py           # Veri artırma stratejileri
│   └── config_loader.py          # Konfigürasyon yönetimi
│
├── ⚙️ Configuration
│   ├── config.yaml               # Eğitim konfigürasyonu
│   └── requirements.txt          # Bağımlılıklar
│
├── 📓 Notebook
│   └── CAIN_GAN_Colab.ipynb      # Colab eğitim notebook'u
│
├── 📖 Documentation
│   ├── CAIN_GAN_IMPLEMENTATION.md  # Detaylı teknik docs
│   ├── QUICKSTART.md               # Hızlı başlangıç
│   ├── PROJECT_SUMMARY.md          # Proje özeti
│   └── README.md                   # Bu dosya
│
└── 📝 Example
    └── example_usage.py          # Kullanım örnekleri
```

---

## 🎯 Özellikler

### Two-Stage Progressive Generation
```
Stage 1: Footprint Construction (Bina yer planı)
   ↓
Stage 2: Height Completion (Bina yükseklikleri)
```

### Mimari Bileşenler
- ✅ **Contextual Attention Mechanism** — Çevre dokusundan özellik çıkarımı
- ✅ **Dual-Path Bottleneck** — Hallucination + Attention paths
- ✅ **Residual Blocks** — Derin özellik çıkarımı
- ✅ **Spectral Normalization** — Eğitim kararlılığı
- ✅ **Multi-Channel Conditioning** — Kontekst, planlama, mask

### Veri Pipeline
- ✅ PyTorch `Dataset` ve `DataLoader` desteği
- ✅ Albumentations augmentation
- ✅ 256×256 görüntü boyutu (standart)
- ✅ Paired ve unpaired modlar
- ✅ Multi-format desteği (JPG, PNG, TIFF)

---

## 💻 Yerel Kurulum

### 1. Repository'i klonlayın

```bash
git clone https://github.com/ilker-23/cain-gan-urban-design.git
cd cain-gan-urban-design
```

### 2. Sanal ortam oluşturun (önerilen)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya: venv\Scripts\activate  # Windows
```

### 3. Bağımlılıkları yükleyin

```bash
pip install -r requirements.txt
```

### 4. Veri yapınızı hazırlayın

```bash
mkdir -p data/{train,val,test}/{site_context,planning_guidance,neighboring_footprints,mask,footprint_target,height_target}
```

### 5. Eğitimi başlatın

```bash
python cain_training.py \
    --data_root ./data \
    --batch_size 16 \
    --epochs_footprint 50 \
    --epochs_height 50
```

---

## 📊 Veri Formatı

CAIN-GAN **multi-channel conditional** veri kullanır:

```
data/
├── train/
│   ├── site_context/              # roads=1, vegetation=2, water=3
│   ├── planning_guidance/         # one-hot encoded land use
│   ├── neighboring_footprints/    # mevcut binalar (binary)
│   ├── mask/                      # 0=tasarım alanı, 1=kontekst
│   ├── footprint_target/          # ground truth footprint
│   └── height_target/             # ground truth heights
├── val/
└── test/
```

Detaylı veri hazırlama için: [QUICKSTART.md](QUICKSTART.md)

---

## 🎓 Kullanım Örnekleri

### Veri yükleme

```python
from cain_dataset import create_cain_dataloaders

fp_train, fp_val, h_train, h_val = create_cain_dataloaders(
    data_root="./data",
    batch_size=16,
)

for batch in fp_train:
    conditional = batch["conditional_inputs"]  # (B, 5+, 256, 256)
    target = batch["footprint"]                 # (B, 1, 256, 256)
```

### Model oluşturma

```python
from cain_architecture import CAINGANModel

model = CAINGANModel(
    conditional_channels=5,
    ngf=64,  # Generator filtre sayısı
    ndf=64,  # Discriminator filtre sayısı
).cuda()
```

### Eğitim

```python
from cain_training import CAINTrainer

trainer = CAINTrainer(data_root="./data", batch_size=16)
trainer.train(num_epochs_footprint=50, num_epochs_height=50)
```

---

## 📈 Beklenen Performans

| Aşama | Epoch | G_loss | D_loss | Val L1 |
|-------|-------|--------|--------|--------|
| Footprint | 1 | ~1.0 | ~0.7 | ~0.45 |
| Footprint | 25 | ~0.3 | ~0.5 | ~0.18 |
| Footprint | 50 | ~0.15 | ~0.45 | ~0.10 |
| Height | 50 | ~0.18 | ~0.45 | ~0.12 |

---

## 🔬 Mimari Detayları

### Loss Function

```
Stage 1 (Footprint):
  L_f = 100 · L1(F_pred, F_gt) + 1 · L_adv(D_f)

Stage 2 (Height):
  L_h = 100 · L1(H_pred, H_gt) + 1 · L_adv(D_h)
```

### Eğitim Hyperparametreleri

```yaml
optimizer: Adam
learning_rate: 0.0002
beta1: 0.5
beta2: 0.999
lambda_rec: 100      # Reconstruction loss weight
lambda_adv: 1        # Adversarial loss weight
image_size: 256
batch_size: 16
```

Detaylar: [CAIN_GAN_IMPLEMENTATION.md](CAIN_GAN_IMPLEMENTATION.md)

---

## 🐛 Sorun Giderme

### Out of Memory (OOM)
```bash
python cain_training.py --batch_size 8  # 16 yerine
```

### Eğitim çok yavaş
```bash
python cain_training.py --num_workers 2
```

### NaN losses
```python
# Learning rate'i düşürün
--learning_rate 0.0001
```

Detaylı troubleshooting: [QUICKSTART.md](QUICKSTART.md#-troubleshooting)

---

## 📚 Dökümantasyon

| Dosya | İçerik |
|-------|--------|
| [CAIN_GAN_IMPLEMENTATION.md](CAIN_GAN_IMPLEMENTATION.md) | Detaylı teknik dökümantasyon ve mimari |
| [QUICKSTART.md](QUICKSTART.md) | Hızlı başlangıç rehberi |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Proje özeti ve istatistikler |
| [example_usage.py](example_usage.py) | 6 kullanım örneği |

---

## 📖 Atıf (Citation)

Bu implementasyonu araştırmanızda kullanırsanız, **orijinal makaleyi** atıf yapın:

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

### İlgili Çalışmalar

- **Pix2Pix:** Isola et al. (CVPR 2017)
- **Spectral Norm:** Miyato et al. (ICLR 2018)
- **CycleGAN:** Zhu et al. (ICCV 2017)

---

## ⚖️ Lisans

Bu proje **MIT Lisansı** altında dağıtılır. Detaylar: [LICENSE](LICENSE)

⚠️ **Not:** Bu kod, akademik araştırma amaçlı bir re-implementasyondur. Orijinal CAIN-GAN makalesi telif hakları ilgili yazarlara ve Elsevier'a aittir.

---

## 🙏 Katkıda Bulunma

Pull request'ler hoş karşılanır! Major değişiklikler için önce bir issue açın.

### Geliştirme rehberi
1. Repo'yu fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add yeni özellik'`)
4. Branch'i push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

---

## 📞 İletişim

- **GitHub Issues:** Bug raporları ve özellik istekleri için
- **Tartışmalar:** GitHub Discussions'da sorularınızı paylaşın

---

## 🌟 Star History

Bu projeyi beğendiyseniz ⭐ vermeyi unutmayın!

---

<div align="center">

**🎓 Akademik araştırma ve tez projeleri için yapıldı**

Made with ❤️ for urban planning research

</div>
