# CAIN-GAN Quick Start Guide

## 📋 Project Structure

```
GAN Urban Design/
├── dataset.py                      # Modular PyTorch Dataset (Pix2Pix style)
├── augmentation.py                 # Data augmentation strategies
├── config_loader.py               # Configuration management
├── config.yaml                    # Training configuration
│
├── cain_dataset.py                # CAIN-specific dataset loader
├── cain_architecture.py           # CAIN-GAN model architecture
├── cain_training.py              # Training script
│
├── CAIN_GAN_IMPLEMENTATION.md     # Detailed architecture docs
├── README.md                      # Dataset module documentation
└── QUICKSTART.md                  # This file
```

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create directory structure:

```bash
mkdir -p data/{train,val,test}/{site_context,planning_guidance,neighboring_footprints,mask,footprint_target,height_target}
```

### 3. Train the Model

```bash
python cain_training.py \
    --data_root ./data \
    --batch_size 16 \
    --learning_rate 0.0002 \
    --epochs_footprint 50 \
    --epochs_height 50 \
    --checkpoint_dir ./checkpoints
```

## 📊 Understanding Your Data

### Input Format

CAIN-GAN requires **5 input channels**:

| Channel | What it is | How to encode |
|---------|-----------|---------------|
| 1 | Site Context | Roads=1, Vegetation=2, Water=3 |
| 2-5 | Planning Guidance | One-hot encode land use types |
| 6 | Neighboring Buildings | Binary (0/1) |
| 7 | Mask | Binary: 0=design area, 1=context |

### Example Data Preparation

```python
import numpy as np
from PIL import Image

# Site context: roads, vegetation, water
site_context = np.zeros((256, 256), dtype=np.uint8)
site_context[roads] = 1
site_context[vegetation] = 2
site_context[water] = 3
Image.fromarray(site_context).save("site_context/sample_001.png")

# Planning guidance: land use types (4 classes)
planning = np.zeros((256, 256, 4), dtype=np.uint8)
planning[:,:,0][residential_areas] = 1  # Residential
planning[:,:,1][commercial_areas] = 1   # Commercial
planning[:,:,2][manufacturing] = 1      # Manufacturing
planning[:,:,3][mixed_use] = 1          # Mixed
Image.fromarray(planning).save("planning_guidance/sample_001.png")

# Neighboring footprints
footprints = np.zeros((256, 256), dtype=np.uint8)
footprints[existing_buildings] = 1
Image.fromarray(footprints * 255).save("neighboring_footprints/sample_001.png")

# Mask: 0=design area, 1=context
mask = np.ones((256, 256), dtype=np.uint8)
mask[design_area] = 0
Image.fromarray(mask * 255).save("mask/sample_001.png")

# Targets
footprint_output = ...  # Ground truth footprint (256x256, binary)
height_output = ...     # Ground truth heights (256x256, grayscale)
```

## 🎯 Training Workflow

### Phase 1: Footprint Construction (50 epochs)

```
Input: Site context + Planning guidance
       ↓
   Generator Gf
       ↓
   Output: Building footprints
       ↓
   Loss = 100 * L1_loss + 1 * adversarial_loss
```

**What the model learns:**
- Where buildings should be placed
- How to respect existing structures
- Planning constraints and zoning rules

### Phase 2: Height Completion (50 epochs)

```
Input: Site context + Generated footprints
       ↓
   Generator Gh
       ↓
   Output: Building heights
       ↓
   Loss = 100 * L1_loss + 1 * adversarial_loss
```

**What the model learns:**
- Building height variation
- Height coherence within districts
- Planning guidance compliance

## 💻 Training Loop Details

### Step-by-step execution:

```python
from cain_training import CAINTrainer

# Initialize trainer
trainer = CAINTrainer(
    data_root="/path/to/data",
    batch_size=16,
    learning_rate=0.0002,
    device="cuda"
)

# Stage 1: Footprint Construction
# --------------------------------
trainer.setup_footprint_stage()
for epoch in range(50):
    # Generate footprints from random noise + conditioning
    footprint = trainer.generator_f(conditional_input)
    
    # Adversarial training loop
    # 1. Generator: Make footprints look real
    # 2. Discriminator: Distinguish real from fake
    
    # Validation + checkpoint
    trainer.validate_footprint()
    if epoch % 10 == 0:
        trainer.save_checkpoint("footprint", epoch)

# Stage 2: Height Completion
# ---------------------------
trainer.setup_height_stage()
for epoch in range(50):
    # Generate heights conditioned on footprints
    heights = trainer.generator_h(conditional_input)
    
    # Similar adversarial training
    trainer.validate_height()
    if epoch % 10 == 0:
        trainer.save_checkpoint("height", epoch)
```

## 📈 Monitoring Training

### Expected Loss Progression

```
Stage 1: Footprint Construction
Epoch 1:   G_loss=0.85 D_loss=0.70
Epoch 10:  G_loss=0.45 D_loss=0.55
Epoch 25:  G_loss=0.28 D_loss=0.48
Epoch 50:  G_loss=0.15 D_loss=0.42

Stage 2: Height Completion
Epoch 1:   G_loss=0.92 D_loss=0.68
Epoch 10:  G_loss=0.52 D_loss=0.54
Epoch 25:  G_loss=0.32 D_loss=0.50
Epoch 50:  G_loss=0.18 D_loss=0.45
```

**Healthy signs:**
✓ Both G and D losses decrease over time
✓ D_loss stabilizes around 0.4-0.5
✓ G_loss decreases consistently
✓ Validation L1 loss decreases

**Warning signs:**
⚠️ D_loss → 0 (discriminator too strong)
⚠️ G_loss → 0 then increases (mode collapse)
⚠️ Losses fluctuate wildly (learning rate too high)
⚠️ Validation loss doesn't decrease (data issue)

### Adjusting Hyperparameters

If D_loss is too low (< 0.3):
```python
# Increase generator loss weight
lambda_adv = 2.0  # Instead of 1.0
```

If G_loss is increasing:
```python
# Reduce learning rate
learning_rate = 0.0001  # Instead of 0.0002
```

If training is unstable:
```python
# Increase reconstruction loss weight
lambda_rec = 200  # Instead of 100
```

## 🎨 Generating New Designs

### Inference on new data:

```python
import torch
from cain_architecture import CAINGANModel

# Load trained model
model = CAINGANModel().cuda()
model.generator_footprint.load_state_dict(
    torch.load("checkpoints/footprint_epoch_50.pth")["generator"]
)
model.generator_height.load_state_dict(
    torch.load("checkpoints/height_epoch_50.pth")["generator"]
)

# Prepare input for a new site
site_context = torch.randn(1, 1, 256, 256).cuda()
planning_guidance = torch.randn(1, 4, 256, 256).cuda()
mask = torch.randn(1, 1, 256, 256).cuda()
footprints = torch.randn(1, 1, 256, 256).cuda()

# Concatenate inputs
conditional = torch.cat(
    [site_context, planning_guidance, footprints, mask],
    dim=1
)  # (1, 7, 256, 256)

# Generate footprints (Stage 1)
with torch.no_grad():
    footprint_output = model.generator_footprint(conditional)

# Update conditional with generated footprint
conditional[:, 6:7] = footprint_output

# Generate heights (Stage 2)
with torch.no_grad():
    height_output = model.generator_height(conditional)

# Save results
footprint_output.cpu().numpy()[0, 0].save("output_footprint.png")
height_output.cpu().numpy()[0, 0].save("output_height.png")
```

## ✅ Checklist Before Training

- [ ] Dataset folder structure created
- [ ] All data channels encoded properly
- [ ] 256×256 image size verified
- [ ] Train/val/test split created (70/15/15)
- [ ] At least 100 samples in training set
- [ ] GPU available (or CPU fallback configured)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Configuration reviewed (`config.yaml`)

## 🐛 Troubleshooting

### "No data found" error
```python
# Check directory structure
import os
assert os.path.exists("data/train/site_context/")
assert len(os.listdir("data/train/site_context/")) > 0
```

### Out of Memory (OOM)
```bash
# Reduce batch size
python cain_training.py --batch_size 8  # Instead of 16
```

### Training is very slow
```bash
# Reduce num_workers on systems with limited CPU
python cain_training.py --num_workers 2  # Instead of 4
```

### NaN losses
```python
# Usually indicates numerical instability
# 1. Reduce learning rate: 0.0001 instead of 0.0002
# 2. Increase reconstruction loss weight: 150 instead of 100
# 3. Check data normalization (should be [0, 1] or [-1, 1])
```

## 📚 File Reference

### Main Files to Use

| File | Purpose | When to use |
|------|---------|------------|
| `cain_dataset.py` | Load CAIN-formatted data | During training |
| `cain_architecture.py` | Model definitions | Model initialization |
| `cain_training.py` | Training script | `python cain_training.py ...` |
| `config.yaml` | Configuration | Customize hyperparameters |

### Reference Files

| File | Purpose |
|------|---------|
| `dataset.py` | Alternative Pix2Pix-style dataset |
| `augmentation.py` | Data augmentation utilities |
| `CAIN_GAN_IMPLEMENTATION.md` | Detailed technical docs |

## 🎓 Next Steps

1. **Data Collection:** Gather satellite and design data for your city
2. **Data Preprocessing:** Create multi-channel inputs as shown above
3. **Model Training:** Run `cain_training.py` with your data
4. **Evaluation:** Implement urban design quality metrics
5. **Deployment:** Use trained model for site planning decisions

## 🔗 Citation

If you use this implementation in your research:

```bibtex
@article{jiang2024automated,
  title={Automated site planning using CAIN-GAN model},
  author={Jiang, Feifeng and Ma, Jun and Webster, Christopher John and Wang, Wei and Cheng, Jack CP},
  journal={Automation in Construction},
  volume={159},
  pages={105286},
  year={2024},
  publisher={Elsevier}
}
```

## 📞 Support

For issues or questions:
1. Check `CAIN_GAN_IMPLEMENTATION.md` for detailed architecture info
2. Review `README.md` for dataset module details
3. Examine example outputs in checkpoints directory
4. Check console output for error messages

---

**Status:** ✅ Ready to train
**Last Updated:** 2026-05-25
**Version:** 1.0
