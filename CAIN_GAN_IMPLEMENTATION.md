# CAIN-GAN Implementation Guide

## Overview

This implementation is based on the research paper:
**"Automated site planning using CAIN-GAN model"**
- Published in: *Automation in Construction* (Elsevier, 2024)
- Authors: Feifeng Jiang, Jun Ma, Christopher John Webster, Wei Wang, Jack C.P. Cheng
- Case study: New York City (NYC)

## Architecture Summary

### Two-Stage Progressive Generation

CAIN-GAN follows a "footprint first, height next" approach, mirroring real-world urban design practices:

```
┌─────────────────────────────────────────┐
│  STAGE 1: FOOTPRINT CONSTRUCTION        │
│  (Ground-level building layout)         │
│                                         │
│  Input: Site context + Planning guide   │
│  Output: Building footprints           │
│                                         │
│  Generator: Gf                          │
│  Discriminator: Df                      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  STAGE 2: HEIGHT COMPLETION             │
│  (3D building heights)                  │
│                                         │
│  Input: Generated footprints + Context  │
│  Output: Building heights              │
│                                         │
│  Generator: Gh                          │
│  Discriminator: Dh                      │
└─────────────────────────────────────────┘
```

## Core Components

### 1. Contextual Attention Mechanism

Extracts characteristic textures from the surrounding built environment.

**Key Innovation:** The dual-path bottleneck combines:
- **Top Path:** Hallucination using residual blocks
- **Bottom Path:** Contextual attention on background features

```python
class ContextualAttention(nn.Module):
    """
    Computes attention weights between foreground and background features.
    Allows the generator to synthesize solutions responsive to context.
    """
```

**Why It Matters:**
- Ensures generated designs are coherent with surrounding structures
- Preserves spatial relationships and urban continuity
- Allows fine-grained control over context influence

### 2. Residual Blocks with Spectral Normalization

```python
class ResidualBlock(nn.Module):
    - Enables efficient training of deep networks
    - Spectral normalization for Lipschitz constraint
    - Improves feature map quality
```

**Benefits:**
- ✓ Better gradient flow during backpropagation
- ✓ Training stability
- ✓ Higher quality feature extraction

### 3. Generator Architecture (Gf/Gh)

**Components:**
- **Encoder:** Compresses multi-channel input features
- **Dual-path Bottleneck:** Contextual attention mechanism
- **Decoder:** Reconstructs output (footprint or height)

**Input Channels (5 total):**
```
1. Site Context: Roads (1) + Vegetation (2) + Water (3)
2. Planning Guidance: Land use types (one-hot encoded)
3. Neighboring Footprints: Existing buildings
4. Mask: Binary mask (0=design area, 1=surrounding)
```

### 4. Discriminator Architecture (Df/Dh)

**Components:**
```
ConvBlock 1 → ConvBlock 2 → ConvBlock 3 → ConvBlock 4
                                          ↓
                                    Classifier
                                          ↓
                                   Binary output (real/fake)
```

**Features:**
- 4 convolutional blocks with spectral normalization
- LeakyReLU activation (0.2 slope)
- No batch normalization (spectral norm replaces it)

## Input Data Structure

### Multi-Channel Conditioning

CAIN-GAN requires structured data preparation:

```
data_root/
├── train/
│   ├── site_context/              # Shape: (256, 256, 1)
│   │   ├── site_001.png           # Encoded: roads=1, veg=2, water=3
│   │   └── ...
│   │
│   ├── planning_guidance/         # Shape: (256, 256, C)
│   │   ├── site_001.png           # One-hot encoded land uses
│   │   └── ...
│   │
│   ├── neighboring_footprints/    # Shape: (256, 256, 1)
│   │   ├── site_001.png           # Binary: existing buildings
│   │   └── ...
│   │
│   ├── mask/                      # Shape: (256, 256, 1)
│   │   ├── site_001.png           # Binary: 0=design, 1=context
│   │   └── ...
│   │
│   ├── footprint_target/          # Shape: (256, 256, 1)
│   │   ├── site_001.png           # Ground truth footprints
│   │   └── ...
│   │
│   └── height_target/             # Shape: (256, 256, 1)
│       ├── site_001.png           # Ground truth heights
│       └── ...
│
├── val/
│   └── (same structure)
│
└── test/
    └── (same structure)
```

**Data Encoding:**

| Channel | Meaning | Values |
|---------|---------|--------|
| Site Context | Road | 1 |
| | Vegetation | 2 |
| | Water | 3 |
| Planning Guidance | Residential | one-hot [1,0,0,0] |
| | Commercial | one-hot [0,1,0,0] |
| | Manufacturing | one-hot [0,0,1,0] |
| | Mixed-use | one-hot [0,0,0,1] |
| Mask | Design area | 0 |
| | Context area | 1 |

## Loss Functions

### Stage 1: Footprint Construction

```
Lf(Gf, Df) = λf_rec · Lf_rec + λf_adv · Lf_adv

where:
  Lf_rec = ||Gf(Isc, Ipg, M, Fin) - Fgt||₁    (L1 reconstruction)
  Lf_adv = -log(Df(Gf(...)))                   (Adversarial)
  λf_rec = 100 (typically)
  λf_adv = 1
```

### Stage 2: Height Completion

```
Lh(Gh, Dh) = λh_rec · Lh_rec + λh_adv · Lh_adv

where:
  Lh_rec = ||Gh(Isc, Ipg, M, Hin) - Igt||₁
  Lh_adv = -log(Dh(Gh(...)))
  λh_rec = 100
  λh_adv = 1
```

**Key Points:**
- L1 loss (not L2) for reconstruction (better for sharp edges)
- High weight on reconstruction (λ=100) vs. adversarial (λ=1)
- Balanced training of G and D through careful weighting

## Implementation Files

### 1. `cain_dataset.py`
**Purpose:** Data loading and preprocessing for CAIN-GAN

**Classes:**
- `CAINDataset`: Single-stage dataset loader
- `CAINProgressiveDataset`: Wrapper for two-stage training
- `create_cain_dataloaders()`: Helper function

**Usage:**
```python
from cain_dataset import create_cain_dataloaders

fp_train, fp_val, h_train, h_val = create_cain_dataloaders(
    data_root="/path/to/data",
    batch_size=16,
)

# Stage 1: Train footprint
for batch in fp_train:
    conditional = batch["conditional_inputs"]  # (B, 5, 256, 256)
    target = batch["footprint"]                 # (B, 1, 256, 256)
```

### 2. `cain_architecture.py`
**Purpose:** Model architecture implementation

**Classes:**
- `ContextualAttention`: Attention mechanism
- `ConvBlock`: Basic convolutional block
- `ResidualBlock`: Residual connection block
- `DualPathBottleneck`: Combined hallucination + attention
- `CAINGeneratorFootprint`: Stage 1 generator
- `CAINDiscriminatorFootprint`: Stage 1 discriminator
- `CAINGeneratorHeight`: Stage 2 generator
- `CAINDiscriminatorHeight`: Stage 2 discriminator
- `CAINGANModel`: Complete model wrapper

**Usage:**
```python
from cain_architecture import CAINGANModel

model = CAINGANModel(
    conditional_channels=5,
    ngf=64,  # Generator filters
    ndf=64,  # Discriminator filters
)

# Stage 1
footprint = model.forward_footprint(conditional_input)

# Stage 2
height = model.forward_height(conditional_input)
```

### 3. `cain_training.py`
**Purpose:** Training loop and optimization

**Class:**
- `CAINTrainer`: Orchestrates two-stage training

**Features:**
- Two separate training phases (footprint → height)
- Proper loss balancing
- Validation at each epoch
- Checkpoint saving

**Usage:**
```bash
python cain_training.py \
    --data_root /path/to/data \
    --batch_size 16 \
    --epochs_footprint 50 \
    --epochs_height 50 \
    --learning_rate 0.0002
```

### 4. Supporting Files
- `dataset.py`: Original modular dataset (Pix2Pix compatible)
- `augmentation.py`: Data augmentation strategies
- `config_loader.py`: Configuration management
- `config.yaml`: Training configuration template

## Training Strategy

### Stage 1: Footprint Construction (50 epochs)

```python
trainer.setup_footprint_stage()

for epoch in range(50):
    # Training
    generator_footprint.train()
    discriminator_footprint.train()
    
    for batch in train_loader:
        # Generate footprints
        Fpred = generator_footprint(conditional_input)
        
        # Generator loss
        L_adv = adversarial_loss(discriminator(Fpred), real_label)
        L_rec = L1_loss(Fpred, target_footprint)
        L_G = 1.0 * L_adv + 100.0 * L_rec
        
        # Discriminator loss
        L_D_real = adversarial_loss(discriminator(target), real_label)
        L_D_fake = adversarial_loss(discriminator(Fpred.detach()), fake_label)
        L_D = (L_D_real + L_D_fake) / 2
    
    # Validation
    validate(footprint_val_loader)
    save_checkpoint()
```

### Stage 2: Height Completion (50 epochs)

```python
trainer.setup_height_stage()

for epoch in range(50):
    # Similar training loop but with:
    # - Generated footprints as input
    # - Height prediction as output
    
    for batch in train_loader:
        Hpred = generator_height(
            site_context,
            planning_guidance,
            mask,
            generated_footprints  # From Stage 1
        )
        # Rest of training loop...
```

## Training Hyperparameters

```yaml
# Optimization
learning_rate: 0.0002
beta1: 0.5              # Adam momentum
beta2: 0.999            # Adam momentum

# Loss weights
lambda_rec: 100         # Reconstruction loss weight
lambda_adv: 1           # Adversarial loss weight

# Architecture
ngf: 64                 # Generator base filters
ndf: 64                 # Discriminator base filters
conditional_channels: 5 # Input channels

# Training
batch_size: 16
num_epochs_footprint: 50
num_epochs_height: 50
save_interval: 10
```

## Batch Information

### Input Batch Structure

```python
batch = {
    "conditional_inputs": torch.Tensor,   # (B, 5, 256, 256)
    "footprint": torch.Tensor,            # (B, 1, 256, 256)
    "height": torch.Tensor,               # (B, 1, 256, 256)
    "sample_id": str,                     # Sample identifier
    "stage": str,                         # "footprint" or "height"
}
```

### Tensor Specifications

| Tensor | Shape | Range | Description |
|--------|-------|-------|-------------|
| Site Context | (B, 1, 256, 256) | [0, 3] | Categorical: road/veg/water |
| Planning Guidance | (B, 4, 256, 256) | {0, 1} | One-hot land use types |
| Neighboring Footprints | (B, 1, 256, 256) | [0, 1] | Binary existing buildings |
| Mask | (B, 1, 256, 256) | {0, 1} | Binary design area |
| Footprint Target | (B, 1, 256, 256) | [0, 1] | Generated/target footprint |
| Height Target | (B, 1, 256, 256) | [0, 1] | Generated/target heights |

## Key Differences from Standard GANs

### vs. Pix2Pix
| Feature | Pix2Pix | CAIN-GAN |
|---------|---------|----------|
| Architecture | U-Net | U-Net + Attention |
| Attention | None | Contextual attention |
| Training | Single-stage | Two-stage progressive |
| Input channels | 3-4 | 5+ (multi-channel) |
| **Target domain** | Image-to-image | Urban site planning |

### vs. CycleGAN
- CAIN-GAN: **Paired** training (requires paired data)
- CycleGAN: **Unpaired** training (no correspondence needed)
- CAIN-GAN: Conditional on context/planning guidance
- CycleGAN: No conditioning

## Advantages of CAIN-GAN

✓ **Context-Aware:** Explicitly models surrounding environment
✓ **Semantic Alignment:** Two-stage approach mirrors design workflow
✓ **Stable Training:** Spectral normalization + attention mechanism
✓ **Flexible Conditioning:** Multi-channel input for diverse constraints
✓ **Urban Design:** Domain-specific architecture for site planning

## Limitations & Considerations

⚠ **Data Requirements:** Requires paired (satellite, design) data
⚠ **Preprocessing:** Multi-channel encoding is complex
⚠ **Computational Cost:** Two-stage training requires 2× epochs
⚠ **Hyperparameter Tuning:** λ ratios affect output quality
⚠ **Evaluation Metrics:** May need custom metrics (urban coherence, etc.)

## Next Steps

1. **Data Preparation:** Encode multi-channel input data
2. **Model Training:** Run `cain_training.py` with your dataset
3. **Inference:** Generate designs for new sites
4. **Evaluation:** Measure urban design quality metrics
5. **Fine-tuning:** Adjust hyperparameters based on results

## References

- **Original Paper:** Jiang et al., "Automated site planning using CAIN-GAN model", *Automation in Construction*, 2024
- **Related Work:** 
  - Pix2Pix: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017
  - Spectral Norm: Miyato et al., "Spectral Normalization for GAN", ICLR 2018
  - Attention: Balntas et al., "Paying More Attention to Attention", ICCV 2017

---

**Last Updated:** 2026-05-25
**Status:** Implementation complete, ready for training
