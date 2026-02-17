# Semantic Change Detection in Satellite Imagery

Comparing FC-EF-Res (CNN) and ChangeFormer (Transformer) 
for semantic change detection on the HRSCD dataset.

## Methods
- **FC-EF-Res**: Siamese ResNet with dual decoder, two-phase training
- **ChangeFormer**: Siamese EfficientNet-B0 with multi-scale fusion

## Dataset
HRSCD - High Resolution Semantic Change Detection Dataset
- 291 image pairs (2006 and 2012)
- 6 land cover classes + binary change mask

## Setup
pip install torch torchvision albumentations opencv-python timm einops tqdm scikit-learn

## Usage
# Extract patches (run once)
python extract_patches.py

# Train FC-EF-Res
python main_train.py

# Skip Phase 1, go straight to Phase 2
python main_train.py --skip-phase1

# Train ChangeFormer
python train_changeformer.py
```

---

## **Important**
- `archive/` and `patches/` folders are large (GBs) — keep them in `.gitignore`
- `checkpoints/` can be uploaded if you want to share trained models, but they're large too
- Only upload **code files** (.py) and README

---

## **Your File Structure on GitHub Should Look Like**
```
semantic-change-detection/
├── README.md
├── .gitignore
├── dataloader.py
├── model.py
├── trainer.py
├── main_train.py
├── extract_patches.py
├── changeformer.py
└── train_changeformer.py
