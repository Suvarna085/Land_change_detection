"""
One-time patch extraction script.
Reads raw 10k×10k TIFFs and saves 256×256 PNG patches to disk.
Run once before training: python extract_patches.py
"""

import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# ── CONFIG ──────────────────────────────────────────────────
DATA_ROOT      = Path('./archive')
OUT_ROOT       = Path('./patches')   # where patches will be saved
PATCH_SIZE     = 256
PATCHES_PER_IMG = 50                 # how many random patches per image pair
SEED           = 42
# ────────────────────────────────────────────────────────────

np.random.seed(SEED)

def collect_pairs(data_root):
    pairs = []
    img1_base    = data_root / 'images_2006'          / '2006'
    img2_base    = data_root / 'images_2012'          / '2012'
    label1_base  = data_root / 'labels_land_cover_2006' / '2006'
    label2_base  = data_root / 'labels_land_cover_2012' / '2012'
    change_base  = data_root / 'labels_change'        / 'change'

    for region in ['D14', 'D35']:
        for img_path in sorted((img1_base / region).glob('*')):
            if img_path.suffix.lower() not in ['.tif', '.tiff', '.png', '.jpg']:
                continue
            name = img_path.name
            p = {
                'img1':   img1_base   / region / name,
                'img2':   img2_base   / region / name,
                'label1': label1_base / region / name,
                'label2': label2_base / region / name,
                'change': change_base / region / name,
                'stem':   f"{region}_{img_path.stem}",
            }
            if all(p[k].exists() for k in ['img1','img2','label1','label2','change']):
                pairs.append(p)
    return pairs


def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # ← UNCHANGED preserves uint16
    if mask is None:
        raise ValueError(f"Cannot load {path}")
    return mask   # stays uint16


def save_patch(arr, out_path):
    """Save RGB image or grayscale mask as PNG."""
    if arr.ndim == 3:
        cv2.imwrite(str(out_path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    else:
        np.save(str(out_path), arr)  # save masks as .npy for lossless storage


def extract_random_patches(img1, img2, label1, label2, change,
                            patch_size, n_patches):
    h, w = img1.shape[:2]
    patches = []
    max_top  = max(0, h - patch_size)
    max_left = max(0, w - patch_size)

    for _ in range(n_patches):
        top  = np.random.randint(0, max_top  + 1) if max_top  > 0 else 0
        left = np.random.randint(0, max_left + 1) if max_left > 0 else 0
        b, r = top + patch_size, left + patch_size

        patches.append({
            'img1':   img1  [top:b, left:r],
            'img2':   img2  [top:b, left:r],
            'label1': label1[top:b, left:r],
            'label2': label2[top:b, left:r],
            'change': change[top:b, left:r],
        })
    return patches


def main():
    pairs = collect_pairs(DATA_ROOT)
    print(f"Found {len(pairs)} image pairs")
    print(f"Extracting {PATCHES_PER_IMG} patches each → "
          f"{len(pairs) * PATCHES_PER_IMG} total patches")
    print(f"Saving to: {OUT_ROOT.resolve()}\n")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    total_saved = 0

    for pair in tqdm(pairs, desc="Processing image pairs"):
        img1   = load_image(pair['img1'])
        img2   = load_image(pair['img2'])
        label1 = load_mask(pair['label1'])
        label2 = load_mask(pair['label2'])
        change = load_mask(pair['change'])

        patches = extract_random_patches(
            img1, img2, label1, label2, change,
            PATCH_SIZE, PATCHES_PER_IMG
        )

        for i, patch in enumerate(patches):
            stem = f"{pair['stem']}_{i:03d}"
            save_patch(patch['img1'],   OUT_ROOT / f"{stem}_img1.png")
            save_patch(patch['img2'],   OUT_ROOT / f"{stem}_img2.png")
            save_patch(patch['label1'], OUT_ROOT / f"{stem}_lbl1.npy")
            save_patch(patch['label2'], OUT_ROOT / f"{stem}_lbl2.npy")
            save_patch(patch['change'], OUT_ROOT / f"{stem}_chg.npy")
            total_saved += 1

    print(f"\nDone! {total_saved} patch sets saved to {OUT_ROOT}/")
    print(f"Each patch set = 5 PNG files (~0.1–0.2 MB total)")
    print(f"Total disk usage: ~{total_saved * 0.15 / 1024:.1f} GB estimated")
    print(f"\nNow update dataloader.py: set data_root='./patches'")


if __name__ == "__main__":
    main()