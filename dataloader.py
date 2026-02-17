"""
HRSCD Dataloader - OPTIMIZED VERSION

OPTIMIZATIONS:
- Reduced augmentations (only essential flips and rotations)
- Warning suppression for TIFF files
- Better memory management
- Faster patch extraction

Dataset structure:
archive/
├── images_2006/2006/D14/ and D35/
├── images_2012/2012/D14/ and D35/
├── labels_land_cover_2006/2006/D14/ and D35/
├── labels_land_cover_2012/2012/D14/ and D35/
└── labels_change/change/D14/ and D35/
"""

import os
import warnings

# Suppress warnings
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HRSCDDataset(Dataset):

    def __init__(self, data_root, patch_size=256, patches_per_image=50,
                 augment=True, train_split=0.8):

        self.data_root = Path(data_root)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment

        self.data_list = self.collect_patches()

        n_train = int(len(self.data_list) * train_split)
        self.train_list = self.data_list[:n_train]
        self.test_list  = self.data_list[n_train:]
        self.current_list = self.train_list

        print(f"Total patches: {len(self.data_list)}")
        print(f"Train: {len(self.train_list)}, Test: {len(self.test_list)}")

        self.classes = {
            0: 'No information',    1: 'Artificial surfaces',
            2: 'Agricultural areas',3: 'Forests',
            4: 'Wetlands',          5: 'Water'
        }
        self.transform = self.get_transforms()

    def collect_patches(self):
        """Each patch set is identified by its img1 file stem."""
        patches = []
        for img1_path in sorted(self.data_root.glob('*_img1.png')):
            stem = img1_path.stem[:-5]   # strip '_img1'
            p = {
                'img1':   self.data_root / f"{stem}_img1.png",
                'img2':   self.data_root / f"{stem}_img2.png",
                'label1': self.data_root / f"{stem}_lbl1.npy",
                'label2': self.data_root / f"{stem}_lbl2.npy",
                'change': self.data_root / f"{stem}_chg.npy",
                'name':   stem,
            }
            if all(p[k].exists() for k in ['img1','img2','label1','label2','change']):
                patches.append(p)
        return patches

    def set_split(self, split='train'):
        if split == 'train':
            self.current_list = self.train_list
            self.augment = True
        else:
            self.current_list = self.test_list
            self.augment = False
        self.transform = self.get_transforms()
        print(f"Using {split} set: {len(self.current_list)} patches")

    def get_transforms(self):
        if self.augment:
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={
                'image2': 'image', 'label1': 'mask',
                'label2': 'mask',  'change_mask': 'mask'
            })
        else:
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], additional_targets={
                'image2': 'image', 'label1': 'mask',
                'label2': 'mask',  'change_mask': 'mask'
            })

    def load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot load {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def load_mask(self, path):
        return np.load(str(path)).astype(np.int32)


    def __len__(self):
        return len(self.current_list)   # no patches_per_image multiplier needed

    def __getitem__(self, idx):
        item = self.current_list[idx]

        img1       = self.load_image(item['img1'])
        img2       = self.load_image(item['img2'])
        label1     = self.load_mask(item['label1'])
        label2     = self.load_mask(item['label2'])
        change_mask = self.load_mask(item['change'])

        transformed = self.transform(
            image=img1, image2=img2,
            label1=label1, label2=label2,
            change_mask=change_mask
        )

        return {
            'image1':      transformed['image'],
            'image2':      transformed['image2'],
            'label1':      transformed['label1'].long(),
            'label2':      transformed['label2'].long(),
            'change_mask': transformed['change_mask'].long(),
            'name':        item['name']
        }

def compute_class_weights(dataset, num_batches=10):
    """
    Compute class weights using sampled patches (FAST).
    """
    print("Computing class weights (patch-based)...")

    from torch.utils.data import DataLoader

    dataset.set_split('train')

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # keep 0 to avoid Windows multiprocessing issues
    )

    lc_counts = torch.zeros(6)
    change_counts = torch.zeros(2)

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        labels = batch['label1']
        changes = batch['change_mask']

        for c in range(6):
            lc_counts[c] += (labels == c).sum()

        change_counts[0] += (changes == 0).sum()
        change_counts[1] += (changes == 1).sum()

    # Safe inverse frequency weights
    lc_weights = torch.zeros_like(lc_counts, dtype=torch.float32)
    for i in range(len(lc_counts)):
        if lc_counts[i] > 0:
            lc_weights[i] = lc_counts.sum() / (6 * lc_counts[i])
        else:
            lc_weights[i] = 0.0

    change_weights = torch.zeros_like(change_counts, dtype=torch.float32)
    for i in range(len(change_counts)):
        if change_counts[i] > 0:
            change_weights[i] = change_counts.sum() / (2 * change_counts[i])
        else:
            change_weights[i] = 0.0

    print("\nLand Cover Weights:", lc_weights.numpy())
    print("Change Weights:", change_weights.numpy())

    return lc_weights, change_weights