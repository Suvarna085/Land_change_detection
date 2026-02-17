"""
Main Training Script - OPTIMIZED VERSION
- Suppressed TIFF warnings
- Optimized data loading (num_workers=4, persistent_workers)
- Reduced patches per image (20 instead of 50)
- Mixed precision training support
- Validate every 2 epochs
- Better performance overall
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import warnings
import contextlib

# ============================================================
# SUPPRESS WARNINGS (including TIFF warnings)
# ============================================================
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
warnings.filterwarnings('ignore')

# Suppress stderr during cv2 import (Windows compatibility)
if sys.platform == 'win32':
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            import cv2
else:
    import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your modules
from dataloader import HRSCDDataset
from model import SemanticChangeDetectionNet
from trainer import Trainer

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--skip-phase1', action='store_true',
                    help='Skip Phase 1 and go straight to Phase 2 using saved checkpoint')
args = parser.parse_args()


def compute_class_weights_fixed(dataset, num_batches=20):
    """
    FIXED: Compute class weights with better sampling to avoid zeros
    """
    print("Computing class weights (suppressing warnings)...")
    print("Computing class weights on GPU...")

    from torch.utils.data import DataLoader
    
    dataset.set_split('train')
    
    # Use num_workers=0 during weight computation to reduce warning spam
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lc_counts = torch.zeros(6, dtype=torch.float32)
    change_counts = torch.zeros(2, dtype=torch.float32)
    
    print(f"Sampling {num_batches} batches to compute weights...")
    
    # Suppress warnings during batch iteration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
            
            labels = batch['label1']
            changes = batch['change_mask']
            
            for c in range(6):
                lc_counts[c] += (labels == c).sum().float()
            
            change_counts[0] += (changes == 0).sum().float()
            change_counts[1] += (changes == 1).sum().float()
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{num_batches} batches")
    
    lc_counts = lc_counts.cpu()
    change_counts = change_counts.cpu()

    print("\nClass counts:")
    for i in range(6):
        print(f"  Class {i}: {lc_counts[i].item():.0f} pixels")
    print(f"  No Change: {change_counts[0].item():.0f} pixels")
    print(f"  Change: {change_counts[1].item():.0f} pixels")
    
    # Compute weights with smoothing to avoid zeros
    lc_weights = torch.zeros(6, dtype=torch.float32)
    for i in range(6):
        if lc_counts[i] > 0:
            lc_weights[i] = lc_counts.sum() / (6 * lc_counts[i])
        else:
            # If class not seen, give it a default weight
            lc_weights[i] = 1.0
            print(f"  WARNING: Class {i} not found in samples, using default weight 1.0")
    
    # Normalize weights to sum to num_classes
    lc_weights = lc_weights / lc_weights.sum() * 6
    
    # Change weights
    change_weights = torch.zeros(2, dtype=torch.float32)
    for i in range(2):
        if change_counts[i] > 0:
            change_weights[i] = change_counts.sum() / (2 * change_counts[i])
        else:
            change_weights[i] = 1.0
            print(f"  WARNING: Change class {i} not found, using default weight 1.0")
    
    # Normalize
    change_weights = change_weights / change_weights.sum() * 2
    
    print("\nFinal weights (normalized):")
    print("  Land Cover Weights:", lc_weights.numpy())
    print("  Change Weights:", change_weights.numpy())
    
    return lc_weights, change_weights


def main():
    # ============================================================
    # CONFIGURATION - OPTIMIZED FOR SPEED
    # ============================================================
    config = {
        'data_root': './patches',
        'batch_size': 32,  # Increased from 8 (faster training)
        'num_workers': 4,  # Changed from 0 (parallel data loading)
        'patch_size': 256,
        'patches_per_image': 1,  # Reduced from 50 (faster)
        'train_split': 0.8,
        
        # Model
        'num_lc_classes': 6,
        'base_channels': 16,
        
        # Training - Reduced epochs for faster iteration
        'phase1_epochs': 10,  # Reduced from 20
        'phase1_lr': 1e-3,
        'phase2_epochs': 10,  # Reduced from 20
        'phase2_lr': 1e-4,
        
        # Performance
        'use_amp': True,  # Mixed precision training (faster on modern GPUs)
        'validate_every': 2,  # Validate every N epochs (faster)
        
        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print("="*60)
    print("SEMANTIC CHANGE DETECTION - OPTIMIZED TRAINING")
    print("="*60)
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        if config['use_amp']:
            print("Mixed Precision: ENABLED (AMP)")
    else:
        print("\nRunning on CPU (will be slower)")
        config['use_amp'] = False
    print()
    
    # ============================================================
    # 1. LOAD DATASET
    # ============================================================
    print("="*60)
    print("1. LOADING DATASET")
    print("="*60)
    
    try:
        train_dataset = HRSCDDataset(
            data_root=config['data_root'],
            patch_size=config['patch_size'],
            patches_per_image=config['patches_per_image'],
            augment=True,
            train_split=config['train_split']
        )
        train_dataset.set_split('train')
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False
        )
        
        val_dataset = HRSCDDataset(
            data_root=config['data_root'],
            patch_size=config['patch_size'],
            patches_per_image=10,  # Fewer patches for validation
            augment=False,
            train_split=config['train_split']
        )
        val_dataset.set_split('test')
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True if config['device'] == 'cuda' else False
        )
        
        print(f"✓ Dataset loaded")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # 2. COMPUTE CLASS WEIGHTS (FIXED VERSION)
    # ============================================================
    print("\n" + "="*60)
    print("2. COMPUTING CLASS WEIGHTS")
    print("="*60)
    
    try:
        lc_weights, change_weights = compute_class_weights_fixed(
            train_dataset, 
            num_batches=10
        )
    except Exception as e:
        print(f"✗ Error computing weights: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # 3. CREATE MODEL
    # ============================================================
    print("\n" + "="*60)
    print("3. CREATING MODEL")
    print("="*60)
    
    try:
        model = SemanticChangeDetectionNet(
            num_lc_classes=config['num_lc_classes'],
            base_channels=config['base_channels']
        )
        
        print(f"✓ Model created")
        print(f"  Total parameters: {model.get_trainable_params():,}")
        print(f"  Device: {config['device']}")
        
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # 4. CREATE TRAINER
    # ============================================================
    print("\n" + "="*60)
    print("4. INITIALIZING TRAINER")
    print("="*60)
    
    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lc_weights=lc_weights,
            change_weights=change_weights,
            device=config['device'],
            use_amp=config['use_amp'],
            validate_every=config['validate_every']
        )
        
        print("✓ Trainer initialized")
        
    except Exception as e:
        print(f"✗ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============================================================
    # 5. TRAINING
    # ============================================================
    print("\n" + "="*60)
    print("5. STARTING TRAINING")
    print("="*60)
    print("\nNote: TIFF warnings during first batches are normal and harmless.\n")
    
    try:
        # Phase 1: Land Cover Mapping
        # Phase 1: Land Cover Mapping
        if args.skip_phase1:
            print("\n" + "="*60)
            print("PHASE 1: SKIPPED (loading checkpoint)")
            print("="*60)
            ckpt_path = Path('checkpoints/phase1_best.pth')
            if not ckpt_path.exists():
                print("✗ No checkpoint found at checkpoints/phase1_best.pth")
                print("  Run without --skip-phase1 first to train Phase 1")
                return
            epoch, kappa = trainer.load_checkpoint('phase1_best.pth')
            print(f"✓ Loaded Phase 1 checkpoint (epoch {epoch+1}, Kappa {kappa:.4f})")
        else:
            print("\n" + "="*60)
            print("PHASE 1: LAND COVER MAPPING")
            print("="*60)
            trainer.train_phase1_lcm(
                epochs=config['phase1_epochs'],
                lr=config['phase1_lr']
            )
        
        # Phase 2: Change Detection
        print("\n" + "="*60)
        print("PHASE 2: CHANGE DETECTION")
        print("="*60)
        trainer.train_phase2_cd(
            epochs=config['phase2_epochs'],
            lr=config['phase2_lr']
        )
        
        print("\n" + "="*60)
        print("✓ TRAINING COMPLETE!")
        print("="*60)
        print(f"\nCheckpoints saved in: checkpoints/")
        print(f"  - phase1_best.pth (Land Cover Mapping)")
        print(f"  - phase2_best.pth (Full Model)")
        print(f"\nYou can now run inference on test images!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Partial checkpoints may be available in checkpoints/")
    
    except Exception as e:
        print(f"\n\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()