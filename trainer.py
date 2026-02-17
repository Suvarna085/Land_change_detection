"""
Training Script for Semantic Change Detection - OPTIMIZED
Strategy 4.2: Sequential Training

OPTIMIZATIONS:
- Mixed precision training (AMP) for 1.5-2x speedup
- Configurable validation frequency
- Better progress reporting
- Combined CE + Dice loss for better precision/recall balance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path


class Trainer:
    def __init__(self, model, train_loader, val_loader,
                 lc_weights, change_weights, device='cuda',
                 use_amp=True, validate_every=1):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.validate_every = validate_every

        # Mixed precision training
        self.use_amp = use_amp and device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if self.use_amp:
            print("✓ Mixed Precision Training (AMP) enabled")

        # Loss functions with class weights
        self.lc_criterion = nn.CrossEntropyLoss(weight=lc_weights.to(device))
        self.cd_criterion = nn.CrossEntropyLoss(weight=change_weights.to(device))

        self.best_val_kappa = 0.0
        self.save_dir = Path('checkpoints')
        self.save_dir.mkdir(exist_ok=True)

    def combined_loss(self, pred, target):
        """
        CrossEntropy + Dice loss to balance precision and recall.
        CE handles class weights, Dice directly penalises false positives.
        """
        ce_loss = self.cd_criterion(pred, target)

        # Dice loss on the change class (class 1)
        pred_soft = torch.softmax(pred, dim=1)[:, 1]   # probability of change
        target_float = (target == 1).float()

        intersection = (pred_soft * target_float).sum()
        dice_loss = 1 - (2 * intersection + 1) / (
            pred_soft.sum() + target_float.sum() + 1
        )

        return ce_loss + dice_loss

    def train_phase1_lcm(self, epochs=20, lr=1e-3):
        """Phase 1: Train Land Cover Mapping only"""
        print("\nOptimizer: Adam")
        print(f"Learning Rate: {lr}")
        print(f"Validation: Every {self.validate_every} epoch(s)")

        optimizer = optim.Adam([
            {'params': self.model.encoder_lcm.parameters()},
            {'params': self.model.decoder_lcm.parameters()}
        ], lr=lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    img1 = batch['image1'].to(self.device, non_blocking=True)
                    img2 = batch['image2'].to(self.device, non_blocking=True)
                    label1 = batch['label1'].to(self.device, non_blocking=True)
                    label2 = batch['label2'].to(self.device, non_blocking=True)

                    optimizer.zero_grad()

                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            lcm1 = self.model.forward_lcm_only(img1)
                            lcm2 = self.model.forward_lcm_only(img2)
                            loss1 = self.lc_criterion(lcm1, label1)
                            loss2 = self.lc_criterion(lcm2, label2)
                            loss = (loss1 + loss2) / 2

                        if torch.isnan(loss):
                            print(f"\n⚠️ NaN loss at batch {batch_idx}, skipping...")
                            continue

                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()

                    else:
                        lcm1 = self.model.forward_lcm_only(img1)
                        lcm2 = self.model.forward_lcm_only(img2)
                        loss1 = self.lc_criterion(lcm1, label1)
                        loss2 = self.lc_criterion(lcm2, label2)
                        loss = (loss1 + loss2) / 2

                        if torch.isnan(loss):
                            print(f"\n⚠️ NaN loss at batch {batch_idx}, skipping...")
                            continue

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️ OOM at batch {batch_idx}, clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            avg_train_loss = train_loss / len(self.train_loader)

            should_validate = (
                (epoch + 1) % self.validate_every == 0 or
                (epoch + 1) == epochs
            )

            if should_validate:
                val_loss, val_acc, val_kappa = self.validate_lcm()

                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Val Acc:    {val_acc:.4f}")
                print(f"  Val Kappa:  {val_kappa:.4f}")

                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"  LR: {old_lr:.6f} → {new_lr:.6f}")

                if val_kappa > self.best_val_kappa:
                    self.best_val_kappa = val_kappa
                    self.save_checkpoint('phase1_best.pth', epoch, val_kappa)
                    print(f"  ★ New best Kappa: {val_kappa:.4f}")
            else:
                print(f"\nEpoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f} (validation skipped)")

        print(f"\n✓ Phase 1 complete! Best Kappa: {self.best_val_kappa:.4f}")

    def train_phase2_cd(self, epochs=20, lr=1e-4):
        """Phase 2: Freeze LCM, train Change Detection only"""
        print("\nOptimizer: Adam")
        print(f"Learning Rate: {lr}")
        print(f"Validation: Every {self.validate_every} epoch(s)")

        # Load best Phase 1 checkpoint before starting Phase 2
        phase1_ckpt = self.save_dir / 'phase1_best.pth'
        if phase1_ckpt.exists():
            self.load_checkpoint('phase1_best.pth')
            print("✓ Loaded best Phase 1 checkpoint for Phase 2")
        else:
            print("⚠️ No Phase 1 checkpoint found, using current weights")

        # Freeze LCM parameters
        self.model.freeze_lcm()

        # Optimizer for CD parameters only
        optimizer = optim.Adam([
            {'params': self.model.cd_proj.parameters()},
            {'params': self.model.decoder_cd.parameters()}
        ], lr=lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_cd_kappa = 0.0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    img1 = batch['image1'].to(self.device, non_blocking=True)
                    img2 = batch['image2'].to(self.device, non_blocking=True)
                    change_gt = batch['change_mask'].to(self.device, non_blocking=True)

                    optimizer.zero_grad()

                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            _, _, change_pred = self.model(img1, img2, return_all=True)
                            loss = self.combined_loss(change_pred, change_gt)

                        if torch.isnan(loss):
                            print(f"\n⚠️ NaN loss at batch {batch_idx}, skipping...")
                            continue

                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(optimizer)
                        self.scaler.update()

                    else:
                        _, _, change_pred = self.model(img1, img2, return_all=True)
                        loss = self.combined_loss(change_pred, change_gt)

                        if torch.isnan(loss):
                            print(f"\n⚠️ NaN loss at batch {batch_idx}, skipping...")
                            continue

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"\n⚠️ OOM at batch {batch_idx}, clearing cache...")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e

            avg_train_loss = train_loss / len(self.train_loader)

            should_validate = (
                (epoch + 1) % self.validate_every == 0 or
                (epoch + 1) == epochs
            )

            if should_validate:
                val_loss, val_metrics = self.validate_cd()

                print(f"\nEpoch {epoch+1}/{epochs}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss:   {val_loss:.4f}")
                print(f"  Val Kappa:  {val_metrics['kappa']:.4f}")
                print(f"  Val Dice:   {val_metrics['dice']:.4f}")
                print(f"  Val Prec:   {val_metrics['precision']:.4f}")
                print(f"  Val Rec:    {val_metrics['recall']:.4f}")

                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"  LR: {old_lr:.6f} → {new_lr:.6f}")

                if val_metrics['kappa'] > best_cd_kappa:
                    best_cd_kappa = val_metrics['kappa']
                    self.save_checkpoint('phase2_best.pth', epoch, val_metrics['kappa'])
                    print(f"  ★ New best Kappa: {val_metrics['kappa']:.4f}")
            else:
                print(f"\nEpoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f} (validation skipped)")

        print(f"\n✓ Phase 2 complete! Best CD Kappa: {best_cd_kappa:.4f}")

    def validate_lcm(self):
        """Validate Land Cover Mapping"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                img1 = batch['image1'].to(self.device, non_blocking=True)
                label1 = batch['label1'].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        lcm1 = self.model.forward_lcm_only(img1)
                        loss = self.lc_criterion(lcm1, label1)
                else:
                    lcm1 = self.model.forward_lcm_only(img1)
                    loss = self.lc_criterion(lcm1, label1)

                val_loss += loss.item()

                pred = lcm1.argmax(dim=1)
                correct += (pred == label1).sum().item()
                total += label1.numel()

                all_preds.append(pred.cpu().numpy().flatten())
                all_labels.append(label1.cpu().numpy().flatten())

        avg_loss = val_loss / len(self.val_loader)
        accuracy = correct / total

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        kappa = self.compute_kappa(all_preds, all_labels)

        return avg_loss, accuracy, kappa

    def validate_cd(self):
        """Validate Change Detection"""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                img1 = batch['image1'].to(self.device, non_blocking=True)
                img2 = batch['image2'].to(self.device, non_blocking=True)
                change_gt = batch['change_mask'].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        _, _, change_pred = self.model(img1, img2, return_all=True)
                        loss = self.combined_loss(change_pred, change_gt)
                else:
                    _, _, change_pred = self.model(img1, img2, return_all=True)
                    loss = self.combined_loss(change_pred, change_gt)

                val_loss += loss.item()

                pred = change_pred.argmax(dim=1)
                all_preds.append(pred.cpu().numpy().flatten())
                all_labels.append(change_gt.cpu().numpy().flatten())

        avg_loss = val_loss / len(self.val_loader)

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        metrics = {
            'kappa':     self.compute_kappa(all_preds, all_labels),
            'dice':      self.compute_dice(all_preds, all_labels),
            'precision': self.compute_precision(all_preds, all_labels),
            'recall':    self.compute_recall(all_preds, all_labels)
        }

        return avg_loss, metrics

    def compute_kappa(self, pred, gt):
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(gt, pred)

    def compute_dice(self, pred, gt):
        pred_change = (pred == 1)
        gt_change = (gt == 1)
        intersection = np.logical_and(pred_change, gt_change).sum()
        return (2 * intersection) / (pred_change.sum() + gt_change.sum() + 1e-6)

    def compute_precision(self, pred, gt):
        pred_change = (pred == 1)
        gt_change = (gt == 1)
        tp = np.logical_and(pred_change, gt_change).sum()
        fp = np.logical_and(pred_change, ~gt_change).sum()
        return tp / (tp + fp + 1e-6)

    def compute_recall(self, pred, gt):
        pred_change = (pred == 1)
        gt_change = (gt == 1)
        tp = np.logical_and(pred_change, gt_change).sum()
        fn = np.logical_and(~pred_change, gt_change).sum()
        return tp / (tp + fn + 1e-6)

    def save_checkpoint(self, filename, epoch, kappa):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'kappa': kappa
        }
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        print(f"  💾 Checkpoint saved: {path}")

    def load_checkpoint(self, filename):
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✓ Checkpoint loaded: {path}")
        return checkpoint['epoch'], checkpoint['kappa']