"""
Training script for ChangeFormer
End-to-end single phase training — no sequential phases needed
Uses same dataloader and patches as FC-EF-Res
"""

import os
import sys
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataloader import HRSCDDataset
from changeformer import ChangeFormer


# ── CONFIG ───────────────────────────────────────────────────
config = {
    'data_root':    './patches',
    'batch_size':   32,           # smaller than FC-EF-Res due to Transformer memory
    'num_workers':  4,
    'patch_size':   256,
    'train_split':  0.8,
    'epochs':       20,
    'lr':           6e-5,         # standard ChangeFormer LR
    'decoder_dim':  128,
    'validate_every': 2,
    'use_amp':      True,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu'
}


def combined_loss(pred, target, ce_criterion):
    """CE + Dice — same as FC-EF-Res Phase 2 for fair comparison"""
    ce = ce_criterion(pred, target)
    pred_soft = torch.softmax(pred, dim=1)[:, 1]
    target_f  = (target == 1).float()
    intersection = (pred_soft * target_f).sum()
    dice = 1 - (2 * intersection + 1) / (pred_soft.sum() + target_f.sum() + 1)
    return ce + dice


def compute_metrics(all_preds, all_labels):
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    kappa = cohen_kappa_score(labels, preds)

    pred_c = (preds == 1)
    gt_c   = (labels == 1)
    tp = np.logical_and(pred_c,  gt_c).sum()
    fp = np.logical_and(pred_c, ~gt_c).sum()
    fn = np.logical_and(~pred_c, gt_c).sum()

    precision = tp / (tp + fp + 1e-6)
    recall    = tp / (tp + fn + 1e-6)
    dice      = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    oa        = (preds == labels).mean()

    return {
        'kappa': kappa, 'dice': dice,
        'precision': precision, 'recall': recall, 'oa': oa
    }


def main():
    print("=" * 60)
    print("CHANGEFORMER TRAINING")
    print("=" * 60)
    device = config['device']

    # ── Dataset ──────────────────────────────────────────────
    train_dataset = HRSCDDataset(
        data_root=config['data_root'],
        patch_size=config['patch_size'],
        augment=True,
        train_split=config['train_split']
    )
    train_dataset.set_split('train')

    val_dataset = HRSCDDataset(
        data_root=config['data_root'],
        patch_size=config['patch_size'],
        augment=False,
        train_split=config['train_split']
    )
    val_dataset.set_split('test')

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"Train patches: {len(train_dataset)}")
    print(f"Val patches:   {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # ── Class weights ─────────────────────────────────────────
    # Compute change ratio from a few batches
    print("\nEstimating change class weights...")
    change_counts = torch.zeros(2)
    for i, batch in enumerate(train_loader):
        if i >= 20:
            break
        cm = batch['change_mask']
        change_counts[0] += (cm == 0).sum()
        change_counts[1] += (cm == 1).sum()

    ratio = change_counts[0] / (change_counts[1] + 1e-6)
    change_weights = torch.tensor([1.0, min(ratio, 20.0)])
    print(f"No-change: {change_counts[0]:.0f} | Change: {change_counts[1]:.0f}")
    print(f"Change weight: {change_weights[1]:.2f}x")

    ce_criterion = nn.CrossEntropyLoss(
        weight=change_weights.to(device)
    )

    # ── Model ─────────────────────────────────────────────────
    print("\nCreating ChangeFormer (MiT-B0)...")
    model = ChangeFormer(
        num_classes=2,
        decoder_dim=config['decoder_dim'],
        pretrained=True
    ).to(device)
    print(f"Parameters: {model.get_trainable_params():,}")

    # ── Optimizer ─────────────────────────────────────────────
    # Lower LR for pretrained encoder, higher for fresh decoder
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': config['lr']},
        {'params': model.decoder.parameters(), 'lr': config['lr'] * 10}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )

    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None

    # ── Training loop ─────────────────────────────────────────
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    best_kappa = 0.0

    print(f"\nTraining for {config['epochs']} epochs...")
    print(f"Validating every {config['validate_every']} epochs\n")

    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch_idx, batch in enumerate(pbar):
            try:
                img1      = batch['image1'].to(device, non_blocking=True)
                img2      = batch['image2'].to(device, non_blocking=True)
                change_gt = batch['change_mask'].to(device, non_blocking=True)

                optimizer.zero_grad()

                if config['use_amp']:
                    with torch.cuda.amp.autocast():
                        change_pred = model(img1, img2)
                        loss = combined_loss(change_pred, change_gt, ce_criterion)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    change_pred = model(img1, img2)
                    loss = combined_loss(change_pred, change_gt, ce_criterion)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    continue
                raise e

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validate
        should_validate = (
            (epoch + 1) % config['validate_every'] == 0 or
            (epoch + 1) == config['epochs']
        )

        if should_validate:
            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for batch in val_loader:
                    img1      = batch['image1'].to(device, non_blocking=True)
                    img2      = batch['image2'].to(device, non_blocking=True)
                    change_gt = batch['change_mask'].to(device, non_blocking=True)

                    if config['use_amp']:
                        with torch.cuda.amp.autocast():
                            change_pred = model(img1, img2)
                            loss = combined_loss(change_pred, change_gt, ce_criterion)
                    else:
                        change_pred = model(img1, img2)
                        loss = combined_loss(change_pred, change_gt, ce_criterion)

                    val_loss += loss.item()
                    pred = change_pred.argmax(dim=1)
                    all_preds.append(pred.cpu().numpy().flatten())
                    all_labels.append(change_gt.cpu().numpy().flatten())

            avg_val_loss = val_loss / len(val_loader)
            metrics = compute_metrics(all_preds, all_labels)

            print(f"\nEpoch {epoch+1}/{config['epochs']}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.4f}")
            print(f"  Val Kappa:  {metrics['kappa']:.4f}")
            print(f"  Val Dice:   {metrics['dice']:.4f}")
            print(f"  Val Prec:   {metrics['precision']:.4f}")
            print(f"  Val Rec:    {metrics['recall']:.4f}")
            print(f"  Val OA:     {metrics['oa']:.4f}")

            if metrics['kappa'] > best_kappa:
                best_kappa = metrics['kappa']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'kappa': best_kappa,
                    'metrics': metrics
                }, save_dir / 'changeformer_best.pth')
                print(f"  💾 Saved best model (Kappa {best_kappa:.4f})")
        else:
            print(f"\nEpoch {epoch+1}/{config['epochs']}: "
                  f"Train Loss={avg_train_loss:.4f} (validation skipped)")

    print(f"\n✓ ChangeFormer training complete!")
    print(f"  Best Kappa: {best_kappa:.4f}")
    print(f"  Checkpoint: checkpoints/changeformer_best.pth")


if __name__ == "__main__":
    main()