"""
train_cnn.py
Train the 3D U-Net with leave-one-G0-out cross-validation and on-the-fly
symmetry augmentation.

Each training cube is converted to a multi-channel volume at native 128³
resolution (pass --downsample to average-pool to 64³, the legacy behaviour
for constrained VRAM), then augmented with the safe z-preserving operations
at load time.  The held-out cube is never augmented.

Usage:
    python train_cnn.py [--safe-only] [--all-ops] [--epochs N] [--save]
                        [--downsample]
"""

import argparse
import json
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from data_loader import (load_all_cubes, cube_to_volumes,
                         get_g0_values, get_feature_cols, add_drop_args, build_drop_set,
                         FEATURE_COLS, LOG_TARGET_COL, compute_data_checksum)
from augmentation import augment_cube, get_symmetry_ops
from cnn_model import UNet3D, count_parameters
from classical_models import compute_metrics, print_results
from model_helpers import (GLOBAL_SEED, _set_seeds, _get_env_info,
                           normalize_channels_inplace, normalize_targets_inplace)


# ── Columns that form the CNN input channels ──────────────────────────────────
# (same as FEATURE_COLS but expressed as volume keys, not flat-table names)
CNN_INPUT_COLS  = FEATURE_COLS                 # 15 channels
CNN_TARGET_COL  = LOG_TARGET_COL               # log10(nH2)
RAW_GRID        = 128                          # native resolution


# ── Dataset ───────────────────────────────────────────────────────────────────

class CubeDataset(Dataset):
    """
    Each item is one (possibly augmented) cube at native 128³ resolution, or
    64³ when pool=True (average-pooled; legacy behaviour for constrained VRAM).
    Returns (input_tensor, target_tensor) of shapes (C, N, N, N) and (1, N, N, N).

    All preprocessing (augmentation, stacking, optional pooling) is done once
    in __init__ so that __getitem__ is a trivial list lookup with no CPU work
    per training step.
    """
    def __init__(self, cube_vols: list[dict[str, np.ndarray]],
                 ops: list[np.ndarray] | None,
                 augment: bool = True,
                 input_cols: list[str] | None = None,
                 pool: bool = False):
        """
        cube_vols  : list of volume dicts (one per source cube)
        ops        : list of 3x3 symmetry matrices; None means no augmentation
        augment    : if True, apply ops; if False, use identity only
        input_cols : feature columns to use (default: CNN_INPUT_COLS)
        pool       : if True, average-pool 128³ -> 64³
        """
        _cols = input_cols if input_cols is not None else CNN_INPUT_COLS
        self.xs: list[torch.Tensor] = []
        self.ys: list[torch.Tensor] = []

        identity   = np.eye(3, dtype=int)
        active_ops = ops if (augment and ops) else [identity]

        for vol in cube_vols:
            for R in active_ops:
                aug = augment_cube(vol, R)

                channels = np.stack([aug[c] for c in _cols], axis=0)  # (C, 128, 128, 128)
                target   = aug[CNN_TARGET_COL][None]                            # (1, 128, 128, 128)

                ch_t  = torch.from_numpy(channels).unsqueeze(0)   # (1, C, 128, 128, 128)
                tgt_t = torch.from_numpy(target).unsqueeze(0)     # (1, 1, 128, 128, 128)

                if pool:
                    ch_t  = F.avg_pool3d(ch_t,  kernel_size=2, stride=2)
                    tgt_t = F.avg_pool3d(tgt_t, kernel_size=2, stride=2)

                self.xs.append(torch.nan_to_num(ch_t.squeeze(0).float()))
                self.ys.append(torch.nan_to_num(tgt_t.squeeze(0).float()))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_fold(train_vols: list[dict],
                   val_vols:   list[dict],
                   ops:        list[np.ndarray],
                   device:     torch.device,
                   epochs:     int = 150,
                   lr:         float = 5e-4,
                   warmup_epochs: int = 10,
                   base_ch:    int = 32,
                   dropout:    float = 0.0,
                   seed:       int = 0,
                   input_cols: list[str] | None = None,
                   pool:       bool = False,
                   ) -> tuple[UNet3D, dict, list[dict]]:

    _set_seeds(GLOBAL_SEED + seed)
    train_ds = CubeDataset(train_vols, ops, augment=True, input_cols=input_cols,
                           pool=pool)
    val_ds   = CubeDataset(val_vols,   ops=None, augment=False,
                           input_cols=input_cols, pool=pool)

    # Fold-safe normalization — same pattern as StandardScaler for the MLP:
    # statistics from training cubes only, then applied to val.  Streaming and
    # in place (a full-dataset stack would be a ~6 GB temporary at 128³).
    # Target normalisation balances the log_nH2 loss across the full dynamic
    # range instead of letting the MSE be dominated by the near-zero-nH2 tail.
    normalize_channels_inplace(train_ds.xs, val_ds.xs)
    y_mean, y_std = normalize_targets_inplace(train_ds.ys, val_ds.ys)

    use_amp  = False   # batch_size=1 → no AMP throughput benefit; unscaled physical inputs risk fp16 overflow
    pin_mem  = device.type == 'cuda'

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=pin_mem)

    _n_ch  = len(input_cols) if input_cols is not None else len(CNN_INPUT_COLS)
    model      = UNet3D(n_channels=_n_ch, base_ch=base_ch, dropout=dropout).to(device)
    opt        = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Linear warmup for `warmup_epochs`, then cosine decay to 0
    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched      = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    loss_fn    = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    best_state    = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=pin_mem), yb.to(device, non_blocking=pin_mem)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(opt)
            scaler_amp.update()
            train_loss += loss.item()
        sched.step()
        train_loss /= len(train_dl)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=pin_mem), yb.to(device, non_blocking=pin_mem)
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_dl)

        history.append({'epoch': epoch,
                        'train_loss': round(float(train_loss), 6),
                        'val_loss':   round(float(val_loss),   6)})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # Restore best weights and compute final metrics
    model.load_state_dict(best_state)
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        for xb, yb in val_dl:
            xb = xb.to(device, non_blocking=pin_mem)
            pred = model(xb).float().cpu().numpy().ravel()
            true = yb.numpy().ravel()
            y_true_all.append(true)
            y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    # Inverse-transform back to log_nH2 space before computing metrics
    y_true = y_true * y_std + y_mean
    y_pred = y_pred * y_std + y_mean
    metrics = compute_metrics(y_true, y_pred)

    return model, metrics, history


# ── Main CV loop ──────────────────────────────────────────────────────────────

def run_cnn_cv(safe_only:    bool = True,
               epochs:       int  = 150,
               base_ch:      int  = 32,
               dropout:      float = 0.0,
               save_models:  bool = False,
               log_path:     str | None = None,
               feature_cols: list[str] | None = None,
               downsample:   bool = False) -> list[dict]:

    _set_seeds(GLOBAL_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)
    n_folds = len(cubes)

    input_cols = feature_cols if feature_cols is not None else CNN_INPUT_COLS

    # Pre-compute all volume dicts (expensive but done once)
    vol_cols = input_cols + [CNN_TARGET_COL]
    avail    = [c for c in vol_cols if c in cubes[0].columns]
    print("Converting cubes to volumes (128³)...")
    all_vols = [cube_to_volumes(df, avail) for df in cubes]

    ops = get_symmetry_ops(safe_only=safe_only)
    print(f"Symmetry ops: {len(ops)}  ({'safe z-preserving' if safe_only else 'full Oh'})")
    n_params = count_parameters(UNet3D(n_channels=len(input_cols), base_ch=base_ch, dropout=dropout))
    print(f"UNet3D params: {n_params:,}  (base_ch={base_ch}, dropout={dropout})")

    print("Computing data checksum...")
    data_checksum = compute_data_checksum()
    env_info = _get_env_info()

    run_config = {
        'timestamp':          datetime.datetime.now().isoformat(timespec='seconds'),
        'global_seed':        GLOBAL_SEED,
        'device':             str(device),
        'epochs':             epochs,
        'safe_only':          safe_only,
        'n_folds':            n_folds,
        'n_params':           n_params,
        'input_cols':         input_cols,
        'target_col':         CNN_TARGET_COL,
        'grid_size':          64 if downsample else 128,
        'raw_grid':           RAW_GRID,
        'n_sym_ops':          len(ops),
        'sym_ops_type':       'z-preserving' if safe_only else 'full_Oh',
        'architecture':       'UNet3D',
        'base_ch':            base_ch,
        'dropout':            dropout,
        'lr':                 5e-4,
        'weight_decay':       1e-5,
        'warmup_epochs':      10,
        'optimizer':          'Adam',
        'scheduler':          'LambdaLR_warmup+cosine',
        'loss':               'MSE',
        'batch_size':         1,
        'cudnn_deterministic': True,
        'env':                env_info,
        'data':               data_checksum,
    }

    fold_metrics  = []
    log_folds     = []
    for fold in range(n_folds):
        print(f"\n[Fold {fold+1}/{n_folds}] Val G0={g0_vals[fold]:.1f}")
        train_vols = [v for i, v in enumerate(all_vols) if i != fold]
        val_vols   = [all_vols[fold]]

        model, metrics, history = train_one_fold(
            train_vols, val_vols, ops, device, epochs=epochs,
            base_ch=base_ch, dropout=dropout,
            input_cols=input_cols, seed=fold, pool=downsample)

        fold_metrics.append(metrics)
        log_folds.append({
            'fold':    fold,
            'g0':      g0_vals[fold],
            'history': history,
            'metrics': {k: float(v) for k, v in metrics.items()},
        })
        print(f"  R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4e}  MAE={metrics['MAE']:.4e}")

        if save_models:
            path = f"cnn_fold{fold}_G0{g0_vals[fold]}.pt"
            torch.save(model.state_dict(), path)
            print(f"  Saved -> {path}")

    print_results("3D U-Net CNN", fold_metrics, g0_vals)

    # ── Write JSON training log ────────────────────────────────────────────────
    if log_path is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = f'results/cnn_training_{ts}.json'

    summary = {}
    for metric in ('R2', 'R2_lin', 'RMSE', 'MAE'):
        vals = [m[metric] for m in fold_metrics]
        summary[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    log = {'run_config': run_config, 'g0_values': g0_vals, 'folds': log_folds, 'summary': summary}
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log saved -> {log_path}")

    return fold_metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-ops',  action='store_true',
                        help='Use all 48 Oh operations (default: 8 safe z-preserving)')
    parser.add_argument('--epochs',   type=int,   default=150,
                        help='Training epochs per fold (paper uses 150 for unet_standard)')
    parser.add_argument('--base-ch',  type=int,   default=32,
                        help='Base channel count: 16=unet_small, 32=unet_standard (default), '
                             '64=unet_large')
    parser.add_argument('--dropout',  type=float, default=0.0,
                        help='Dropout probability (default 0.0; paper shows dropout '
                             'catastrophically degrades extrapolation folds)')
    parser.add_argument('--save',     action='store_true',
                        help='Save best model weights per fold')
    parser.add_argument('--downsample', action='store_true',
                        help='Average-pool volumes 128^3 -> 64^3 (legacy '
                             'behaviour for constrained VRAM; default trains '
                             'at native 128^3)')
    parser.add_argument('--log',      type=str,   default=None,
                        help='Path for JSON training log (default: results/cnn_training_TIMESTAMP.json)')
    add_drop_args(parser)
    args = parser.parse_args()

    feat_cols = get_feature_cols(build_drop_set(args))
    run_cnn_cv(safe_only=not args.all_ops,
               epochs=args.epochs,
               base_ch=args.base_ch,
               dropout=args.dropout,
               save_models=args.save,
               log_path=args.log,
               feature_cols=feat_cols,
               downsample=args.downsample)
