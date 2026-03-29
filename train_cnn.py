# Usage: python train_cnn.py [--all-ops] [--epochs N] [--save]

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
                         FEATURE_COLS, LOG_TARGET_COL)
from augmentation import augment_cube, get_symmetry_ops
from cnn_model import UNet3D, count_parameters
from classical_models import compute_metrics, print_results

CNN_INPUT_COLS = FEATURE_COLS
CNN_TARGET_COL = LOG_TARGET_COL
GRID_SIZE = 64
RAW_GRID  = 128


class CubeDataset(Dataset):
    """Pre-processes all augmented cubes upfront so __getitem__ is just a list lookup."""
    def __init__(self, cube_vols: list[dict[str, np.ndarray]],
                 ops: list[np.ndarray] | None,
                 augment: bool = True,
                 input_cols: list[str] | None = None):
        _cols = input_cols if input_cols is not None else CNN_INPUT_COLS
        self.xs: list[torch.Tensor] = []
        self.ys: list[torch.Tensor] = []

        identity = np.eye(3, dtype=int)
        active_ops = ops if (augment and ops) else [identity]

        for vol in cube_vols:
            for R in active_ops:
                aug = augment_cube(vol, R)
                channels = np.stack([aug[c] for c in _cols], axis=0)
                target = aug[CNN_TARGET_COL][None]

                ch_t  = torch.from_numpy(channels).unsqueeze(0)
                tgt_t = torch.from_numpy(target).unsqueeze(0)

                # downsample 128^3 -> 64^3
                ch_t  = F.avg_pool3d(ch_t,  kernel_size=2, stride=2).squeeze(0).float()
                tgt_t = F.avg_pool3d(tgt_t, kernel_size=2, stride=2).squeeze(0).float()

                self.xs.append(torch.nan_to_num(ch_t))
                self.ys.append(torch.nan_to_num(tgt_t))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


def train_one_fold(train_vols: list[dict],
                   val_vols:   list[dict],
                   ops:        list[np.ndarray],
                   device:     torch.device,
                   epochs:     int = 50,
                   lr:         float = 5e-4,
                   warmup_epochs: int = 10,
                   input_cols: list[str] | None = None) -> tuple[UNet3D, dict, list[dict]]:

    train_ds = CubeDataset(train_vols, ops, augment=True,  input_cols=input_cols)
    val_ds   = CubeDataset(val_vols,   ops=None, augment=False, input_cols=input_cols)

    # normalize per-channel using training stats only
    all_x   = torch.stack(train_ds.xs)
    ch_mean = all_x.mean(dim=(0, 2, 3, 4), keepdim=True).squeeze(0)
    ch_std  = all_x.std( dim=(0, 2, 3, 4), keepdim=True).squeeze(0).clamp(min=1e-6)
    train_ds.xs = [(x - ch_mean) / ch_std for x in train_ds.xs]
    val_ds.xs   = [(x - ch_mean) / ch_std for x in val_ds.xs]

    # normalize target -- without this MSE is dominated by low-nH2 cells
    all_y  = torch.stack(train_ds.ys)
    y_mean = all_y.mean()
    y_std  = all_y.std().clamp(min=1e-6)
    train_ds.ys = [(y - y_mean) / y_std for y in train_ds.ys]
    val_ds.ys   = [(y - y_mean) / y_std for y in val_ds.ys]

    # AMP disabled: batch_size=1 gives no throughput benefit, and physical fields
    # can exceed fp16 max (65504)
    use_amp = False
    pin_mem = device.type == 'cuda'

    n_ch = len(input_cols) if input_cols else len(CNN_INPUT_COLS)
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,  num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False, num_workers=0, pin_memory=pin_mem)

    model   = UNet3D(n_channels=n_ch, base_ch=16, dropout=0.1).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    best_val_loss = float('inf')
    best_state    = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
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
    y_std_np  = y_std.item()
    y_mean_np = y_mean.item()
    y_true = y_true * y_std_np + y_mean_np
    y_pred = y_pred * y_std_np + y_mean_np
    metrics = compute_metrics(y_true, y_pred)

    return model, metrics, history


def run_cnn_cv(safe_only: bool = True,
               epochs: int = 150,
               save_models: bool = False,
               log_path: str | None = None,
               feature_cols: list[str] | None = None) -> list[dict]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)
    n_folds = len(cubes)

    input_cols = feature_cols if feature_cols is not None else CNN_INPUT_COLS
    vol_cols   = input_cols + [CNN_TARGET_COL]
    avail      = [c for c in vol_cols if c in cubes[0].columns]

    print("Converting cubes to volumes (128^3)...")
    all_vols = [cube_to_volumes(df, avail) for df in cubes]

    ops = get_symmetry_ops(safe_only=safe_only)
    print(f"Symmetry ops: {len(ops)}  ({'safe z-preserving' if safe_only else 'full Oh'})")
    n_params = count_parameters(UNet3D(n_channels=len(input_cols)))
    print(f"UNet3D params: {n_params:,}")

    run_config = {
        'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
        'device':     str(device),
        'epochs':     epochs,
        'safe_only':  safe_only,
        'n_folds':    n_folds,
        'n_params':   n_params,
        'input_cols': input_cols,
        'target_col': CNN_TARGET_COL,
        'grid_size':  GRID_SIZE,
        'n_sym_ops':  len(ops),
    }

    fold_metrics = []
    log_folds    = []
    for fold in range(n_folds):
        print(f"\n[Fold {fold+1}/{n_folds}] Val G0={g0_vals[fold]:.1f}")
        train_vols = [v for i, v in enumerate(all_vols) if i != fold]
        val_vols   = [all_vols[fold]]

        model, metrics, history = train_one_fold(
            train_vols, val_vols, ops, device, epochs=epochs, input_cols=input_cols)

        fold_metrics.append(metrics)
        log_folds.append({
            'fold':    fold,
            'g0':      g0_vals[fold],
            'history': history,
            'metrics': {k: float(v) for k, v in metrics.items()},
        })
        print(f"  R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4e}  MAE={metrics['MAE']:.4e}")

        if save_models:
            path = f"cnn_fold{fold}_G0{g0_vals[fold]}.pt"
            torch.save(model.state_dict(), path)
            print(f"  Saved {path}")

    print_results("3D U-Net CNN", fold_metrics, g0_vals)

    if log_path is None:
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = f'cnn_training_{ts}.json'

    summary = {}
    for metric in ('R2', 'R2_lin', 'RMSE', 'MAE'):
        vals = [m[metric] for m in fold_metrics]
        summary[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    log = {'run_config': run_config, 'g0_values': g0_vals, 'folds': log_folds, 'summary': summary}
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nTraining log -> {log_path}")

    return fold_metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all-ops', action='store_true',
                        help='Use all 48 Oh operations (default: 8 safe z-preserving)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save',   action='store_true', help='Save best model weights per fold')
    parser.add_argument('--log',    type=str, default=None)
    add_drop_args(parser)
    args = parser.parse_args()

    feat_cols = get_feature_cols(build_drop_set(args))
    run_cnn_cv(safe_only=not args.all_ops,
               epochs=args.epochs,
               save_models=args.save,
               log_path=args.log,
               feature_cols=feat_cols)
