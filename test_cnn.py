"""
test_cnn.py
===========
Evaluate the current UNet3D (ResConvBlock-based) under leave-one-G0-out CV.
Runs only CNN variants — use compare_architectures.py for full model comparison.

Variants
--------
  unet_baseline  base_ch=32, dropout=0.0, no warmup  (old hyperparams, new arch)
  unet_residual  base_ch=16, dropout=0.1, warmup=10  (colleague's full config)
  unet_large     base_ch=64, dropout=0.1, warmup=10  (capacity upper bound)

Usage
-----
  python test_cnn.py
  python test_cnn.py --variants unet_residual unet_baseline
  python test_cnn.py --epochs 100 --no-fh2
  python test_cnn.py --all-ops
"""

import argparse
import datetime
import json
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torch.nn.functional as F
from torch.utils.data import Dataset

from data_loader import (load_all_cubes, cube_to_volumes, get_g0_values,
                         get_feature_cols, add_drop_args, build_drop_set,
                         FEATURE_COLS, LOG_TARGET_COL)
from augmentation import augment_cube
from augmentation   import get_symmetry_ops
from cnn_model      import UNet3D, count_parameters
from classical_models import compute_metrics, print_results
from model_helpers  import normalize_channels_inplace, normalize_targets_inplace


# ── Dataset ───────────────────────────────────────────────────────────────────

class CubeDataset(Dataset):
    """Augment and stack 128³ cubes at construction time; pool=True
    average-pools to 64³ (legacy behaviour for constrained VRAM)."""
    def __init__(self, cube_vols: list[dict],
                 ops: list[np.ndarray] | None,
                 augment: bool = True,
                 input_cols: list[str] | None = None,
                 pool: bool = False):
        _cols = input_cols if input_cols is not None else FEATURE_COLS
        self.xs: list[torch.Tensor] = []
        self.ys: list[torch.Tensor] = []
        identity   = np.eye(3, dtype=int)
        active_ops = ops if (augment and ops) else [identity]
        for vol in cube_vols:
            for R in active_ops:
                aug     = augment_cube(vol, R)
                ch_t    = torch.from_numpy(
                              np.stack([aug[c] for c in _cols], axis=0)
                          ).unsqueeze(0).float()                          # (1, C, 128, 128, 128)
                tgt_t   = torch.from_numpy(
                              aug[LOG_TARGET_COL][None]
                          ).unsqueeze(0).float()                          # (1, 1, 128, 128, 128)
                if pool:
                    ch_t  = F.avg_pool3d(ch_t,  2, 2)
                    tgt_t = F.avg_pool3d(tgt_t, 2, 2)
                self.xs.append(torch.nan_to_num(ch_t.squeeze(0)))
                self.ys.append(torch.nan_to_num(tgt_t.squeeze(0)))

    def __len__(self) -> int:               return len(self.xs)
    def __getitem__(self, i: int):          return self.xs[i], self.ys[i]


# ── Variant catalogue ──────────────────────────────────────────────────────────

CNN_VARIANTS: dict[str, dict] = {
    # Old hyperparams with new ResConvBlock architecture — control condition
    'unet_baseline': {'base_ch': 32, 'dropout': 0.0, 'warmup_epochs':  0, 'lr': 5e-4},
    # Colleague's full recommendation: smaller model, dropout, warmup
    'unet_residual': {'base_ch': 16, 'dropout': 0.1, 'warmup_epochs': 10, 'lr': 5e-4},
    # Capacity upper bound
    'unet_large':    {'base_ch': 64, 'dropout': 0.1, 'warmup_epochs': 10, 'lr': 5e-4},
}

CNN_TARGET_COL = LOG_TARGET_COL


# ── Single-fold training ───────────────────────────────────────────────────────

def _run_fold(train_vols: list[dict], val_vols: list[dict],
              ops: list[np.ndarray], device: torch.device,
              epochs: int, lr: float, base_ch: int,
              input_cols: list[str],
              dropout: float = 0.0,
              warmup_epochs: int = 0,
              seed: int = 0,
              pool: bool = False) -> tuple[dict, list[dict]]:
    """Train one CV fold, return (metrics, per-epoch history)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CubeDataset(train_vols, ops,      augment=True,
                           input_cols=input_cols, pool=pool)
    val_ds   = CubeDataset(val_vols,   ops=None, augment=False,
                           input_cols=input_cols, pool=pool)

    # Fold-safe normalisation (training stats only), streaming + in place —
    # a full-dataset stack would be a ~6 GB temporary at 128³
    normalize_channels_inplace(train_ds.xs, val_ds.xs)
    y_mean, y_std = normalize_targets_inplace(train_ds.ys, val_ds.ys)

    pin_mem  = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=pin_mem)

    model   = UNet3D(n_channels=len(input_cols), base_ch=base_ch, dropout=dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Linear warmup then cosine decay; warmup_epochs=0 -> pure cosine
    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched   = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    loss_fn = nn.MSELoss()

    best_val   = float('inf')
    best_state: dict | None = None
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device, non_blocking=pin_mem), yb.to(device, non_blocking=pin_mem)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_loss += loss.item()
        sched.step()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device, non_blocking=pin_mem), yb.to(device, non_blocking=pin_mem)
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_dl)

        history.append({'epoch': epoch,
                        'train_loss': round(float(train_loss), 6),
                        'val_loss':   round(float(val_loss),   6)})

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}  val={val_loss:.4f}")

    # Restore best checkpoint, evaluate in original log_nH2 space
    model.load_state_dict(best_state)
    model.eval()
    y_true_parts, y_pred_parts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            y_pred_parts.append(model(xb.to(device)).float().cpu().numpy().ravel())
            y_true_parts.append(yb.numpy().ravel())
    y_true = np.concatenate(y_true_parts) * y_std + y_mean
    y_pred = np.concatenate(y_pred_parts) * y_std + y_mean

    return compute_metrics(y_true, y_pred), history


# ── Full CV for one variant ────────────────────────────────────────────────────

def run_variant(name: str, config: dict,
                all_vols: list[dict], g0_vals: list[float],
                ops: list[np.ndarray], device: torch.device,
                epochs: int, input_cols: list[str],
                pool: bool = False) -> dict:
    base_ch   = config['base_ch']
    dropout   = config.get('dropout', 0.0)
    warmup_ep = config.get('warmup_epochs', 0)
    lr        = config.get('lr', 5e-4)
    n_params  = count_parameters(UNet3D(n_channels=len(input_cols),
                                        base_ch=base_ch, dropout=dropout))
    print(f"\n{'='*62}")
    print(f"  {name}  base_ch={base_ch}  dropout={dropout}  "
          f"warmup={warmup_ep}  lr={lr}  params={n_params:,}  "
          f"grid={64 if pool else 128}^3")
    print(f"{'='*62}")

    fold_metrics: list[dict] = []
    log_folds:   list[dict] = []

    for fold in range(len(all_vols)):
        print(f"\n  [Fold {fold+1}/{len(all_vols)}]  Val G0={g0_vals[fold]:.1f}")
        train_vols = [v for i, v in enumerate(all_vols) if i != fold]
        val_vols   = [all_vols[fold]]

        metrics, history = _run_fold(
            train_vols, val_vols, ops, device, epochs, lr, base_ch,
            input_cols=input_cols, dropout=dropout,
            warmup_epochs=warmup_ep, seed=fold, pool=pool)

        fold_metrics.append(metrics)
        log_folds.append({
            'fold':    fold,
            'g0':      g0_vals[fold],
            'metrics': {k: float(v) for k, v in metrics.items()},
            'history': history,
        })
        print(f"    R2={metrics['R2']:.4f}  R2_lin={metrics['R2_lin']:.4f}  "
              f"RMSE={metrics['RMSE']:.4e}  MAE={metrics['MAE']:.4e}")

    print_results(name, fold_metrics, g0_vals)

    summary = {}
    for metric in ('R2', 'R2_lin', 'RMSE', 'MAE'):
        vals = [m[metric] for m in fold_metrics]
        summary[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    return {
        'config':  {**config, 'n_params': n_params},
        'folds':   log_folds,
        'summary': summary,
    }


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    all_names = list(CNN_VARIANTS)
    parser = argparse.ArgumentParser(
        description='Test current UNet3D (ResConvBlock) variants only.')
    parser.add_argument('--epochs',   type=int, default=150,
                        help='Training epochs per fold (default: 150)')
    parser.add_argument('--all-ops',  action='store_true',
                        help='Use all 48 Oh symmetry ops (default: 8 safe z-preserving)')
    parser.add_argument('--variants', nargs='+', choices=all_names, default=all_names,
                        metavar='V',
                        help=f'Variants to run. Choices: {all_names}')
    parser.add_argument('--log', type=str, default=None,
                        help='JSON output path (default: results/cnn_test_TIMESTAMP.json)')
    parser.add_argument('--downsample', action='store_true',
                        help='Average-pool volumes 128^3 -> 64^3 (legacy '
                             'behaviour for constrained VRAM; default trains '
                             'at native 128^3)')
    add_drop_args(parser)
    args = parser.parse_args()

    input_cols = get_feature_cols(build_drop_set(args))
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Device:    {device}")
    print(f"Epochs:    {args.epochs}")
    print(f"Variants:  {args.variants}")
    print(f"Features:  {len(input_cols)}  ({', '.join(input_cols)})")

    print("\nLoading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    vol_cols = list(dict.fromkeys(input_cols + [CNN_TARGET_COL]))
    avail    = [c for c in vol_cols if c in cubes[0].columns]
    print("Converting cubes to volumes (128^3)...")
    all_vols = [cube_to_volumes(df, avail) for df in cubes]

    ops = get_symmetry_ops(safe_only=not args.all_ops)
    print(f"Symmetry ops: {len(ops)}  ({'safe z-preserving' if not args.all_ops else 'full Oh'})")

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = args.log or f'results/cnn_test_{ts}.json'

    results: dict[str, dict] = {}
    for name in args.variants:
        results[name] = run_variant(
            name, CNN_VARIANTS[name], all_vols, g0_vals, ops, device,
            epochs=args.epochs, input_cols=input_cols,
            pool=args.downsample)

    # Final comparison table
    print(f"\n{'Variant':<20}  {'R2 mean':>8}  {'+-':>6}  {'R2_lin mean':>11}  {'MAE mean':>9}")
    print('-' * 62)
    for name, res in results.items():
        s = res['summary']
        print(f"{name:<20}  {s['R2']['mean']:>8.4f}  "
              f"{s['R2']['std']:>6.4f}  "
              f"{s['R2_lin']['mean']:>11.4f}  "
              f"{s['MAE']['mean']:>9.4e}")

    log = {
        'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
        'device':     str(device),
        'epochs':     args.epochs,
        'grid_size':  64 if args.downsample else 128,
        'input_cols': input_cols,
        'g0_values':  g0_vals,
        'variants':   results,
    }
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved -> {log_path}")


if __name__ == '__main__':
    main()
