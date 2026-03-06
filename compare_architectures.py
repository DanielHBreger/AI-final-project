"""
compare_architectures.py
Systematic comparison of architectural variants for XGBoost, MLP, and CNN under
leave-one-G0-out cross-validation (7 folds, one held-out cube per fold).

Architecture selection rationale
---------------------------------
XGBoost (3 variants)
  The structural knobs are tree depth (controls feature-interaction order) and
  ensemble density (n_estimators x learning_rate).  All other hyperparameters
  are fixed to isolate these two axes.

  xgb_shallow  depth=4, 600 trees, lr=0.05
    Bias-variance shift toward variance reduction.  More, weaker trees reduce
    memorisation.  Tests whether low-order feature splits suffice.

  xgb_standard  depth=6, 400 trees, lr=0.10  [BASELINE — matches run_xgboost]
    Standard XGBoost sweet spot for tabular physics data.

  xgb_deep  depth=8, 300 trees, lr=0.10
    Tests whether 4th/5th-order feature interactions (e.g. nH x T x G0 x ext)
    improve prediction.  Risk: may memorise training G0 values.

  Rejected: DART (dropout on trees) — analogous to neural dropout which was
  shown experimentally (CNN Run 2) to hurt OOD extrapolation.

MLP (3 variants)
  With ~15 M training rows per fold the MLP is data-rich; underfitting is the
  primary concern, not overfitting.

  mlp_standard  [256, 256, 128, 64]  [BASELINE — matches run_mlp]
    Tapering width forces progressive abstraction.

  mlp_wide  [512, 512, 256, 128]
    Double width throughout.  Tests raw capacity gain.

  mlp_residual  256 x 4 residual blocks
    Principled for smooth physics regression: residuals let the network learn
    corrections delta(x) on top of a linear projection rather than a full
    non-linear remapping.  Constant width preserves information flow.
    Prevents gradient vanishing in deeper networks.

  Rejected: SELU (sensitive to init), LayerNorm (slower than BN at batch_size
  262144), Dropout (hurts OOD extrapolation per CNN experiment).

CNN (3 variants)
  The only spatial model.  The main architectural axis is base channel count.

  A 4th encoder level is explicitly rejected: it would compress 64^3 -> 4^3
  (16x per axis), destroying the fine-grained field gradients needed for
  dense prediction.  The 3-level encoder giving an 8^3 bottleneck is correct
  for 64^3 inputs.

  unet_small   base_ch=16  ~1.5 M params
    Tests whether reduced capacity closes the train/val gap (training loss
    collapses to ~0.005 with base_ch=32, suggesting over-parameterisation).

  unet_standard  base_ch=32  ~5.8 M params  [BASELINE — current best, R2=0.803]

  unet_large  base_ch=64  ~23 M params
    Upper bound.  Predicted to NOT improve: 480 000 : 1 param-to-sample ratio.
    Included to confirm the over-parameterisation hypothesis.

Usage
-----
  # Fast smoke test (XGBoost + MLP, 5 MLP epochs):
  python compare_architectures.py --skip-cnn --mlp-epochs 5

  # Full comparison with 50 CNN epochs (~30 min on GPU):
  python compare_architectures.py --cnn-epochs 50

  # Production-quality CNN comparison (matches best known config):
  python compare_architectures.py --cnn-epochs 150
"""

import argparse
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from data_loader import (load_all_cubes, cube_to_volumes, get_X_y,
                          get_g0_values, FEATURE_COLS, LOG_TARGET_COL)
from classical_models import compute_metrics
from cnn_model import UNet3D, count_parameters
from augmentation import augment_cube, get_symmetry_ops


# ── Shared constants ──────────────────────────────────────────────────────────

CNN_INPUT_COLS = FEATURE_COLS   # 14 physical field channels
CNN_TARGET_COL = LOG_TARGET_COL # log10(fh2)


# ── XGBoost variant configs ───────────────────────────────────────────────────

XGB_VARIANTS: dict[str, dict] = {
    # Shallow trees — more diverse ensemble, lower-order interactions
    'xgb_shallow': dict(
        max_depth=4, n_estimators=600, learning_rate=0.05,
        subsample=0.3, colsample_bytree=0.8, tree_method='hist',
        random_state=42, verbosity=0,
    ),
    # Standard depth — baseline matching run_xgboost in classical_models.py
    'xgb_standard': dict(
        max_depth=6, n_estimators=400, learning_rate=0.10,
        subsample=0.3, colsample_bytree=0.8, tree_method='hist',
        random_state=42, verbosity=0,
    ),
    # Deep trees — captures higher-order feature interactions up to depth-8 paths
    'xgb_deep': dict(
        max_depth=8, n_estimators=300, learning_rate=0.10,
        subsample=0.3, colsample_bytree=0.8, tree_method='hist',
        random_state=42, verbosity=0,
    ),
}


# ── MLP architecture classes ──────────────────────────────────────────────────

class FlexMLP(nn.Module):
    """Standard MLP with configurable hidden layer widths.
    Architecture: Linear -> BN -> ReLU -> ... -> Linear(1).
    """
    def __init__(self, in_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class _ResidualBlock(nn.Module):
    """Pre-activation residual block: BN->ReLU->Linear->BN->ReLU->Linear + skip.
    Constant-width only (no projection required for the identity shortcut).
    """
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLP(nn.Module):
    """Residual MLP: input projection -> N residual blocks -> Linear(1).
    Constant internal width (hidden_dim) preserves information throughout.
    Learns corrections on top of a linear projection rather than a full
    non-linear remapping — well-suited for smooth physics-based regression.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 256, n_blocks: int = 4):
        super().__init__()
        self.proj   = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.Sequential(*[_ResidualBlock(hidden_dim) for _ in range(n_blocks)])
        self.out    = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.proj(x))).squeeze(-1)


# ── MLP variant configs ───────────────────────────────────────────────────────

MLP_VARIANTS: dict[str, dict] = {
    # Standard tapering MLP — matches run_mlp baseline
    'mlp_standard': {'arch': 'flex',     'hidden_dims': [256, 256, 128, 64]},
    # Wider MLP — double capacity at each layer
    'mlp_wide':     {'arch': 'flex',     'hidden_dims': [512, 512, 256, 128]},
    # Residual MLP — principled for smooth physics regression
    'mlp_residual': {'arch': 'residual', 'hidden_dim': 256, 'n_blocks': 4},
}


# ── CNN variant configs ───────────────────────────────────────────────────────

CNN_VARIANTS: dict[str, dict] = {
    # Small — tests whether reduced capacity closes the train/val gap
    'unet_small':    {'base_ch': 16},
    # Standard — current best (R2=0.803 mean, Run 3)
    'unet_standard': {'base_ch': 32},
    # Large — upper bound; expected to confirm over-parameterisation hypothesis
    'unet_large':    {'base_ch': 64},
}


# ── Shared CubeDataset (mirrors train_cnn.CubeDataset exactly) ─────────────────

class CubeDataset(Dataset):
    """Pre-computes all augmented, pooled, and normalised tensors at init time."""
    def __init__(self, cube_vols: list[dict],
                 ops: list[np.ndarray] | None,
                 augment: bool = True):
        self.xs: list[torch.Tensor] = []
        self.ys: list[torch.Tensor] = []
        identity   = np.eye(3, dtype=int)
        active_ops = ops if (augment and ops) else [identity]
        for vol in cube_vols:
            for R in active_ops:
                aug      = augment_cube(vol, R)
                channels = np.stack([aug[c] for c in CNN_INPUT_COLS], axis=0)
                target   = aug[CNN_TARGET_COL][None]
                ch_t     = torch.from_numpy(channels).unsqueeze(0)
                tgt_t    = torch.from_numpy(target).unsqueeze(0)
                ch_t     = F.avg_pool3d(ch_t,  kernel_size=2, stride=2).squeeze(0).float()
                tgt_t    = F.avg_pool3d(tgt_t, kernel_size=2, stride=2).squeeze(0).float()
                self.xs.append(torch.nan_to_num(ch_t))
                self.ys.append(torch.nan_to_num(tgt_t))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


# ── Training helpers ──────────────────────────────────────────────────────────

def run_xgb_cv(variant_name: str,
               config: dict,
               X: np.ndarray,
               y: np.ndarray,
               fold_labels: np.ndarray,
               g0_values: list[float]) -> list[dict]:
    """7-fold leave-one-G0-out CV for one XGBoost config."""
    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg  = {**config, 'device': _dev}
    fold_metrics: list[dict] = []
    for fold in range(len(g0_values)):
        mask   = fold_labels != fold
        X_tr, y_tr = X[mask],  y[mask]
        X_va, y_va = X[~mask], y[~mask]
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)
        model = xgb.XGBRegressor(**cfg)
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
        fold_metrics.append(compute_metrics(y_va, model.predict(X_va_s)))
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")
    return fold_metrics


def _build_mlp(config: dict, in_dim: int) -> nn.Module:
    """Dispatch to the appropriate MLP class based on config['arch']."""
    if config['arch'] == 'flex':
        return FlexMLP(in_dim, config['hidden_dims'])
    if config['arch'] == 'residual':
        return ResidualMLP(in_dim, config['hidden_dim'], config['n_blocks'])
    raise ValueError(f"Unknown arch: {config['arch']!r}")


def run_mlp_cv(variant_name: str,
               config: dict,
               X: np.ndarray,
               y: np.ndarray,
               fold_labels: np.ndarray,
               g0_values: list[float],
               epochs: int = 30,
               batch_size: int = 262144,
               lr: float = 1e-3) -> list[dict]:
    """7-fold CV for one MLP config.
    Training loop is identical to classical_models.run_mlp (GPU-preload,
    torch.randperm batching, AMP, CosineAnnealingLR) but uses a configurable
    model architecture.
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    fold_metrics: list[dict] = []

    for fold in range(len(g0_values)):
        mask   = fold_labels != fold
        X_tr, y_tr = X[mask],  y[mask]
        X_va, y_va = X[~mask], y[~mask]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
        X_va_s = sc.transform(X_va).astype(np.float32)

        # Preload entire training fold to GPU to eliminate per-batch transfer overhead
        X_tr_t = torch.from_numpy(X_tr_s).to(device)
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
        n_tr   = len(X_tr_t)

        model      = _build_mlp(config, X.shape[1]).to(device)
        opt        = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn    = nn.MSELoss()
        scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, batch_size):
                xb = X_tr_t[perm[i : i + batch_size]]
                yb = y_tr_t[perm[i : i + batch_size]]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = loss_fn(model(xb), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            sched.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            y_pred = model(torch.from_numpy(X_va_s).to(device)).float().cpu().numpy()

        fold_metrics.append(compute_metrics(y_va, y_pred))
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")

    return fold_metrics


def _train_cnn_fold(train_vols: list[dict],
                    val_vols: list[dict],
                    ops: list[np.ndarray],
                    device: torch.device,
                    epochs: int,
                    lr: float,
                    base_ch: int) -> dict:
    """Single CNN fold.  Mirrors train_cnn.train_one_fold but parametrizes base_ch."""
    train_ds = CubeDataset(train_vols, ops, augment=True)
    val_ds   = CubeDataset(val_vols,   ops=None, augment=False)

    # Per-channel input normalisation (training stats only)
    all_x   = torch.stack(train_ds.xs)
    ch_mean = all_x.mean(dim=(0, 2, 3, 4), keepdim=True).squeeze(0)
    ch_std  = all_x.std( dim=(0, 2, 3, 4), keepdim=True).squeeze(0).clamp(min=1e-6)
    train_ds.xs = [(x - ch_mean) / ch_std for x in train_ds.xs]
    val_ds.xs   = [(x - ch_mean) / ch_std for x in val_ds.xs]

    # Per-fold target normalisation (balanced log_fh2 loss)
    all_y  = torch.stack(train_ds.ys)
    y_mean = all_y.mean()
    y_std  = all_y.std().clamp(min=1e-6)
    train_ds.ys = [(y - y_mean) / y_std for y in train_ds.ys]
    val_ds.ys   = [(y - y_mean) / y_std for y in val_ds.ys]

    pin_mem  = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=pin_mem)

    model   = UNet3D(n_channels=len(CNN_INPUT_COLS), base_ch=base_ch).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    best_val   = float('inf')
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=pin_mem)
            yb = yb.to(device, non_blocking=pin_mem)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = xb.to(device, non_blocking=pin_mem)
                yb = yb.to(device, non_blocking=pin_mem)
                val_loss += loss_fn(model(xb), yb).item()
        val_loss /= len(val_dl)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{epochs}  val_loss={val_loss:.4f}")

    # Restore best checkpoint and evaluate in original log_fh2 space
    model.load_state_dict(best_state)
    model.eval()
    y_true_parts, y_pred_parts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            y_pred_parts.append(model(xb.to(device)).float().cpu().numpy().ravel())
            y_true_parts.append(yb.numpy().ravel())

    y_std_np  = y_std.item()
    y_mean_np = y_mean.item()
    y_true = np.concatenate(y_true_parts) * y_std_np + y_mean_np
    y_pred = np.concatenate(y_pred_parts) * y_std_np + y_mean_np
    return compute_metrics(y_true, y_pred)


def run_cnn_cv_variant(variant_name: str,
                       config: dict,
                       all_vols: list[dict],
                       g0_values: list[float],
                       device: torch.device,
                       ops: list[np.ndarray],
                       epochs: int = 50,
                       lr: float = 1e-3) -> list[dict]:
    """7-fold CV for one CNN config variant."""
    base_ch  = config['base_ch']
    n_params = count_parameters(UNet3D(n_channels=len(CNN_INPUT_COLS), base_ch=base_ch))
    print(f"  {variant_name}  base_ch={base_ch}  params={n_params:,}")
    fold_metrics: list[dict] = []
    for fold in range(len(g0_values)):
        print(f"  [Fold {fold + 1}/{len(g0_values)}] Val G0={g0_values[fold]:.1f}")
        train_vols = [v for i, v in enumerate(all_vols) if i != fold]
        val_vols   = [all_vols[fold]]
        metrics = _train_cnn_fold(train_vols, val_vols, ops, device, epochs, lr, base_ch)
        fold_metrics.append(metrics)
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={metrics['R2']:.4f}")
    return fold_metrics


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_comparison(all_results: dict[str, list[dict]],
                     g0_values: list[float]) -> None:
    """Print a compact R2 comparison table: variants (rows) x G0 folds (cols)."""
    col_w   = 9   # width per G0 column
    name_w  = 16
    g0_hdr  = "".join(f"G0={g:<5.1f}" for g in g0_values)
    header  = f"  {'Variant':<{name_w}}  {g0_hdr}  {'MeanR2':>8}  {'Std':>6}"
    sep     = "=" * len(header)

    print(f"\n{sep}")
    print("  Architecture Comparison  (R2 per fold)")
    print(sep)
    print(header)
    print("-" * len(header))

    for name, fms in all_results.items():
        r2s = [m['R2'] for m in fms]
        r2_cols = "".join(f"{r2:>+8.4f} " for r2 in r2s)
        print(f"  {name:<{name_w}}  {r2_cols}  {np.mean(r2s):>+8.4f}  {np.std(r2s):>6.4f}")

    print(sep)


def save_comparison_log(all_results: dict[str, list[dict]],
                        g0_values: list[float],
                        run_config: dict,
                        log_path: str) -> None:
    out: dict = {'run_config': run_config, 'g0_values': g0_values, 'variants': {}}
    for name, fms in all_results.items():
        r2s  = [m['R2']   for m in fms]
        rmse = [m['RMSE'] for m in fms]
        mae  = [m['MAE']  for m in fms]
        out['variants'][name] = {
            'folds': [
                {'fold': i, 'g0': g0_values[i],
                 'metrics': {k: float(v) for k, v in m.items()}}
                for i, m in enumerate(fms)
            ],
            'summary': {
                'R2':   {'mean': float(np.mean(r2s)),  'std': float(np.std(r2s))},
                'RMSE': {'mean': float(np.mean(rmse)), 'std': float(np.std(rmse))},
                'MAE':  {'mean': float(np.mean(mae)),  'std': float(np.std(mae))},
            },
        }
    with open(log_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nComparison log saved -> {log_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare XGBoost / MLP / CNN architecture variants '
                    'with leave-one-G0-out cross-validation.')
    parser.add_argument('--skip-xgb',   action='store_true',
                        help='Skip all XGBoost variants')
    parser.add_argument('--skip-mlp',   action='store_true',
                        help='Skip all MLP variants')
    parser.add_argument('--skip-cnn',   action='store_true',
                        help='Skip all CNN variants')
    parser.add_argument('--cnn-epochs', type=int, default=50,
                        help='CNN epochs per variant/fold '
                             '(default 50 for quick comparison; use 150 for final runs)')
    parser.add_argument('--mlp-epochs', type=int, default=30,
                        help='MLP epochs per variant/fold (default 30)')
    parser.add_argument('--all-ops',    action='store_true',
                        help='Use all 48 Oh symmetry ops for CNN (default: 8 z-preserving)')
    parser.add_argument('--log',        type=str, default=None,
                        help='Output JSON path '
                             '(default: arch_comparison_TIMESTAMP.json)')
    args = parser.parse_args()

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = args.log or f'arch_comparison_{ts}.json'

    print("Loading data...")
    cubes         = load_all_cubes()
    g0_vals       = get_g0_values(cubes)
    X, y, folds   = get_X_y(cubes, use_log_target=True)
    print(f"Total samples: {len(X):,}  |  Features: {X.shape[1]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    run_config = {
        'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
        'device':     str(device),
        'cnn_epochs': args.cnn_epochs,
        'mlp_epochs': args.mlp_epochs,
        'all_ops':    args.all_ops,
    }

    all_results: dict[str, list[dict]] = {}

    # ── XGBoost variants ──────────────────────────────────────────────────────
    if not args.skip_xgb:
        print("\n--- XGBoost variants ---")
        for name, cfg in XGB_VARIANTS.items():
            print(f"\n[{name}]")
            all_results[name] = run_xgb_cv(name, cfg, X, y, folds, g0_vals)

    # ── MLP variants ──────────────────────────────────────────────────────────
    if not args.skip_mlp:
        print(f"\n--- MLP variants ({args.mlp_epochs} epochs) ---")
        for name, cfg in MLP_VARIANTS.items():
            print(f"\n[{name}]")
            all_results[name] = run_mlp_cv(name, cfg, X, y, folds, g0_vals,
                                           epochs=args.mlp_epochs)

    # ── CNN variants ──────────────────────────────────────────────────────────
    if not args.skip_cnn:
        print(f"\n--- CNN variants ({args.cnn_epochs} epochs) ---")
        print("Converting cubes to 128^3 volumes...")
        vol_cols = [c for c in CNN_INPUT_COLS + [CNN_TARGET_COL]
                    if c in cubes[0].columns]
        all_vols = [cube_to_volumes(df, vol_cols) for df in cubes]
        ops      = get_symmetry_ops(safe_only=not args.all_ops)
        print(f"Symmetry ops: {len(ops)}  "
              f"({'z-preserving' if not args.all_ops else 'full Oh'})")
        for name, cfg in CNN_VARIANTS.items():
            print(f"\n[{name}]")
            all_results[name] = run_cnn_cv_variant(
                name, cfg, all_vols, g0_vals, device, ops,
                epochs=args.cnn_epochs)

    print_comparison(all_results, g0_vals)
    save_comparison_log(all_results, g0_vals, run_config, log_path)
