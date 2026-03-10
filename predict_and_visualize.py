#!/usr/bin/env python3
"""
predict_and_visualize.py
========================
Train the best ensemble (ens_sp: XGBoost + MLP with multi-scale spatial
features 3^3+5^3+7^3) on 6 of the 7 G0 cubes, predict on the held-out cube,
and display a 3-panel interactive 3D comparison:

    [ Ground Truth ]  |  [ ens_sp Prediction ]  |  [ |Error| ]

Best ensemble from run 121724: ens_sp R2=0.948, std=0.024
G0 per-fold: 0.1->0.908 | 0.2->0.915 | 0.4->0.953 | 0.8->0.962

Usage
-----
  # Hold out G0=0.8 (default, near-best fold):
  python predict_and_visualize.py

  # Hold out the hardest extrapolation fold:
  python predict_and_visualize.py --g0 0.2

  # Quick demo (fewer MLP epochs):
  python predict_and_visualize.py --g0 1.6 --mlp-epochs 30

  # Single-scale spatial only:
  python predict_and_visualize.py --spatial-kernels 3
"""

import argparse
import datetime
import os
import sys
import numpy as np

# Ensure relative data paths (icedrive-dl-182bd/UVonly) resolve correctly
# regardless of which directory Code Runner / the shell launches from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn as nn
import xgboost as xgb
import pyvista as pv
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import uniform_filter

from data_loader import (
    load_all_cubes, cube_to_volumes, get_X_y,
    get_g0_values, FEATURE_COLS, LOG_TARGET_COL,
)
from classical_models import compute_metrics

# ── XGBoost config (xgb_standard) ─────────────────────────────────────────────

_XGB_CFG = dict(
    max_depth=6, n_estimators=400, learning_rate=0.10,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method='hist',
)

# ── MLP model (mlp_wide) ───────────────────────────────────────────────────────

class _FlexMLP(nn.Module):
    """Feed-forward MLP with configurable hidden dimensions and BatchNorm."""
    def __init__(self, in_dim: int, hidden_dims: list[int] = (512, 512, 256, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

# ── Spatial feature helper ─────────────────────────────────────────────────────

def _compute_spatial_X(cubes: list, all_vols: list[dict], feature_cols: list[str],
                        kernel_sizes: tuple[int, ...] = (3, 5, 7)) -> np.ndarray:
    """Multi-scale neighbourhood mean features via scipy.ndimage.uniform_filter.

    For each kernel size in kernel_sizes, computes the k^3 neighbourhood mean at
    every grid point, then indexes back to DataFrame row order using ix/iy/iz.
    Results from all scales are concatenated along the feature axis.

    Returns (N, len(feature_cols) * len(kernel_sizes)) float32 array.
    Default scales (3, 5, 7) give 3 x 14 = 42 spatial features.
    """
    parts = []
    for cube, vol in zip(cubes, all_vols):
        ix = cube['ix'].values.astype(int) - 1
        iy = cube['iy'].values.astype(int) - 1
        iz = cube['iz'].values.astype(int) - 1
        scale_feats = [
            np.stack([
                uniform_filter(vol[col], size=ks)[ix, iy, iz]
                for col in feature_cols
            ], axis=-1).astype(np.float32)
            for ks in kernel_sizes
        ]
        parts.append(np.concatenate(scale_feats, axis=-1))
    return np.concatenate(parts, axis=0)


def _preds_to_volume(cube_df, y_pred_log: np.ndarray) -> np.ndarray:
    """Reshape flat per-cell predictions to a 128^3 float32 volume."""
    vol = np.zeros((128, 128, 128), dtype=np.float32)
    ix  = cube_df['ix'].values.astype(int) - 1
    iy  = cube_df['iy'].values.astype(int) - 1
    iz  = cube_df['iz'].values.astype(int) - 1
    vol[ix, iy, iz] = y_pred_log.astype(np.float32)
    return np.nan_to_num(vol)

# ── Training functions ─────────────────────────────────────────────────────────

def _train_xgb(X_tr: np.ndarray, y_tr: np.ndarray,
               X_va: np.ndarray, y_va: np.ndarray) -> np.ndarray:
    """Fit XGBoost (xgb_standard config) and return val predictions.

    CPU only: XGBRegressor's sklearn API takes numpy arrays (CPU); setting
    device='cuda' would cause a DMatrix device-mismatch warning at predict time.
    """
    sc = StandardScaler()
    model = xgb.XGBRegressor(**_XGB_CFG)
    model.fit(sc.fit_transform(X_tr), y_tr,
              eval_set=[(sc.transform(X_va), y_va)], verbose=False)
    return model.predict(sc.transform(X_va)).astype(np.float32)


def _train_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
               X_va: np.ndarray, y_va: np.ndarray,
               epochs: int = 100) -> np.ndarray:
    """Fit MLP (mlp_wide config) and return val predictions."""
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    X_va_s = sc.transform(X_va).astype(np.float32)

    X_tr_t = torch.from_numpy(X_tr_s).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    n_tr   = len(X_tr_t)

    torch.manual_seed(0)
    np.random.seed(0)

    model      = _FlexMLP(X_tr_s.shape[1]).to(device)
    opt        = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn    = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
    batch_size = 262_144

    for ep in range(epochs):
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
        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                y_vp = model(torch.from_numpy(X_va_s).to(device)).float().cpu().numpy()
            m = compute_metrics(y_va, y_vp)
            print(f"    ep {ep+1:3d}/{epochs}  R2={m['R2']:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_va_s).to(device)).float().cpu().numpy()
    return y_pred.astype(np.float32)

# ── Pyvista helpers ────────────────────────────────────────────────────────────

def _to_pv_grid(vol: np.ndarray, scalar_name: str = 'values') -> pv.ImageData:
    """Wrap a (128, 128, 128) numpy array in a pyvista ImageData cell grid."""
    grid = pv.ImageData()
    grid.dimensions = np.array(vol.shape) + 1   # cell-centered: dims = cells+1
    grid.origin  = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.cell_data[scalar_name] = vol.flatten(order='F')
    return grid


def visualize(truth_vol: np.ndarray, pred_vol: np.ndarray,
              err_vol: np.ndarray, g0: float, r2_xgb: float,
              r2_mlp: float, r2_ens: float,
              scale_label: str = 'nH2') -> None:
    """Three-panel interactive 3D volume rendering.

    Left   – Ground truth log10(nH2)
    Centre – ens_sp prediction (equal-weight XGB_sp + MLP_sp)
    Right  – Absolute prediction error |pred - truth| in log10 space

    All three panels share a linked camera so rotating one rotates all.
    """
    # Shared colour limits from truth (1st–99th percentile avoids outlier stretch)
    p1, p99 = np.percentile(truth_vol, [1, 99])
    clim    = (float(p1), float(p99))

    plotter = pv.Plotter(shape=(1, 3), window_size=(1920, 700))
    plotter.background_color = 'black'

    # ── Left: Ground truth ────────────────────────────────────────────────────
    plotter.subplot(0, 0)
    plotter.add_volume(_to_pv_grid(truth_vol), scalars='values',
                       cmap='magma', opacity='linear', clim=clim,
                       scalar_bar_args={'title': scale_label, 'color': 'white'})
    plotter.add_text(f"Ground Truth  G0={g0}", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # ── Centre: Prediction ────────────────────────────────────────────────────
    plotter.subplot(0, 1)
    plotter.add_volume(_to_pv_grid(pred_vol), scalars='values',
                       cmap='magma', opacity='linear', clim=clim,
                       scalar_bar_args={'title': scale_label, 'color': 'white'})
    plotter.add_text(
        f"ens_sp Prediction  R2={r2_ens:.4f}\n"
        f"XGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}",
        position='upper_edge', font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # ── Right: Absolute error ─────────────────────────────────────────────────
    plotter.subplot(0, 2)
    plotter.add_volume(_to_pv_grid(err_vol), scalars='values',
                       cmap='bwr', opacity='linear', clim=clim,
                       scalar_bar_args={'title': f'error ({scale_label})', 'color': 'white'})
    plotter.add_text("Prediction - Truth", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # Link cameras so all three panels rotate together
    plotter.link_views()

    print("\nControls: left-click+drag to rotate | right-click+drag to zoom | "
          "middle-click+drag to pan")
    print("All three panels share a linked camera.")
    plotter.show()

# ── Per-fold training / saving ─────────────────────────────────────────────────

def _run_fold(fold: int, g0_val: float,
              X_sp: np.ndarray, y: np.ndarray, fold_labels: np.ndarray,
              cubes: list, mlp_epochs: int, spatial_kernels: list) -> None:
    """Train ens_sp on N-1 cubes, predict on the held-out fold, and save .npz."""
    mask = fold_labels != fold
    X_tr = X_sp[mask];   y_tr = y[mask]
    X_va = X_sp[~mask];  y_va = y[~mask]

    print(f"\n[xgb_standard_sp]  {X_tr.shape[0]:,} training samples...")
    y_xgb = _train_xgb(X_tr, y_tr, X_va, y_va)
    m_xgb = compute_metrics(y_va, y_xgb)
    print(f"  XGB  R2={m_xgb['R2']:.4f}  RMSE={m_xgb['RMSE']:.6f}")

    print(f"\n[mlp_wide_sp]  {mlp_epochs} epochs...")
    y_mlp = _train_mlp(X_tr, y_tr, X_va, y_va, epochs=mlp_epochs)
    m_mlp = compute_metrics(y_va, y_mlp)
    print(f"  MLP  R2={m_mlp['R2']:.4f}  RMSE={m_mlp['RMSE']:.6f}")

    y_ens = 0.5 * y_xgb + 0.5 * y_mlp
    m_ens = compute_metrics(y_va, y_ens)
    print(f"\n[ens_sp]  R2={m_ens['R2']:.4f}  RMSE={m_ens['RMSE']:.6f}")

    pred_vol = _preds_to_volume(cubes[fold], y_ens)

    os.makedirs('predictions', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'predictions/pred_g0_{g0_val}_{timestamp}.npz'
    np.savez_compressed(
        save_path,
        pred_vol        = pred_vol,
        g0              = np.float64(g0_val),
        r2_xgb          = np.float32(m_xgb['R2']),
        r2_mlp          = np.float32(m_mlp['R2']),
        r2_ens          = np.float32(m_ens['R2']),
        spatial_kernels = np.array(spatial_kernels, dtype=np.int32),
        mlp_epochs      = np.int32(mlp_epochs),
    )
    print(f"Prediction saved -> {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict nH2 with ens_sp and compare 3D volumes against truth.')
    parser.add_argument('--g0', type=float, default=0.8,
                        help='G0 value to hold out as validation (default: 0.8). '
                             'Available: 0.1 0.2 0.4 0.8 1.6 3.2 6.4')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='MLP training epochs (default: 100; use 30 for a fast demo)')
    parser.add_argument('--spatial-kernels', nargs='+', type=int, default=[3, 5, 7],
                        help='Spatial filter kernel sizes (default: 3 5 7)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 7 G0 folds and save a prediction file for each')
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    # ── Baseline features ─────────────────────────────────────────────────────
    print("\nBuilding feature matrix...")
    X, y, fold_labels = get_X_y(cubes, use_log_target=True)
    print(f"  Base X shape: {X.shape}   y shape: {y.shape}")

    # ── Spatial features (needs volumes) ──────────────────────────────────────
    print(f"\nBuilding volumes and spatial features "
          f"(kernels={args.spatial_kernels})...")
    all_vols = [cube_to_volumes(df, FEATURE_COLS) for df in cubes]
    X_extra  = _compute_spatial_X(cubes, all_vols, FEATURE_COLS,
                                   kernel_sizes=tuple(args.spatial_kernels))
    n_sp = len(FEATURE_COLS) * len(args.spatial_kernels)
    X_sp = np.concatenate([X, X_extra], axis=1)
    print(f"  Spatial features: {n_sp}  ->  X_sp shape: {X_sp.shape}")

    # ── Run fold(s) ────────────────────────────────────────────────────────────
    if args.all:
        print(f"\nRunning all {len(g0_vals)} folds...")
        for fold, g0_val in enumerate(g0_vals):
            print(f"\n{'='*60}\nFold {fold+1}/{len(g0_vals)}  G0={g0_val}\n{'='*60}")
            _run_fold(fold, g0_val, X_sp, y, fold_labels, cubes,
                      args.mlp_epochs, args.spatial_kernels)
    else:
        if args.g0 not in g0_vals:
            print(f"ERROR: G0={args.g0} not found. Available: {g0_vals}", file=sys.stderr)
            sys.exit(1)
        fold = g0_vals.index(args.g0)
        print(f"\nValidation fold: {fold}  (G0={args.g0})")
        print(f"Training on {len(cubes) - 1} cubes  |  "
              f"Validating on 1 cube (G0={args.g0})")
        _run_fold(fold, args.g0, X_sp, y, fold_labels, cubes,
                  args.mlp_epochs, args.spatial_kernels)


if __name__ == '__main__':
    main()
