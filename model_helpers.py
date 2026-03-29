"""
model_helpers.py
================
Shared helpers for the stacked_sp training pipeline.

Imported by: predict_and_visualize.py, compare_architectures.py,
             single_cube_extrapolation.py, intra_cube_section.py

Contents
--------
  _XGB_CFG             -- xgb_standard hyper-parameter dict
  FlexMLP              -- configurable feed-forward MLP (BatchNorm + ReLU)
  _compute_spatial_X   -- multi-scale neighbourhood mean features
  _compute_weights     -- exponential density-weighted sample weights
  _preds_to_volume     -- reshape flat predictions to 128^3 volume
  _fit_xgb             -- fit XGBoost, return (model, scaler)
  _predict_xgb         -- predict with fitted XGBoost
  _fit_mlp             -- fit MLP with density weighting, return (model, scaler, device, y_min, y_max)
  _predict_mlp         -- predict with fitted MLP (with output clipping)
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge   # noqa: F401 — re-exported for convenience

from classical_models import compute_metrics


# ── XGBoost config ─────────────────────────────────────────────────────────────

_XGB_CFG = dict(
    max_depth=6, n_estimators=400, learning_rate=0.10,
    subsample=0.3, colsample_bytree=0.8, tree_method='hist',
    random_state=42, verbosity=0,
)


# ── MLP model ──────────────────────────────────────────────────────────────────

class FlexMLP(nn.Module):
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
    Default scales (3, 5, 7) give 3 x 15 = 45 spatial features.
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


# ── Density-weighted sample weights ────────────────────────────────────────────

def _compute_weights(y_tr: np.ndarray,
                     alpha: float = 100.0,
                     lo_pct: float = 99.0,
                     hi_pct: float = 99.99) -> np.ndarray:
    """Smooth exponential weight: 1x at p99, alpha x at p99.99, flat outside.
    Mean-normalized so the effective learning rate is unchanged."""
    p_lo = float(np.percentile(y_tr, lo_pct))
    p_hi = float(np.percentile(y_tr, hi_pct))
    if p_hi <= p_lo:
        return np.ones(len(y_tr), dtype=np.float32)
    t = np.clip((y_tr - p_lo) / (p_hi - p_lo), 0.0, 1.0).astype(np.float64)
    w = np.exp(np.log(alpha) * t).astype(np.float32)
    return (w / w.mean()).astype(np.float32)


# ── Volume reshape ─────────────────────────────────────────────────────────────

def _preds_to_volume(cube_df, y_pred_log: np.ndarray) -> np.ndarray:
    """Reshape flat per-cell predictions to a 128^3 float32 volume."""
    vol = np.zeros((128, 128, 128), dtype=np.float32)
    ix  = cube_df['ix'].values.astype(int) - 1
    iy  = cube_df['iy'].values.astype(int) - 1
    iz  = cube_df['iz'].values.astype(int) - 1
    vol[ix, iy, iz] = y_pred_log.astype(np.float32)
    return np.nan_to_num(vol)


# ── Fit / predict helpers (single-model, return model for reuse) ───────────────

def _fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray,
             ) -> tuple[xgb.XGBRegressor, StandardScaler]:
    """Fit XGBoost (xgb_standard + density weighting). Return (model, scaler)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sc = StandardScaler()
    X_s = sc.fit_transform(X_tr)
    model = xgb.XGBRegressor(**_XGB_CFG, device=device)
    model.fit(X_s, y_tr, sample_weight=_compute_weights(y_tr), verbose=False)
    return model, sc


def _predict_xgb(model: xgb.XGBRegressor, sc: StandardScaler,
                 X: np.ndarray) -> np.ndarray:
    """Predict with a fitted XGBoost model."""
    return model.predict(sc.transform(X)).astype(np.float32)


def _fit_mlp(X_tr: np.ndarray, y_tr: np.ndarray,
             epochs: int = 100, quiet: bool = False,
             ) -> tuple[nn.Module, StandardScaler, torch.device, float, float]:
    """Fit MLP (mlp_wide + density weighting). Return (model, scaler, device, y_min, y_max)."""
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    sc     = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr).astype(np.float32)
    y_min  = float(y_tr.min())
    y_max  = float(y_tr.max())

    X_tr_t = torch.from_numpy(X_tr_s).to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    n_tr   = len(X_tr_t)
    w_tr_t = torch.from_numpy(_compute_weights(y_tr)).to(device)

    torch.manual_seed(0)
    np.random.seed(0)

    model = FlexMLP(X_tr_s.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
    batch_size = 262_144

    log_at = {0, epochs // 4, epochs // 2, 3 * epochs // 4, epochs - 1}

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, batch_size):
            xb = X_tr_t[perm[i : i + batch_size]]
            yb = y_tr_t[perm[i : i + batch_size]]
            wb = w_tr_t[perm[i : i + batch_size]]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = (wb * (model(xb) - yb) ** 2).mean()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
        sched.step()
        if not quiet and ep in log_at:
            model.eval()
            with torch.no_grad():
                y_tr_p = model(X_tr_t).float().cpu().numpy()
            m = compute_metrics(y_tr, y_tr_p)
            print(f"    ep {ep+1:3d}/{epochs}  train-R2={m['R2']:.4f}")

    model.eval()
    return model, sc, device, y_min, y_max


def _predict_mlp(model: nn.Module, sc: StandardScaler,
                 device: torch.device,
                 X: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """Apply MLP and clip predictions to training range +/- 2 dex."""
    use_amp = device.type == 'cuda'
    X_s = sc.transform(X).astype(np.float32)
    model.eval()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
        y_pred = model(torch.from_numpy(X_s).to(device)).float().cpu().numpy()
    return np.clip(y_pred, y_min - 2.0, y_max + 2.0).astype(np.float32)
