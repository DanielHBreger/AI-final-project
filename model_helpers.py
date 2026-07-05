"""
model_helpers.py
================
Shared helpers for the stacked_sp training pipeline.

Imported by: predict_and_visualize.py, compare_architectures.py,
             single_cube_extrapolation.py, intra_cube_section.py

Contents
--------
  GLOBAL_SEED          -- canonical random seed used across all scripts
  _set_seeds           -- set all RNG seeds + cudnn determinism flags
  _get_env_info        -- collect package versions for experiment logs
  _XGB_CFG             -- xgb_standard hyper-parameter dict
  FlexMLP              -- configurable feed-forward MLP (BatchNorm + ReLU)
  _compute_spatial_X   -- multi-scale neighbourhood mean features
  _compute_weights     -- exponential density-weighted sample weights
  _preds_to_volume     -- reshape flat predictions to 128^3 volume
  fit_g0_bias_correction -- fit per-cube OOF bias vs log10(G0) (mass-budget fix)
  predict_bias         -- evaluate the fitted bias correction at a G0 value
  _fit_xgb             -- fit density-weighted XGBoost (*_sp_w config), return (model, scaler)
  _predict_xgb         -- predict with fitted XGBoost
  _fit_mlp             -- fit density-weighted MLP (*_sp_w config), return (model, scaler, device, y_min, y_max)
  _predict_mlp         -- predict with fitted MLP (with output clipping)

NOTE: _fit_xgb/_fit_mlp always apply density weighting (_compute_weights),
so every pipeline built on them (predict_and_visualize,
single_cube_extrapolation, intra_cube_section) trains the WEIGHTED model
family — the *_sp_w / stacked_weighted rows of the comparison logs, not
the unweighted *_sp rows.
"""

import platform
import random
import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge   # noqa: F401 — re-exported for convenience

from classical_models import compute_metrics


# ── Reproducibility ───────────────────────────────────────────────────────────

GLOBAL_SEED = 67


def _set_seeds(seed: int) -> None:
    """Set all global RNG seeds and enable cudnn determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_env_info() -> dict:
    """Return package versions and hardware info for experiment logs."""
    import sklearn
    import scipy
    info: dict = {
        'python':             sys.version,
        'platform':           platform.platform(),
        'numpy':              np.__version__,
        'torch':              torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'xgboost':            xgb.__version__,
        'sklearn':            sklearn.__version__,
        'scipy':              scipy.__version__,
        'cuda_available':     torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info['cuda_device']       = torch.cuda.get_device_name(0)
        info['cuda_device_count'] = torch.cuda.device_count()
    return info


# ── XGBoost config ─────────────────────────────────────────────────────────────

_XGB_CFG = dict(
    max_depth=6, n_estimators=400, learning_rate=0.10,
    subsample=0.3, colsample_bytree=0.8, tree_method='hist',
    random_state=42, verbosity=0,
)


# ── MLP model ──────────────────────────────────────────────────────────────────

class FlexMLP(nn.Module):
    """Feed-forward MLP with configurable hidden dimensions and BatchNorm."""
    def __init__(self, in_dim: int, hidden_dims: list[int] = [512, 512, 256, 128]):
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

    kernel_sizes are FULL side lengths passed directly to scipy uniform_filter's
    `size` parameter (e.g. size=3 → 3×3×3 = 27-voxel window). The paper's
    Eq. 3 label "kernel half-widths" is a notation error; the code is correct.

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


# ── CNN volume normalisation ───────────────────────────────────────────────────

def normalize_channels_inplace(train_xs: list[torch.Tensor],
                               val_xs: list[torch.Tensor]) -> None:
    """Per-channel standardisation of lists of (C, D, H, W) tensors, in place.

    Statistics come from the training list only (fold-safe) and are
    accumulated in float64 one volume at a time: at native 128^3 resolution,
    stacking the full augmented training set for a single .mean()/.std()
    call would allocate a ~6 GB temporary."""
    n_ch = train_xs[0].shape[0]
    s  = torch.zeros(n_ch, dtype=torch.float64)
    s2 = torch.zeros(n_ch, dtype=torch.float64)
    n  = 0
    for x in train_xs:
        xd = x.double()
        s  += xd.sum(dim=(1, 2, 3))
        s2 += (xd * xd).sum(dim=(1, 2, 3))
        n  += x[0].numel()
    mean = s / n
    std  = (s2 / n - mean * mean).clamp(min=0.0).sqrt()
    mean_f = mean.float().view(-1, 1, 1, 1)
    std_f  = std.float().clamp(min=1e-6).view(-1, 1, 1, 1)
    for x in train_xs:
        x.sub_(mean_f).div_(std_f)
    for x in val_xs:
        x.sub_(mean_f).div_(std_f)


def normalize_targets_inplace(train_ys: list[torch.Tensor],
                              val_ys: list[torch.Tensor]) -> tuple[float, float]:
    """Scalar standardisation of target volumes, in place (training stats
    only).  Returns (mean, std) for inverse-transforming predictions."""
    s = s2 = 0.0
    n = 0
    for y in train_ys:
        yd = y.double()
        s  += float(yd.sum())
        s2 += float((yd * yd).sum())
        n  += y.numel()
    mean = s / n
    std  = max(max(s2 / n - mean * mean, 0.0) ** 0.5, 1e-6)
    for y in train_ys:
        y.sub_(mean).div_(std)
    for y in val_ys:
        y.sub_(mean).div_(std)
    return float(mean), float(std)


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


# ── G0-dependent bias recalibration (mass-budget fix) ─────────────────────────
#
# The stacked ensemble carries a systematic positive residual (pred - true) of
# ~+0.2 dex in the molecular phase, which compounds to a x1.6-1.9 error in the
# total H2 mass of a predicted cube.  A constant offset cannot fix this: the
# Ridge meta-learner fits an intercept on the pooled OOF sample, so the pooled
# mean residual is already ~0 and the surviving bias is G0-dependent.  The fix
# fits the per-training-cube OOF bias as a linear function of log10(G0) and
# subtracts the value interpolated/extrapolated at the held-out G0.  Only
# training-cube quantities are used: leakage-free.
#
# The per-cube bias can be computed two ways: unweighted mean residual
# (centres the cell-mean log bias; 'mean'/'_cal') or mass-weighted via
# mass_weighted_bias below (centres the total-H2-mass budget;
# 'mass'/'_mwcal').  The 2026-07-03 runs showed the two disagree: the
# stacked models are cell-mean unbiased yet under-recover mass at high G0,
# so only the mass-weighted variant addresses the mass budget.

def fit_g0_bias_correction(per_cube_bias: np.ndarray,
                           g0_train: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of per-cube OOF prediction bias against log10(G0).

    per_cube_bias[j] is the mean residual (pred - true, dex) of the stacked
    model's out-of-fold predictions on training cube j; g0_train[j] is that
    cube's G0.  Returns (slope, intercept) such that

        predicted_bias(G0) = slope * log10(G0) + intercept.
    """
    lg = np.log10(np.asarray(g0_train, dtype=np.float64))
    A  = np.stack([lg, np.ones_like(lg)], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(per_cube_bias, dtype=np.float64),
                               rcond=None)
    return float(coef[0]), float(coef[1])


def predict_bias(slope: float, intercept: float, g0: float) -> float:
    """Evaluate the fitted bias correction at a G0 value (dex offset)."""
    return slope * float(np.log10(g0)) + intercept


def mass_weighted_bias(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mass-weighted mean residual (dex): weights 10**y_true (linear nH2).

    Unlike the unweighted mean residual, this is the quantity whose
    subtraction (as a constant dex offset) centres the predicted total H2
    mass on the true mass — dense cells dominate both the weights and the
    mass budget."""
    yt = np.asarray(y_true, dtype=np.float64)
    w  = 10.0 ** yt
    r  = np.asarray(y_pred, dtype=np.float64) - yt
    return float((w * r).sum() / w.sum())


# ── Fit / predict helpers (single-model, return model for reuse) ───────────────

def _fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray,
             ) -> tuple[xgb.XGBRegressor, StandardScaler]:
    """Fit XGBoost with density weighting (the *_sp_w config, matching the
    xgb_standard_sp_w rows of the comparison logs). Return (model, scaler)."""
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
             seed: int = 0,
             ) -> tuple[nn.Module, StandardScaler, torch.device, float, float]:
    """Fit MLP with density weighting (the *_sp_w config, matching the
    mlp_wide_sp_w rows of the comparison logs).
    Return (model, scaler, device, y_min, y_max)."""
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = FlexMLP(X_tr_s.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp) # type: ignore
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
            with torch.amp.autocast('cuda', enabled=use_amp): # type: ignore
                loss = (wb * (model(xb) - yb) ** 2).mean()
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
        sched.step()
        if not quiet and ep in log_at:
            model.eval()
            with torch.no_grad():
                y_tr_p = model(X_tr_t).float().cpu().numpy()
            m = compute_metrics(y_tr, y_tr_p, fast=True)
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
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp): # type: ignore
        y_pred = model(torch.from_numpy(X_s).to(device)).float().cpu().numpy()
    return np.clip(y_pred, y_min - 2.0, y_max + 2.0).astype(np.float32)
