"""
compare_architectures.py
Systematic comparison of architectural variants for XGBoost, MLP, and CNN under
leave-one-G0-out cross-validation (7 folds, one held-out cube per fold).

Architecture selection rationale
---------------------------------
XGBoost (1 variant)
  xgb_standard  depth=6, 400 trees, lr=0.10
    Confirmed optimal across 3 runs.  depth=4 and depth=8 were both tested
    and retired (depth=4 slightly underfit; depth=8 overfits).

MLP (1 variant)
  mlp_wide  [512, 512, 256, 128]
    Confirmed best for ensemble/stacking.  mlp_standard and mlp_residual
    variants were tested across 3 runs and retired (similar standalone R2
    but never improve ensemble performance).

CNN (3 variants)
  The only spatial model.  The main architectural axis is base channel count.

  unet_small   base_ch=16  ~1.5 M params
    Tests whether reduced capacity closes the train/val gap (training loss
    collapses to ~0.005 with base_ch=32, suggesting over-parameterisation).

  unet_standard  base_ch=32  ~5.8 M params  [BASELINE — current best, R2=0.803]

  unet_large  base_ch=64  ~23 M params
    Upper bound on CNN capacity.

Extras (Paths B-E)
  ens_xgb+mlp / ens_xgb+cnn / ens_all
    Equal-weight prediction ensemble across model families.

  xgb_standard_sp / mlp_wide_sp  (--spatial flag)
    Append local 3^3 neighbourhood-mean features via scipy.ndimage.uniform_filter.
    Gives pointwise models a glimpse of spatial context without full CNN.

  unet_xgb_guided
    3D U-Net with XGBoost's OOB prediction prepended as a 15th input channel.
    Each cube's XGBoost volume comes from the fold where it was held out (no
    leakage). Hypothesis: providing a near-correct "prior" speeds convergence
    and improves extrapolation folds.

Usage
-----
  # Default run (XGBoost + MLP + multi-scale spatial, no CNN):
  python compare_architectures.py

  # Fast smoke test (no spatial, 5 MLP epochs):
  python compare_architectures.py --no-spatial --mlp-epochs 5

  # Include CNN variants (slow — adds ~2x runtime):
  python compare_architectures.py --cnn

  # Single-scale spatial only (legacy 3^3):
  python compare_architectures.py --spatial-kernels 3

  # Disable spatial neighbourhood features:
  python compare_architectures.py --no-spatial
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

from data_loader import (load_all_cubes, cube_to_volumes, get_X_y,
                          get_g0_values, get_feature_cols, add_drop_args, build_drop_set,
                          FEATURE_COLS, LOG_TARGET_COL, compute_data_checksum)
from classical_models import compute_metrics
from cnn_model import UNet3D, count_parameters
from augmentation import augment_cube, get_symmetry_ops
from model_helpers import (
    GLOBAL_SEED, _set_seeds, _get_env_info,
    _XGB_CFG, FlexMLP, _compute_spatial_X, _compute_weights, _preds_to_volume,
    fit_g0_bias_correction, predict_bias, mass_weighted_bias,
    normalize_channels_inplace, normalize_targets_inplace,
)


# ── Shared constants ──────────────────────────────────────────────────────────

CNN_INPUT_COLS        = FEATURE_COLS                    # 15 physical field channels
CNN_TARGET_COL        = LOG_TARGET_COL                  # log10(nH2)
CNN_INPUT_COLS_GUIDED = CNN_INPUT_COLS + ['xgb_pred']  # 16 channels (XGBoost-guided)


# ── XGBoost variant configs ───────────────────────────────────────────────────

XGB_VARIANTS: dict[str, dict] = {
    # Standard depth=6 — confirmed optimal across 3 runs (depth=4 and depth=8 both worse)
    'xgb_standard': _XGB_CFG,
}


# ── CNN guided variant configs ─────────────────────────────────────────────────

CNN_GUIDED_VARIANTS: dict[str, dict] = {
    'unet_xgb_guided': {'base_ch': 32},
}


# ── MLP variant configs ───────────────────────────────────────────────────────

MLP_VARIANTS: dict[str, dict] = {
    # Wide MLP — confirmed best for ensemble/stacking (narrower and residual variants retired)
    'mlp_wide': {'arch': 'flex', 'hidden_dims': [512, 512, 256, 128]},
}


# ── CNN variant configs ───────────────────────────────────────────────────────

CNN_VARIANTS: dict[str, dict] = {
    'unet_small':    {'base_ch': 16},
    'unet_standard': {'base_ch': 32},
    'unet_large':    {'base_ch': 64},
    # Residual U-Net with improved training (base_ch=16, dropout=0.1, lr=5e-4, warmup)
    'unet_residual': {'base_ch': 16, 'dropout': 0.1, 'warmup_epochs': 10},
}


# ── Shared CubeDataset ────────────────────────────────────────────────────────

class CubeDataset(Dataset):
    """Pre-computes all augmented (and optionally pooled) tensors at init time.

    pool=True average-pools 128^3 volumes to 64^3 (the legacy behaviour, for
    constrained VRAM); the default keeps native 128^3 resolution.
    Accepts an optional input_cols list to support the guided CNN variant
    (15-channel input with XGBoost prediction prepended).
    """
    def __init__(self, cube_vols: list[dict],
                 ops: list[np.ndarray] | None,
                 augment: bool = True,
                 input_cols: list[str] | None = None,
                 pool: bool = False):
        _cols = input_cols if input_cols is not None else CNN_INPUT_COLS
        self.xs: list[torch.Tensor] = []
        self.ys: list[torch.Tensor] = []
        identity   = np.eye(3, dtype=int)
        active_ops = ops if (augment and ops) else [identity]
        for vol in cube_vols:
            for R in active_ops:
                aug = augment_cube(vol, R)
                # augment_cube only handles known physical fields; apply the same
                # scalar grid transformation to any extra channels (e.g. 'xgb_pred')
                perm  = [int(np.nonzero(R[i])[0][0]) for i in range(3)]
                signs = [int(R[i, perm[i]])           for i in range(3)]
                for extra in _cols:
                    if extra not in aug and extra in vol:
                        v = np.transpose(vol[extra], perm)
                        for ax, s in enumerate(signs):
                            if s == -1:
                                v = np.flip(v, axis=ax)
                        aug[extra] = np.ascontiguousarray(v)
                channels = np.stack([aug[c] for c in _cols], axis=0)
                target   = aug[CNN_TARGET_COL][None]
                ch_t     = torch.from_numpy(channels).unsqueeze(0)
                tgt_t    = torch.from_numpy(target).unsqueeze(0)
                if pool:
                    ch_t  = F.avg_pool3d(ch_t,  kernel_size=2, stride=2)
                    tgt_t = F.avg_pool3d(tgt_t, kernel_size=2, stride=2)
                self.xs.append(torch.nan_to_num(ch_t.squeeze(0).float()))
                self.ys.append(torch.nan_to_num(tgt_t.squeeze(0).float()))

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]


# ── Training helpers ──────────────────────────────────────────────────────────

def _standardize_inplace(X_tr: np.ndarray, X_va: np.ndarray,
                         chunk: int = 2_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Memory-efficient per-column standardization (float32, in place).

    Equivalent to StandardScaler fit on X_tr / transform on both, but avoids
    sklearn's full-size float64 temporaries (which need >5 GB for the
    14.7M x 56 spatial matrix).  Means/variances are accumulated in float64
    over row chunks; the arrays are modified in place.
    """
    n, d = X_tr.shape
    s  = np.zeros(d, dtype=np.float64)
    s2 = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        c = X_tr[i:i + chunk].astype(np.float64)
        s  += c.sum(axis=0)
        s2 += (c * c).sum(axis=0)
    mu = (s / n)
    sd = np.sqrt(np.maximum(s2 / n - mu * mu, 0.0))
    sd[sd < 1e-8] = 1.0
    mu32 = mu.astype(np.float32)
    sd32 = sd.astype(np.float32)
    for arr in (X_tr, X_va):
        m = len(arr)
        for i in range(0, m, chunk):
            arr[i:i + chunk] -= mu32
            arr[i:i + chunk] /= sd32
    return X_tr, X_va


def run_xgb_cv(variant_name: str,
               config: dict,
               X: np.ndarray,
               y: np.ndarray,
               fold_labels: np.ndarray,
               g0_values: list[float],
               cubes: list,
               weighted: bool = False,
               ) -> tuple[list[dict], list[np.ndarray], list[np.ndarray],
                          list[np.ndarray], list[list[float]]]:
    """7-fold leave-one-G0-out CV for one XGBoost config.

    Returns (fold_metrics, y_true_folds, y_pred_folds, xgb_vols_128,
    importances) where importances[fold] is the model's per-feature gain
    importance vector (recorded in the run log for the feature-importance
    figure).  xgb_vols_128[i] is the 128^3 log_nH2 prediction volume for
    cube i from the fold where it was held out — fully OOB, no data leakage.
    """
    _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg  = {**config, 'device': _dev}
    fold_metrics: list[dict]        = []
    y_true_folds: list[np.ndarray]  = []
    y_pred_folds: list[np.ndarray]  = []
    xgb_vols:     list[np.ndarray]  = []
    importances:  list[list[float]] = []
    for fold in range(len(g0_values)):
        _set_seeds(GLOBAL_SEED + fold)
        mask   = fold_labels != fold
        X_tr, y_tr = X[mask],  y[mask]
        X_va, y_va = X[~mask], y[~mask]
        X_tr_s, X_va_s = _standardize_inplace(X_tr, X_va)
        model = xgb.XGBRegressor(**cfg)
        sw = _compute_weights(y_tr) if weighted else None
        model.fit(X_tr_s, y_tr, sample_weight=sw,
                  eval_set=[(X_va_s, y_va)], verbose=False)
        y_pred = model.predict(X_va_s).astype(np.float32)
        fold_metrics.append(compute_metrics(y_va, y_pred))
        y_true_folds.append(y_va.astype(np.float32))
        y_pred_folds.append(y_pred)
        xgb_vols.append(_preds_to_volume(cubes[fold], y_pred))
        importances.append([float(v) for v in model.feature_importances_])
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")
    return fold_metrics, y_true_folds, y_pred_folds, xgb_vols, importances


def _build_mlp(config: dict, in_dim: int) -> nn.Module:
    """Dispatch to the appropriate MLP class based on config['arch']."""
    if config['arch'] == 'flex':
        return FlexMLP(in_dim, config['hidden_dims'])
    raise ValueError(f"Unknown arch: {config['arch']!r}")


def run_mlp_cv(variant_name: str,
               config: dict,
               X: np.ndarray,
               y: np.ndarray,
               fold_labels: np.ndarray,
               g0_values: list[float],
               epochs: int = 100,
               batch_size: int = 262144,
               lr: float = 1e-3,
               weighted: bool = False,
               ) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """7-fold CV for one MLP config.

    Training loop is identical to classical_models.run_mlp (GPU-preload,
    torch.randperm batching, AMP, CosineAnnealingLR) but uses a configurable
    model architecture.

    Returns (fold_metrics, y_true_folds, y_pred_folds).
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    fold_metrics: list[dict]       = []
    y_true_folds: list[np.ndarray] = []
    y_pred_folds: list[np.ndarray] = []

    for fold in range(len(g0_values)):
        _set_seeds(GLOBAL_SEED + fold)
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

        w_tr_t = (torch.from_numpy(_compute_weights(y_tr.astype(np.float32))).to(device)
                  if weighted else None)

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
                    if weighted:
                        wb   = w_tr_t[perm[i : i + batch_size]]
                        loss = (wb * (model(xb) - yb) ** 2).mean()
                    else:
                        loss = loss_fn(model(xb), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
            sched.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            y_pred = model(torch.from_numpy(X_va_s).to(device)).float().cpu().numpy()

        # Clamp to training range ±2 dex — prevents linear-space explosion from
        # the few voxels where the unbounded MLP extrapolates wildly.
        y_pred = np.clip(y_pred, float(y_tr.min()) - 2.0, float(y_tr.max()) + 2.0)

        fold_metrics.append(compute_metrics(y_va, y_pred))
        y_true_folds.append(y_va.astype(np.float32))
        y_pred_folds.append(y_pred.astype(np.float32))
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")

    return fold_metrics, y_true_folds, y_pred_folds


def _train_cnn_fold(train_vols: list[dict],
                    val_vols: list[dict],
                    ops: list[np.ndarray],
                    device: torch.device,
                    epochs: int,
                    lr: float,
                    base_ch: int,
                    input_cols: list[str] | None = None,
                    seed: int = 0,
                    dropout: float = 0.0,
                    warmup_epochs: int = 0,
                    pool: bool = False,
                    ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Single CNN fold.  Mirrors train_cnn.train_one_fold but parametrizes
    base_ch, dropout, and accepts an optional input_cols list.

    pool=True trains on 64^3 average-pooled volumes (legacy behaviour);
    the default is native 128^3.

    seed fixes torch/numpy randomness so results are reproducible across runs.
    Pass fold index as seed to get deterministic but fold-varied initialisation.

    Returns (metrics_dict, y_true_log, y_pred_log) in original log_nH2 space.
    """
    _set_seeds(GLOBAL_SEED + seed)
    _cols    = input_cols if input_cols is not None else CNN_INPUT_COLS
    train_ds = CubeDataset(train_vols, ops, augment=True,  input_cols=_cols,
                           pool=pool)
    val_ds   = CubeDataset(val_vols,   ops=None, augment=False, input_cols=_cols,
                           pool=pool)

    # Fold-safe normalisation, streaming + in place (a full-dataset stack
    # would be a ~6 GB temporary at 128^3)
    normalize_channels_inplace(train_ds.xs, val_ds.xs)
    y_mean, y_std = normalize_targets_inplace(train_ds.ys, val_ds.ys)

    pin_mem  = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          num_workers=0, pin_memory=pin_mem)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=pin_mem)

    model   = UNet3D(n_channels=len(_cols), base_ch=base_ch, dropout=dropout).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # Linear warmup then cosine decay (warmup_epochs=0 → pure cosine)
    def _lr_lambda(ep: int) -> float:
        if ep < warmup_epochs:
            return (ep + 1) / warmup_epochs
        progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched   = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
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

    # Restore best checkpoint and evaluate in original log_nH2 space
    model.load_state_dict(best_state)
    model.eval()
    y_true_parts, y_pred_parts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            y_pred_parts.append(model(xb.to(device)).float().cpu().numpy().ravel())
            y_true_parts.append(yb.numpy().ravel())

    y_true = np.concatenate(y_true_parts) * y_std + y_mean
    y_pred = np.concatenate(y_pred_parts) * y_std + y_mean
    return compute_metrics(y_true, y_pred), y_true.astype(np.float32), y_pred.astype(np.float32)


def run_cnn_cv_variant(variant_name: str,
                       config: dict,
                       all_vols: list[dict],
                       g0_values: list[float],
                       device: torch.device,
                       ops: list[np.ndarray],
                       epochs: int = 200,
                       lr: float = 5e-4,
                       pool: bool = False,
                       input_cols: list[str] | None = None,
                       ) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """7-fold CV for one CNN config variant.
    Returns (fold_metrics, y_true_folds, y_pred_folds).
    """
    base_ch      = config['base_ch']
    dropout      = config.get('dropout', 0.0)
    warmup_ep    = config.get('warmup_epochs', 0)
    _cols        = input_cols if input_cols is not None else CNN_INPUT_COLS
    n_params = count_parameters(UNet3D(n_channels=len(_cols),
                                       base_ch=base_ch, dropout=dropout))
    print(f"  {variant_name}  base_ch={base_ch}  dropout={dropout}  "
          f"params={n_params:,}  grid={64 if pool else 128}^3")
    fold_metrics: list[dict]       = []
    y_true_folds: list[np.ndarray] = []
    y_pred_folds: list[np.ndarray] = []
    for fold in range(len(g0_values)):
        print(f"  [Fold {fold + 1}/{len(g0_values)}] Val G0={g0_values[fold]:.1f}")
        train_vols = [v for i, v in enumerate(all_vols) if i != fold]
        val_vols   = [all_vols[fold]]
        metrics, y_true, y_pred = _train_cnn_fold(
            train_vols, val_vols, ops, device, epochs, lr, base_ch,
            input_cols=_cols, seed=fold, dropout=dropout,
            warmup_epochs=warmup_ep, pool=pool)
        fold_metrics.append(metrics)
        y_true_folds.append(y_true)
        y_pred_folds.append(y_pred)
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={metrics['R2']:.4f}")
    return fold_metrics, y_true_folds, y_pred_folds


def _align_preds(all_preds: dict,
                 cubes: list,
                 g0_values: list[float],
                 force_pool: bool = False) -> dict:
    """Return a copy of all_preds with every model's per-fold predictions in
    one common, cell-aligned raveled-volume space.

    CNN variants (names starting with 'unet_') predict in raveled-volume
    order: 64^3 when trained with --cnn-downsample, otherwise native 128^3.
    XGBoost/MLP predictions are flat 128^3 vectors in DataFrame row order;
    they are mapped to volumes via _preds_to_volume so that per-cell
    alignment is guaranteed before predictions are averaged or stacked, and
    avg-pooled to 64^3 only when some model lives in the pooled space (or
    force_pool=True).
    """
    n64 = 64 ** 3   # 262 144
    to_pooled = force_pool or any(len(yp[0]) == n64
                                  for _, yp in all_preds.values())
    out = {}
    for name, (yt_folds, yp_folds) in all_preds.items():
        vol_order = name.startswith('unet_')   # already raveled-volume order
        if len(yp_folds[0]) == n64 or (vol_order and not to_pooled):
            out[name] = (yt_folds, yp_folds)
        else:
            yt_c, yp_c = [], []
            for fold in range(len(g0_values)):
                def _conv(flat, fold=fold):
                    vol = (flat.reshape(128, 128, 128) if vol_order
                           else _preds_to_volume(cubes[fold], flat))
                    if not to_pooled:
                        return vol.ravel()
                    t = torch.from_numpy(
                        np.ascontiguousarray(vol)).unsqueeze(0).unsqueeze(0)
                    return F.avg_pool3d(t, kernel_size=2, stride=2).numpy().ravel()
                yt_c.append(_conv(yt_folds[fold]))
                yp_c.append(_conv(yp_folds[fold]))
            out[name] = (yt_c, yp_c)
    return out


def run_ensemble_cv(ensemble_name: str,
                    model_names: list[str],
                    all_preds: dict[str, tuple[list, list]],
                    g0_values: list[float]) -> list[dict]:
    """Equal-weight average ensemble of per-fold predictions from listed models.

    all_preds[name] = (y_true_folds, y_pred_folds) in log_nH2 space.
    All prediction arrays must be cell-aligned with the same length per fold
    (call _align_preds first when mixing pointwise and CNN models).
    y_true is taken from the first model (all val cubes share the same target).
    """
    fold_metrics: list[dict] = []
    for fold in range(len(g0_values)):
        y_true_log = all_preds[model_names[0]][0][fold]
        y_preds    = [all_preds[name][1][fold] for name in model_names]
        y_ens      = np.mean(y_preds, axis=0)
        fold_metrics.append(compute_metrics(y_true_log, y_ens))
        print(f"  {ensemble_name:<20}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")
    return fold_metrics


def run_stacked_ensemble_cv(ensemble_name: str,
                             model_names: list[str],
                             aligned_preds: dict[str, tuple[list, list]],
                             g0_values: list[float]) -> dict[str, list[dict]]:
    """Ridge-regression stacked ensemble trained on OOF predictions (no leakage).

    For each held-out fold i:
      - Meta-train: stack per-cell OOF predictions from the 6 other folds.
        X_meta_tr shape (6 * n_cells, n_models); y_meta_tr = ground-truth log_nH2.
      - Fit Ridge(alpha=1.0) on meta-train, then predict on fold i's predictions.
      - Reported weights show which model dominates for each fold.

    G0-linear recalibration, two flavours (both leakage-free — fitted on
    per-training-cube OOF biases vs log10(G0), evaluated at the held-out G0):
      '_cal'    unweighted per-cube mean residual — centres the *cell-mean*
                log bias.  The Ridge intercept already zeroes the pooled OOF
                residual, so this mostly corrects the G0 trend of the mean.
      '_mwcal'  mass-weighted per-cube residual (mass_weighted_bias) —
                centres the *total-H2-mass* budget, which the cell-mean
                correction cannot move when the surviving error is
                density-dependent (dense cells under-predicted at high G0).

    Returns {ensemble_name, ensemble_name + '_cal', ensemble_name + '_mwcal'}
    -> fold_metrics for the raw and recalibrated predictions respectively.

    aligned_preds must already be in a common cell-aligned volume space
    (call _align_preds first).
    """
    n_folds = len(g0_values)
    fold_metrics:       list[dict] = []
    fold_metrics_cal:   list[dict] = []
    fold_metrics_mwcal: list[dict] = []
    for fold in range(n_folds):
        y_true_val = aligned_preds[model_names[0]][0][fold]

        X_meta_tr_parts, y_meta_tr_parts, g0_tr = [], [], []
        for j in range(n_folds):
            if j == fold:
                continue
            yt_j = aligned_preds[model_names[0]][0][j]
            yp_j = np.stack([aligned_preds[m][1][j] for m in model_names], axis=1)
            X_meta_tr_parts.append(yp_j)
            y_meta_tr_parts.append(yt_j)
            g0_tr.append(g0_values[j])

        X_meta_tr  = np.concatenate(X_meta_tr_parts, axis=0)   # (6*n_cells, n_models)
        y_meta_tr  = np.concatenate(y_meta_tr_parts, axis=0)   # (6*n_cells,)
        X_meta_val = np.stack([aligned_preds[m][1][fold] for m in model_names], axis=1)

        meta = Ridge(alpha=1.0, fit_intercept=True)
        meta.fit(X_meta_tr, y_meta_tr)
        y_stacked = meta.predict(X_meta_val).astype(np.float32)

        # Per-training-cube OOF biases of the fitted meta-learner -> G0 trends
        oof_preds = [meta.predict(Xp) for Xp in X_meta_tr_parts]
        per_cube_bias = np.array([
            float(np.mean(pp - yp))
            for pp, yp in zip(oof_preds, y_meta_tr_parts)
        ])
        per_cube_bias_mw = np.array([
            mass_weighted_bias(pp, yp)
            for pp, yp in zip(oof_preds, y_meta_tr_parts)
        ])
        slope, intercept = fit_g0_bias_correction(per_cube_bias,
                                                  np.array(g0_tr))
        offset = predict_bias(slope, intercept, g0_values[fold])
        y_stacked_cal = (y_stacked - offset).astype(np.float32)

        slope_mw, intercept_mw = fit_g0_bias_correction(per_cube_bias_mw,
                                                        np.array(g0_tr))
        offset_mw = predict_bias(slope_mw, intercept_mw, g0_values[fold])
        y_stacked_mwcal = (y_stacked - offset_mw).astype(np.float32)

        metrics       = compute_metrics(y_true_val, y_stacked)
        metrics_cal   = compute_metrics(y_true_val, y_stacked_cal)
        metrics_mwcal = compute_metrics(y_true_val, y_stacked_mwcal)
        fold_metrics.append(metrics)
        fold_metrics_cal.append(metrics_cal)
        fold_metrics_mwcal.append(metrics_mwcal)
        weights = {m: f"{w:.3f}" for m, w in zip(model_names, meta.coef_)}
        print(f"  {ensemble_name:<24}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={metrics['R2']:.4f}  massR={metrics['mass_ratio']:.3f}  "
              f"w={weights}")
        print(f"  {ensemble_name + '_cal':<24}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={metrics_cal['R2']:.4f}  massR={metrics_cal['mass_ratio']:.3f}  "
              f"offset={offset:+.3f} dex")
        print(f"  {ensemble_name + '_mwcal':<24}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={metrics_mwcal['R2']:.4f}  massR={metrics_mwcal['mass_ratio']:.3f}  "
              f"offset={offset_mw:+.3f} dex")
    return {ensemble_name: fold_metrics,
            ensemble_name + '_cal': fold_metrics_cal,
            ensemble_name + '_mwcal': fold_metrics_mwcal}


def run_cnn_cv_guided(variant_name: str,
                      config: dict,
                      all_vols: list[dict],
                      xgb_vols: list[np.ndarray],
                      g0_values: list[float],
                      device: torch.device,
                      ops: list[np.ndarray],
                      epochs: int = 150,
                      lr: float = 1e-3,
                      input_cols_base: list[str] | None = None,
                      pool: bool = False,
                      ) -> tuple[list[dict], list[np.ndarray], list[np.ndarray]]:
    """CNN fold training where XGBoost's OOB predictions are the 15th input channel.

    xgb_vols[i] is the 128^3 prediction volume for cube i from its own held-out
    XGBoost fold.  This is fully OOB for both training and val cubes, so there
    is no data leakage.

    Returns (fold_metrics, y_true_folds, y_pred_folds).
    """
    _base_cols  = input_cols_base if input_cols_base is not None else CNN_INPUT_COLS
    _guided_cols = _base_cols + ['xgb_pred']
    base_ch  = config['base_ch']
    n_params = count_parameters(UNet3D(n_channels=len(_guided_cols), base_ch=base_ch))
    print(f"  {variant_name}  base_ch={base_ch}  n_channels={len(_guided_cols)}  params={n_params:,}")
    fold_metrics: list[dict]       = []
    y_true_folds: list[np.ndarray] = []
    y_pred_folds: list[np.ndarray] = []
    for fold in range(len(g0_values)):
        print(f"  [Fold {fold + 1}/{len(g0_values)}] Val G0={g0_values[fold]:.1f}")
        # Inject 'xgb_pred' channel into each cube's volume dict
        train_vols_g = [{**all_vols[i], 'xgb_pred': xgb_vols[i]}
                        for i in range(len(all_vols)) if i != fold]
        val_vols_g   = [{**all_vols[fold], 'xgb_pred': xgb_vols[fold]}]
        metrics, y_true, y_pred = _train_cnn_fold(
            train_vols_g, val_vols_g, ops, device, epochs, lr, base_ch,
            input_cols=_guided_cols, seed=fold, pool=pool)
        fold_metrics.append(metrics)
        y_true_folds.append(y_true)
        y_pred_folds.append(y_pred)
        print(f"  {variant_name:<16}  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={metrics['R2']:.4f}")
    return fold_metrics, y_true_folds, y_pred_folds


# ── Reporting ─────────────────────────────────────────────────────────────────

def add_skill_scores(all_results: dict[str, list[dict]],
                     all_preds: dict[str, tuple[list, list]],
                     cubes: list,
                     g0_values: list[float],
                     ens_preds: dict | None = None,
                     ref_name: str = 'xgb_standard') -> None:
    """Add per-fold 'skill_vs_xgb' = 1 - MSE_model / MSE_ref to every variant.

    The reference is the pointwise XGBoost baseline (ref_name) — a meaningful
    baseline, unlike the fold-mean prediction implicit in R2's denominator.

    Error metrics are permutation-invariant, so at native resolution one
    reference MSE (computed over all 128^3 cells) applies to every variant
    regardless of cell ordering.  Only variants evaluated on *pooled* 64^3
    volumes (--cnn-downsample runs and the ensembles built on them) see a
    different target field, and are scored against the reference's
    predictions pooled to 64^3.  Mutates the metric dicts in place; a no-op
    with a console note if the reference was not run.
    """
    if ref_name not in all_results:
        print(f"\n(Skill scores skipped: reference '{ref_name}' not run)")
        return
    n64 = 64 ** 3
    ens_pooled = (ens_preds is not None and len(ens_preds) > 0 and
                  len(next(iter(ens_preds.values()))[1][0]) == n64)

    def _pooled(name: str) -> bool:
        if name in all_preds:
            return len(all_preds[name][1][0]) == n64
        # ens_*/stacked_* variants live in the aligned ensemble space
        return ens_pooled

    ref128 = [m['RMSE'] ** 2 for m in all_results[ref_name]]
    ref64: list[float] | None = None
    if any(_pooled(n) for n in all_results):
        if ens_pooled and ref_name in ens_preds:
            yt64, yp64 = ens_preds[ref_name]
        else:
            yt64, yp64 = _align_preds({ref_name: all_preds[ref_name]},
                                      cubes, g0_values,
                                      force_pool=True)[ref_name]
        ref64 = [compute_metrics(yt64[f], yp64[f], fast=True)['RMSE'] ** 2
                 for f in range(len(g0_values))]
    for name, fms in all_results.items():
        ref = ref64 if (_pooled(name) and ref64 is not None) else ref128
        for m, mse_ref in zip(fms, ref):
            m['skill_vs_xgb'] = 1.0 - (m['RMSE'] ** 2) / mse_ref


def print_comparison(all_results: dict[str, list[dict]],
                     g0_values: list[float]) -> None:
    """Print a compact comparison table: variants (rows) x G0 folds (cols),
    followed by cross-fold means of the headline metrics."""
    name_w  = 20
    g0_hdr  = "".join(f"G0={g:<5.1f}" for g in g0_values)
    header  = (f"  {'Variant':<{name_w}}  {g0_hdr}  {'MeanR2':>8}  {'Std':>6}  "
               f"{'RMSE':>7}  {'bias':>7}  {'massR':>6}  {'CCC':>7}  {'W1':>6}  "
               f"{'skill':>7}")
    sep     = "=" * len(header)

    print(f"\n{sep}")
    print("  Architecture Comparison  (R2 log-space per fold; "
          "trailing columns are cross-fold means)")
    print(sep)
    print(header)
    print("-" * len(header))

    for name, fms in all_results.items():
        r2s     = [m['R2'] for m in fms]
        r2_cols = "".join(f"{r2:>+8.4f} " for r2 in r2s)
        def _mean(key: str) -> float:
            return float(np.nanmean([m.get(key, np.nan) for m in fms]))
        print(f"  {name:<{name_w}}  {r2_cols}  {np.mean(r2s):>+8.4f}  "
              f"{np.std(r2s):>6.4f}  {_mean('RMSE'):>7.3f}  {_mean('bias'):>+7.3f}  "
              f"{_mean('mass_ratio'):>6.3f}  {_mean('CCC'):>7.4f}  "
              f"{_mean('W1'):>6.3f}  {_mean('skill_vs_xgb'):>+7.3f}")

    print(sep)


def save_comparison_log(all_results: dict[str, list[dict]],
                        g0_values: list[float],
                        run_config: dict,
                        log_path: str,
                        xgb_importances: dict[str, dict] | None = None) -> None:
    out: dict = {'run_config': run_config, 'g0_values': g0_values, 'variants': {}}
    for name, fms in all_results.items():
        out['variants'][name] = {
            'folds': [
                {'fold': i, 'g0': g0_values[i],
                 'metrics': {k: float(v) for k, v in m.items()}}
                for i, m in enumerate(fms)
            ],
            'summary': {
                # nan-safe: phase-conditional metrics are NaN when a phase
                # has fewer than 10 cells in a fold
                key: {'mean': float(np.nanmean([m[key] for m in fms])),
                      'std':  float(np.nanstd([m[key] for m in fms]))}
                for key in fms[0].keys()
            },
        }
        # Per-fold XGBoost feature importances (for plot_feature_importance.py)
        if xgb_importances and name in xgb_importances:
            out['variants'][name]['xgb_feature_importance'] = \
                xgb_importances[name]
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
    parser.add_argument('--cnn',        action='store_true',
                        help='Include CNN variants (slow; off by default)')
    parser.add_argument('--cnn-epochs', type=int, default=150,
                        help='CNN epochs per variant/fold (default 150)')
    parser.add_argument('--cnn-downsample', action='store_true',
                        help='Average-pool CNN volumes 128^3 -> 64^3 (legacy '
                             'behaviour for constrained VRAM; default trains '
                             'at native 128^3)')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='MLP epochs per variant/fold (default 100)')
    parser.add_argument('--all-ops',    action='store_true',
                        help='Use all 48 Oh symmetry ops for CNN (default: 8 z-preserving)')
    parser.add_argument('--no-spatial', action='store_false', dest='spatial',
                        help='Disable spatial neighbourhood-mean feature variants '
                             '(enabled by default when volumes are loaded)')
    parser.add_argument('--spatial-kernels', nargs='+', type=int, default=[3, 5, 7],
                        help='Kernel sizes for multi-scale spatial features '
                             '(default: 3 5 7 -> 45 spatial features)')
    parser.set_defaults(spatial=True)
    parser.add_argument('--log',        type=str, default=None,
                        help='Output JSON path '
                             '(default: arch_comparison_TIMESTAMP.json)')
    add_drop_args(parser)
    args = parser.parse_args()

    _drop = build_drop_set(args)
    feat_cols = get_feature_cols(_drop)

    _set_seeds(GLOBAL_SEED)
    env_info = _get_env_info()

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = args.log or f'arch_comparison_{ts}.json'

    print("Loading data...")
    cubes         = load_all_cubes()
    g0_vals       = get_g0_values(cubes)
    X, y, folds   = get_X_y(cubes, use_log_target=True, feature_cols=feat_cols)
    print(f"Total samples: {len(X):,}  |  Features: {X.shape[1]}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Computing data checksum...")
    data_checksum = compute_data_checksum()

    run_config = {
        'timestamp':        datetime.datetime.now().isoformat(timespec='seconds'),
        'global_seed':      GLOBAL_SEED,
        'device':           str(device),
        'run_cnn':          args.cnn,
        'cnn_epochs':       args.cnn_epochs,
        'cnn_grid':         64 if args.cnn_downsample else 128,
        'mlp_epochs':       args.mlp_epochs,
        'all_ops':          args.all_ops,
        'spatial':          args.spatial,
        'spatial_kernels':  args.spatial_kernels,
        'dropped_features': sorted(_drop),
        'feature_cols':     feat_cols,
        'n_features':       len(feat_cols),
        'n_samples':        int(len(X)),
        'n_folds':          len(g0_vals),
        'g0_values':        g0_vals,
        'xgb_config':       dict(_XGB_CFG),
        'mlp_config':       MLP_VARIANTS,
        'env':              env_info,
        'data':             data_checksum,
    }

    # Load 128^3 volumes whenever CNN variants or spatial features are needed
    need_vols = args.cnn or args.spatial
    all_vols: list[dict] | None = None
    ops: list | None = None
    if need_vols:
        print("Converting cubes to 128^3 volumes...")
        vol_cols = [c for c in feat_cols + [CNN_TARGET_COL]
                    if c in cubes[0].columns]
        all_vols = [cube_to_volumes(df, vol_cols) for df in cubes]
        ops      = get_symmetry_ops(safe_only=not args.all_ops)
        print(f"Symmetry ops: {len(ops)}  "
              f"({'z-preserving' if not args.all_ops else 'full Oh'})")

    all_results:       dict[str, list[dict]]           = {}
    all_preds:         dict[str, tuple[list, list]]    = {}
    xgb_importances:   dict[str, dict]                 = {}
    xgb_standard_vols: list[np.ndarray] | None         = None

    # ── XGBoost variants ──────────────────────────────────────────────────────
    if not args.skip_xgb:
        print("\n--- XGBoost ---")
        for name, cfg in XGB_VARIANTS.items():
            print(f"\n[{name}]")
            fold_metrics, yt, yp, xvols, fi = run_xgb_cv(
                name, cfg, X, y, folds, g0_vals, cubes)
            all_results[name]     = fold_metrics
            all_preds[name]       = (yt, yp)
            xgb_importances[name] = {'feature_names': list(feat_cols),
                                     'per_fold': fi}
            if name == 'xgb_standard':
                xgb_standard_vols = xvols

    # ── MLP variants ──────────────────────────────────────────────────────────
    if not args.skip_mlp:
        print(f"\n--- MLP ({args.mlp_epochs} epochs) ---")
        for name, cfg in MLP_VARIANTS.items():
            print(f"\n[{name}]")
            fold_metrics, yt, yp = run_mlp_cv(
                name, cfg, X, y, folds, g0_vals, epochs=args.mlp_epochs)
            all_results[name] = fold_metrics
            all_preds[name]   = (yt, yp)

    # ── CNN variants ──────────────────────────────────────────────────────────
    if args.cnn:
        print(f"\n--- CNN variants ({args.cnn_epochs} epochs) ---")
        for name, cfg in CNN_VARIANTS.items():
            print(f"\n[{name}]")
            fold_metrics, yt, yp = run_cnn_cv_variant(
                name, cfg, all_vols, g0_vals, device, ops,
                epochs=args.cnn_epochs, pool=args.cnn_downsample,
                input_cols=feat_cols)
            all_results[name] = fold_metrics
            all_preds[name]   = (yt, yp)

    # ── Spatial-feature variants ───────────────────────────────────────────────
    if args.spatial and need_vols:
        print("\n--- Spatial-feature variants ---")
        X_extra = _compute_spatial_X(cubes, all_vols, feat_cols,
                                     kernel_sizes=tuple(args.spatial_kernels))
        n_sp    = len(feat_cols) * len(args.spatial_kernels)
        X_sp    = np.concatenate([X, X_extra], axis=1)   # (N, 15 + n_sp)
        print(f"  Spatial kernels: {args.spatial_kernels}  ->  {n_sp} spatial features  "
              f"(X_sp shape: {X_sp.shape})")

        # Spatial feature-matrix column names: base features, then k-major
        # blocks matching _compute_spatial_X's concatenation order
        sp_names = list(feat_cols) + [f'{c}_k{k}'
                                      for k in args.spatial_kernels
                                      for c in feat_cols]

        if not args.skip_xgb:
            name = 'xgb_standard_sp'
            fold_metrics, yt, yp, _, fi = run_xgb_cv(
                name, XGB_VARIANTS['xgb_standard'], X_sp, y, folds, g0_vals, cubes)
            all_results[name]     = fold_metrics
            all_preds[name]       = (yt, yp)
            xgb_importances[name] = {'feature_names': sp_names, 'per_fold': fi}

        if not args.skip_mlp:
            name = 'mlp_wide_sp'
            fold_metrics, yt, yp = run_mlp_cv(
                name, MLP_VARIANTS['mlp_wide'], X_sp, y, folds, g0_vals,
                epochs=args.mlp_epochs)
            all_results[name] = fold_metrics
            all_preds[name]   = (yt, yp)

        if not args.skip_xgb and not args.skip_mlp:
            print("\n  [weighted variants]")
            name = 'xgb_standard_sp_w'
            fold_metrics, yt, yp, _, fi = run_xgb_cv(
                name, XGB_VARIANTS['xgb_standard'], X_sp, y, folds, g0_vals,
                cubes, weighted=True)
            all_results[name]     = fold_metrics
            all_preds[name]       = (yt, yp)
            xgb_importances[name] = {'feature_names': sp_names, 'per_fold': fi}

            name = 'mlp_wide_sp_w'
            fold_metrics, yt, yp = run_mlp_cv(
                name, MLP_VARIANTS['mlp_wide'], X_sp, y, folds, g0_vals,
                epochs=args.mlp_epochs, weighted=True)
            all_results[name] = fold_metrics
            all_preds[name]   = (yt, yp)

    # ── XGBoost-guided CNN ────────────────────────────────────────────────────
    if args.cnn and xgb_standard_vols is not None:
        print("\n--- XGBoost-guided CNN ---")
        for name, cfg in CNN_GUIDED_VARIANTS.items():
            print(f"\n[{name}]")
            fold_metrics, yt, yp = run_cnn_cv_guided(
                name, cfg, all_vols, xgb_standard_vols, g0_vals, device, ops,
                epochs=args.cnn_epochs, input_cols_base=feat_cols,
                pool=args.cnn_downsample)
            all_results[name] = fold_metrics
            all_preds[name]   = (yt, yp)
    elif args.cnn and args.skip_xgb:
        print("\n(Skipping XGBoost-guided CNN: requires xgb_standard predictions; "
              "re-run without --skip-xgb to enable)")

    # ── Ensemble variants ──────────────────────────────────────────────────────
    ens_groups = [
        ('ens_xgb+mlp',     ['xgb_standard',    'mlp_wide']),
        ('ens_xgb+cnn',     ['xgb_standard',    'unet_standard']),
        ('ens_all',         ['xgb_standard',    'mlp_wide',    'unet_standard']),
        ('ens_sp',          ['xgb_standard_sp', 'mlp_wide_sp']),
    ]
    stacked_groups = [
        ('stacked_xgb+mlp', ['xgb_standard',      'mlp_wide']),
        ('stacked_xgb+cnn', ['xgb_standard',      'unet_standard']),
        ('stacked_all',     ['xgb_standard',      'mlp_wide',    'unet_standard']),
        ('stacked_sp',      ['xgb_standard_sp',   'mlp_wide_sp']),
        ('stacked_weighted',['xgb_standard_sp_w', 'mlp_wide_sp_w']),
    ]
    ens_to_run     = [(n, ms) for n, ms in ens_groups     if all(m in all_preds for m in ms)]
    stacked_to_run = [(n, ms) for n, ms in stacked_groups if all(m in all_preds for m in ms)]
    ens_preds: dict | None = None
    if ens_to_run or stacked_to_run:
        print("\n--- Ensemble variants ---")
        ens_preds = _align_preds(all_preds, cubes, g0_vals)
        for ens_name, members in ens_to_run:
            all_results[ens_name] = run_ensemble_cv(
                ens_name, members, ens_preds, g0_vals)
        for stk_name, members in stacked_to_run:
            all_results.update(run_stacked_ensemble_cv(
                stk_name, members, ens_preds, g0_vals))

    # ── Skill scores vs the pointwise-XGBoost reference ──────────────────────
    add_skill_scores(all_results, all_preds, cubes, g0_vals, ens_preds)

    print_comparison(all_results, g0_vals)
    save_comparison_log(all_results, g0_vals, run_config, log_path,
                        xgb_importances=xgb_importances)
