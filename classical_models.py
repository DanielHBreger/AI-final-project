"""
classical_models.py
Train and evaluate three pointwise baseline models with leave-one-G0-out CV:
  1. Linear Regression
  2. XGBoost
  3. MLP (PyTorch)

All models predict log10(nH2) from FEATURE_COLS.

Primary metrics (RMSE decomposed into bias + scatter, mass_ratio) are
computed in log10(nH2) space — the natural evaluation space for data
spanning many orders of magnitude.  Secondary metrics include log-space R2,
threshold accuracies (fraction within 0.1/0.3/0.5 dex), phase-conditional
errors (diffuse vs molecular at PHASE_SPLIT), and a clipped linear-space
R2_lin.  Optional agreement metrics: Lin's concordance correlation (CCC)
and the 1-D Wasserstein distance between marginal distributions (W1, dex).
"""

import random
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb

from data_loader import load_all_cubes, get_X_y, get_g0_values, FEATURE_COLS

_SEED = 67


def _set_fold_seed(fold: int) -> None:
    seed = _SEED + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Metrics ──────────────────────────────────────────────────────────────────

# log10(nH2) threshold separating the UV-exposed (diffuse) and self-shielded
# (molecular) populations — same split as merit_metrics.py.
PHASE_SPLIT = -4.0


def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                    fast: bool = False) -> dict:
    """
    Compute metrics for nH2 prediction.  Inputs are log10(nH2) arrays.

    Primary metrics (log space, dex):
      RMSE, decomposed as RMSE^2 = bias^2 + scatter^2 where 'bias' is the
      mean residual (pred - true) and 'scatter' the residual std; MAE; and
      'mass_ratio' = predicted / true total nH2 (uniform grid, so the
      cell-volume factor cancels and this equals the H2 mass ratio).  R2 is
      insensitive to a small uniform offset against the 16-dex target range,
      but that offset compounds exponentially in the mass budget — bias and
      mass_ratio expose it.

    Secondary metrics:
      R2 (log space; fold-variance-normalised, kept for literature
      comparability), threshold accuracies frac_01/frac_03/frac_05
      (fraction of cells with |residual| below 0.1/0.3/0.5 dex),
      phase-conditional R2/RMSE/bias for the molecular (true > PHASE_SPLIT)
      and diffuse populations, f_mol (molecular cell fraction), MAE_mw
      (H2-mass-weighted MAE, dex), and R2_lin: R2 in original nH2 space with
      predictions clipped to [min_true - 1 dex, max_true + 1 dex] before
      exponentiation to prevent numerical explosion from outlier
      neural-net predictions.

    Agreement metrics:
      CCC — Lin's concordance correlation coefficient (agreement with the
      identity line; penalises bias and scale errors that R2 forgives);
      W1 — 1-D Wasserstein distance between the true and predicted marginal
      distributions (dex).

    Skill scores against a reference model are cross-model quantities and
    are added at the comparison level (see compare_architectures.py).

    fast=True computes only the cheap log-space metrics (R2, RMSE, MAE,
    bias, scatter) — for in-training progress logging.

    Phase-conditional values are NaN when a phase has fewer than 10 cells;
    aggregate with np.nanmean/np.nanstd.
    """
    yt = np.asarray(y_true_log, dtype=np.float64).ravel()
    yp = np.asarray(y_pred_log, dtype=np.float64).ravel()
    resid = yp - yt

    # Primary: log-space metrics (RMSE^2 = bias^2 + scatter^2)
    r2      = float(r2_score(yt, yp))
    rmse    = float(np.sqrt(np.mean(resid ** 2)))
    mae     = float(np.mean(np.abs(resid)))
    bias    = float(np.mean(resid))
    scatter = float(np.std(resid))

    out = {
        'R2':      r2,
        'RMSE':    rmse,
        'MAE':     mae,
        'bias':    bias,
        'scatter': scatter,
    }
    if fast:
        return out

    # Linear-space R2 and mass budget (clipped predictions)
    clip_lo = float(yt.min()) - 1.0
    clip_hi = float(yt.max()) + 1.0
    y_pred_clip = np.clip(yp, clip_lo, clip_hi)
    y_true_lin  = 10.0 ** yt
    y_pred_lin  = 10.0 ** y_pred_clip
    r2_lin      = float(r2_score(y_true_lin, y_pred_lin))
    mass_ratio  = float(y_pred_lin.sum() / y_true_lin.sum())

    # Threshold accuracies and mass-weighted MAE
    ae      = np.abs(resid)
    frac_01 = float(np.mean(ae < 0.1))
    frac_03 = float(np.mean(ae < 0.3))
    frac_05 = float(np.mean(ae < 0.5))
    mae_mw  = float(np.sum(y_true_lin * ae) / np.sum(y_true_lin))

    # Lin's concordance correlation coefficient
    mt, mp = float(np.mean(yt)), float(np.mean(yp))
    vt, vp = float(np.var(yt)), float(np.var(yp))
    cov    = float(np.mean((yt - mt) * (yp - mp)))
    ccc    = 2.0 * cov / (vt + vp + (mt - mp) ** 2)

    # 1-D Wasserstein distance between marginal distributions (dex)
    w1 = float(wasserstein_distance(yt, yp))

    # Phase-conditional metrics (diffuse vs molecular at PHASE_SPLIT)
    def _phase_metrics(mask: np.ndarray) -> tuple[float, float, float]:
        if mask.sum() < 10:
            return float('nan'), float('nan'), float('nan')
        t, r = yt[mask], resid[mask]
        ss = float(np.sum((t - t.mean()) ** 2))
        r2_ph = 1.0 - float(np.sum(r ** 2)) / ss if ss > 0 else float('nan')
        return r2_ph, float(np.sqrt(np.mean(r ** 2))), float(np.mean(r))

    mol = yt > PHASE_SPLIT
    r2_mol, rmse_mol, bias_mol = _phase_metrics(mol)
    r2_dif, rmse_dif, bias_dif = _phase_metrics(~mol)

    out.update({
        'R2_lin':     r2_lin,
        'mass_ratio': mass_ratio,
        'MAE_mw':     mae_mw,
        'frac_01':    frac_01,
        'frac_03':    frac_03,
        'frac_05':    frac_05,
        'CCC':        ccc,
        'W1':         w1,
        'R2_mol':     r2_mol,
        'RMSE_mol':   rmse_mol,
        'bias_mol':   bias_mol,
        'R2_dif':     r2_dif,
        'RMSE_dif':   rmse_dif,
        'bias_dif':   bias_dif,
        'f_mol':      float(np.mean(mol)),
    })
    return out


def print_results(name: str, fold_metrics: list[dict], g0_values: list[float]) -> None:
    cols = [
        # (header, key, format-width.precision, signed)
        ('R2',    'R2',         '8.4f'),
        ('RMSE',  'RMSE',       '8.4f'),
        ('MAE',   'MAE',        '8.4f'),
        ('bias',  'bias',       '+8.3f'),
        ('scat',  'scatter',    '8.3f'),
        ('massR', 'mass_ratio', '8.3f'),
        ('f0.3',  'frac_03',    '8.3f'),
        ('CCC',   'CCC',        '8.4f'),
        ('W1',    'W1',         '8.3f'),
    ]
    width = 10 + 9 * len(cols)
    print(f"\n{'='*width}")
    print(f"  {name}")
    print(f"{'='*width}")
    hdr = ''.join(f" {h:>8}" for h, _, _ in cols)
    print(f"  {'G0':<8}{hdr}")
    print(f"  {'-'*(width - 4)}")
    for g0, m in zip(g0_values, fold_metrics):
        row = ''.join(f" {m[k]:>{fmt}}" for _, k, fmt in cols)
        print(f"  {g0:<8.1f}{row}")
    print(f"  {'-'*(width - 4)}")
    for label, agg in (('Mean', np.nanmean), ('Std', np.nanstd)):
        row = ''.join(
            f" {agg([m[k] for m in fold_metrics]):>{fmt}}" for _, k, fmt in cols)
        print(f"  {label:<8}{row}")


# ── Linear Regression ─────────────────────────────────────────────────────────

def run_linear(X: np.ndarray, y: np.ndarray,
               folds: np.ndarray, g0_values: list[float]) -> list[dict]:
    fold_metrics = []
    n_folds = len(g0_values)
    for fold in range(n_folds):
        train_mask = folds != fold
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[~train_mask], y[~train_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)

        model = LinearRegression()
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_va_s)

        fold_metrics.append(compute_metrics(y_va, y_pred))
        print(f"  Linear  fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R²={fold_metrics[-1]['R2']:.4f}")

    print_results("Linear Regression", fold_metrics, g0_values)
    return fold_metrics


# ── XGBoost ───────────────────────────────────────────────────────────────────

def run_xgboost(X: np.ndarray, y: np.ndarray,
                folds: np.ndarray, g0_values: list[float],
                subsample: float = 0.3) -> tuple[list[dict], list]:
    """
    subsample: fraction of training data used per tree (speeds up 15M-row training).
    Also returns the trained models for feature importance plotting.
    """
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  XGBoost device: {_device}")
    fold_metrics = []
    models = []
    n_folds = len(g0_values)
    for fold in range(n_folds):
        _set_fold_seed(fold)
        train_mask = folds != fold
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[~train_mask], y[~train_mask]

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=subsample,
            colsample_bytree=0.8,
            tree_method='hist',   # works for both CPU and CUDA
            device=_device,
            random_state=67,
            verbosity=0,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  verbose=False)
        y_pred = model.predict(X_va)

        fold_metrics.append(compute_metrics(y_va, y_pred))
        models.append(model)
        print(f"  XGBoost fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R²={fold_metrics[-1]['R2']:.4f}")

    print_results("XGBoost", fold_metrics, g0_values)
    return fold_metrics, models


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (256, 256, 128, 64)):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def run_mlp(X: np.ndarray, y: np.ndarray,
            folds: np.ndarray, g0_values: list[float],
            epochs: int = 30,
            batch_size: int = 262144,
            lr: float = 1e-3) -> list[dict]:

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f"  MLP device: {device}  AMP: {use_amp}")
    fold_metrics = []
    n_folds = len(g0_values)

    for fold in range(n_folds):
        _set_fold_seed(fold)
        train_mask = folds != fold
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[~train_mask], y[~train_mask]

        # Normalise with training stats
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)

        # Preload entire training set to GPU once — eliminates per-batch transfer overhead.
        # Training data per fold is ~756MB (fits in VRAM on any modern GPU).
        # On CPU, tensors stay in RAM and are transferred per batch as normal.
        X_tr_t = torch.from_numpy(X_tr_s).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)

        n_tr = len(X_tr_t)

        model      = MLP(n_features=X.shape[1]).to(device)
        opt        = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn    = nn.MSELoss()
        scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)

        for epoch in range(epochs):
            model.train()
            # torch.randperm runs on device — no Python/DataLoader overhead between batches
            perm = torch.randperm(n_tr, device=device)
            train_loss, n_batches = 0.0, 0
            for i in range(0, n_tr, batch_size):
                xb = X_tr_t[perm[i : i + batch_size]]
                yb = y_tr_t[perm[i : i + batch_size]]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss = loss_fn(model(xb), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(opt)
                scaler_amp.update()
                train_loss += loss.item()
                n_batches  += 1
            sched.step()

        # Evaluate
        model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            X_va_t  = torch.from_numpy(X_va_s).to(device)
            y_pred  = model(X_va_t).float().cpu().numpy()

        # Clamp to training range ±2 dex — prevents linear-space explosion.
        y_pred = np.clip(y_pred, float(y_tr.min()) - 2.0, float(y_tr.max()) + 2.0)

        fold_metrics.append(compute_metrics(y_va, y_pred))
        print(f"  MLP     fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R²={fold_metrics[-1]['R2']:.4f}")

    print_results("MLP", fold_metrics, g0_values)
    return fold_metrics


# ── Feature importance helper ────────────────────────────────────────────────

def print_feature_importance(models: list, feature_names: list[str]) -> None:
    """Average XGBoost feature importances across folds."""
    importances = np.array([m.feature_importances_ for m in models]).mean(axis=0)
    order = np.argsort(importances)[::-1]
    print("\n--- XGBoost Feature Importance (mean across folds) ---")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<12} {importances[idx]:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Loading data...")
    cubes    = load_all_cubes()
    g0_vals  = get_g0_values(cubes)
    X, y, folds = get_X_y(cubes, use_log_target=True)
    print(f"Total samples: {len(X):,}  |  Features: {X.shape[1]}")

    # 1. Linear Regression
    print("\n[1/3] Linear Regression")
    run_linear(X, y, folds, g0_vals)

    # 2. XGBoost
    print("\n[2/3] XGBoost")
    _, xgb_models = run_xgboost(X, y, folds, g0_vals)
    print_feature_importance(xgb_models, FEATURE_COLS)

    # 3. MLP
    print("\n[3/3] MLP")
    run_mlp(X, y, folds, g0_vals)
