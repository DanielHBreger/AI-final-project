"""
Train and evaluate three pointwise baseline models with leave-one-G0-out CV:
  Linear Regression, XGBoost, MLP (PyTorch)

All models predict log10(nH2). Primary metrics in log space; secondary R2_lin
uses clipped predictions to prevent exponential explosion.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

from data_loader import load_all_cubes, get_X_y, get_g0_values, FEATURE_COLS


def compute_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """
    Primary metrics (R2, RMSE, MAE) in log10(nH2) space (dex).
    R2_lin uses linear nH2 space with predictions clipped to [min-1, max+1] dex
    to avoid numerical explosion from outlier predictions.
    """
    r2   = float(r2_score(y_true_log, y_pred_log))
    rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))
    mae  = float(mean_absolute_error(y_true_log, y_pred_log))

    clip_lo = float(y_true_log.min()) - 1.0
    clip_hi = float(y_true_log.max()) + 1.0
    y_pred_clip = np.clip(y_pred_log, clip_lo, clip_hi)
    r2_lin = float(r2_score(10.0 ** y_true_log, 10.0 ** y_pred_clip))

    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'R2_lin': r2_lin}


def print_results(name: str, fold_metrics: list[dict], g0_values: list[float]) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  {'G0':<8} {'R2_log':>8} {'R2_lin':>8} {'RMSE':>10} {'MAE':>10}")
    print(f"  {'-'*48}")
    for g0, m in zip(g0_values, fold_metrics):
        print(f"  {g0:<8.1f} {m['R2']:>8.4f} {m['R2_lin']:>8.4f} {m['RMSE']:>10.4f} {m['MAE']:>10.4f}")
    r2s     = [m['R2']     for m in fold_metrics]
    r2_lins = [m['R2_lin'] for m in fold_metrics]
    rmses   = [m['RMSE']   for m in fold_metrics]
    maes    = [m['MAE']    for m in fold_metrics]
    print(f"  {'-'*48}")
    print(f"  {'Mean':<8} {np.mean(r2s):>8.4f} {np.mean(r2_lins):>8.4f} {np.mean(rmses):>10.4f} {np.mean(maes):>10.4f}")
    print(f"  {'Std':<8} {np.std(r2s):>8.4f} {np.std(r2_lins):>8.4f} {np.std(rmses):>10.4f} {np.std(maes):>10.4f}")


def run_linear(X: np.ndarray, y: np.ndarray,
               folds: np.ndarray, g0_values: list[float]) -> list[dict]:
    fold_metrics = []
    for fold in range(len(g0_values)):
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
              f"R2={fold_metrics[-1]['R2']:.4f}")

    print_results("Linear Regression", fold_metrics, g0_values)
    return fold_metrics


def run_xgboost(X: np.ndarray, y: np.ndarray,
                folds: np.ndarray, g0_values: list[float],
                subsample: float = 0.3) -> tuple[list[dict], list]:
    """subsample: fraction of training rows per tree (speeds up 15M-row fits)."""
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  XGBoost device: {_device}")
    fold_metrics = []
    models = []
    for fold in range(len(g0_values)):
        train_mask = folds != fold
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[~train_mask], y[~train_mask]

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=subsample,
            colsample_bytree=0.8,
            tree_method='hist',
            device=_device,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        y_pred = model.predict(X_va)

        fold_metrics.append(compute_metrics(y_va, y_pred))
        models.append(model)
        print(f"  XGBoost fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")

    print_results("XGBoost", fold_metrics, g0_values)
    return fold_metrics, models


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

    for fold in range(len(g0_values)):
        train_mask = folds != fold
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[~train_mask], y[~train_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
        X_va_s = scaler.transform(X_va).astype(np.float32)

        # preload entire training fold to GPU to avoid per-batch transfer overhead
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
            X_va_t = torch.from_numpy(X_va_s).to(device)
            y_pred = model(X_va_t).float().cpu().numpy()

        # clip to training range to prevent linear-space explosion
        y_pred = np.clip(y_pred, float(y_tr.min()) - 2.0, float(y_tr.max()) + 2.0)

        fold_metrics.append(compute_metrics(y_va, y_pred))
        print(f"  MLP     fold={fold} (G0={g0_values[fold]:.1f})  "
              f"R2={fold_metrics[-1]['R2']:.4f}")

    print_results("MLP", fold_metrics, g0_values)
    return fold_metrics


def print_feature_importance(models: list, feature_names: list[str]) -> None:
    importances = np.array([m.feature_importances_ for m in models]).mean(axis=0)
    order = np.argsort(importances)[::-1]
    print("\n--- XGBoost Feature Importance (mean across folds) ---")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<12} {importances[idx]:.4f}")


if __name__ == '__main__':
    print("Loading data...")
    cubes    = load_all_cubes()
    g0_vals  = get_g0_values(cubes)
    X, y, folds = get_X_y(cubes, use_log_target=True)
    print(f"Total samples: {len(X):,}  |  Features: {X.shape[1]}")

    print("\n[1/3] Linear Regression")
    run_linear(X, y, folds, g0_vals)

    print("\n[2/3] XGBoost")
    _, xgb_models = run_xgboost(X, y, folds, g0_vals)
    print_feature_importance(xgb_models, FEATURE_COLS)

    print("\n[3/3] MLP")
    run_mlp(X, y, folds, g0_vals)
