#!/usr/bin/env python3
"""
predict_and_visualize.py
========================
Train stacked_sp (Ridge meta-learner over xgb_standard_sp + mlp_wide_sp, with
multi-scale spatial features 3^3+5^3+7^3) on 6 of the 7 G0 cubes, predict on
the held-out cube, and display a 3-panel interactive 3D comparison:

    [ Ground Truth ]  |  [ stacked_sp Prediction ]  |  [ Error ]

Best model from run 231200: stacked_sp R2=0.9911, R2_lin=0.616
4-run average: stacked_sp R2=0.988 +/- 0.002

Mass-budget recalibration (--recal-mode, default 'mass'): the stacked
predictions carry a G0-dependent, density-dependent error that shows up in
the predicted total H2 mass even when the cell-mean log bias is ~0.  The
per-training-cube *mass-weighted* OOF residual of the Ridge meta-learner is
fitted against log10(G0) and the value at the held-out G0 is subtracted —
training-cube quantities only, no leakage.  'mean' restores the legacy
unweighted (cell-mean) correction; 'off' (or --no-recalibrate) disables it.
Both raw and recalibrated volumes are saved; both fits are recorded in the
npz either way.

Usage
-----
  # Hold out G0=0.8 (default, near-best fold):
  python predict_and_visualize.py

  # Hold out the hardest extrapolation fold:
  python predict_and_visualize.py --g0 0.1

  # Quick demo (fewer epochs):
  python predict_and_visualize.py --g0 1.6 --mlp-epochs 30

  # Single-scale spatial only:
  python predict_and_visualize.py --spatial-kernels 3
"""

import argparse
import datetime
import json
import os
import sys
import numpy as np

# Ensure relative data paths (icedrive-dl-182bd/UVonly) resolve correctly
# regardless of which directory Code Runner / the shell launches from.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data_loader import (
    load_all_cubes, cube_to_volumes, get_X_y,
    get_g0_values, get_feature_cols, add_drop_args, build_drop_set,
)
from classical_models import compute_metrics
from model_helpers import (
    _XGB_CFG, FlexMLP, _compute_spatial_X, _compute_weights, _preds_to_volume,
    fit_g0_bias_correction, predict_bias, mass_weighted_bias,
)

# ── Training functions ─────────────────────────────────────────────────────────

def _train_xgb(X_tr: np.ndarray, y_tr: np.ndarray,
               X_va: np.ndarray, y_va: np.ndarray) -> np.ndarray:
    """Fit XGBoost (xgb_standard config) and return val predictions."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sc    = StandardScaler()
    model = xgb.XGBRegressor(**_XGB_CFG, device=device)
    model.fit(sc.fit_transform(X_tr), y_tr,
              sample_weight=_compute_weights(y_tr),
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

    w_tr_t = torch.from_numpy(_compute_weights(y_tr)).to(device)

    model      = FlexMLP(X_tr_s.shape[1]).to(device)
    opt        = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched      = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler_amp = torch.amp.GradScaler('cuda', enabled=use_amp)
    batch_size = 262_144

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
        if (ep + 1) % 10 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                y_vp = model(torch.from_numpy(X_va_s).to(device)).float().cpu().numpy()
            m = compute_metrics(y_va, y_vp, fast=True)
            print(f"    ep {ep+1:3d}/{epochs}  R2={m['R2']:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X_va_s).to(device)).float().cpu().numpy()
    # Clamp to training range ±2 dex — prevents linear-space explosion.
    y_pred = np.clip(y_pred, float(y_tr.min()) - 2.0, float(y_tr.max()) + 2.0)
    return y_pred.astype(np.float32)

# ── Per-fold training / saving ─────────────────────────────────────────────────

def _run_fold(fold: int, g0_val: float,
              X_sp: np.ndarray, y: np.ndarray, fold_labels: np.ndarray,
              cubes: list, g0_vals: list, mlp_epochs: int, spatial_kernels: list,
              recal_mode: str = 'mass') -> None:
    """Train stacked_sp on N-1 cubes, predict on the held-out fold.

    Stacking procedure:
      1. Nested 6-fold CV on the training cubes generates OOF predictions
         for xgb_standard_sp and mlp_wide_sp.
      2. Ridge(alpha=1.0) is fit on those OOF predictions.
      3. Final base models are trained on all 6 training cubes; Ridge meta
         is applied to their predictions on the held-out cube.

    Mass-budget recalibration (recal_mode): the per-training-cube OOF bias of
    the stacked model is fitted against log10(G0) and the fitted value at the
    held-out G0 is subtracted from the predictions (training-cube quantities
    only — leakage-free).  The per-cube bias is computed as
      'mass'  mass-weighted mean residual (weights 10**y_true) — centres the
              total-H2-mass budget (default; the cell-mean correction cannot
              move a density-dependent error)
      'mean'  unweighted mean residual — centres the cell-mean log bias only
      'off'   no correction applied (both fits still recorded in the npz)
    """
    assert recal_mode in ('mass', 'mean', 'off'), recal_mode
    train_folds = [f for f in np.unique(fold_labels).tolist() if f != fold]

    # ── Step 1: Nested OOF predictions for Ridge meta-learner ─────────────────
    print(f"\n[Step 1/3] Nested {len(train_folds)}-fold OOF predictions "
          f"for Ridge meta-learner...")
    meta_yt, meta_xgb, meta_mlp = [], [], []
    for j in train_folds:
        tr = (fold_labels != fold) & (fold_labels != j)
        va = fold_labels == j
        print(f"\n  OOF fold G0={g0_vals[j]:.1f}  ({tr.sum():,} tr / {va.sum():,} va)")
        oof_xgb = _train_xgb(X_sp[tr], y[tr], X_sp[va], y[va])
        oof_mlp = _train_mlp(X_sp[tr], y[tr], X_sp[va], y[va], epochs=mlp_epochs)
        meta_yt.append(y[va])
        meta_xgb.append(oof_xgb)
        meta_mlp.append(oof_mlp)

    X_meta = np.column_stack([np.concatenate(meta_xgb), np.concatenate(meta_mlp)])
    y_meta = np.concatenate(meta_yt)
    meta   = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(X_meta, y_meta)

    # ── Mass-budget recalibration: per-cube stacked-OOF bias vs log10(G0) ─────
    oof_preds = [meta.predict(np.column_stack([xgb_j, mlp_j]))
                 for xgb_j, mlp_j in zip(meta_xgb, meta_mlp)]
    per_cube_bias = np.array([
        float(np.mean(pp - yt_j)) for pp, yt_j in zip(oof_preds, meta_yt)
    ])
    per_cube_bias_mw = np.array([
        mass_weighted_bias(pp, yt_j) for pp, yt_j in zip(oof_preds, meta_yt)
    ])
    g0_train = np.array([g0_vals[j] for j in train_folds])
    bias_slope, bias_intercept = fit_g0_bias_correction(per_cube_bias, g0_train)
    mw_slope,   mw_intercept   = fit_g0_bias_correction(per_cube_bias_mw, g0_train)
    offset_mean = predict_bias(bias_slope, bias_intercept, g0_val)
    offset_mass = predict_bias(mw_slope,   mw_intercept,   g0_val)
    bias_offset = {'mass': offset_mass, 'mean': offset_mean, 'off': 0.0}[recal_mode]
    print(f"\n  OOF bias per training cube (dex):      "
          f"{np.array2string(per_cube_bias, precision=3)}")
    print(f"  OOF mass-wtd bias per training cube:   "
          f"{np.array2string(per_cube_bias_mw, precision=3)}")
    print(f"  Fitted bias(G0={g0_val}): mean={offset_mean:+.3f}  "
          f"mass={offset_mass:+.3f} dex  ->  applying {recal_mode} "
          f"({bias_offset:+.3f})")

    # ── Step 2: Final base models on all 6 training cubes ─────────────────────
    mask = fold_labels != fold
    X_tr = X_sp[mask];  y_tr = y[mask]
    X_va = X_sp[~mask]; y_va = y[~mask]

    print(f"\n[Step 2/3] Final base models on {mask.sum():,} training samples...")

    print(f"\n[xgb_standard_sp]")
    y_xgb = _train_xgb(X_tr, y_tr, X_va, y_va)
    m_xgb = compute_metrics(y_va, y_xgb)
    print(f"  XGB  R2={m_xgb['R2']:.4f}  R2_lin={m_xgb['R2_lin']:.4f}  RMSE={m_xgb['RMSE']:.4f}")

    print(f"\n[mlp_wide_sp]  {mlp_epochs} epochs...")
    y_mlp = _train_mlp(X_tr, y_tr, X_va, y_va, epochs=mlp_epochs)
    m_mlp = compute_metrics(y_va, y_mlp)
    print(f"  MLP  R2={m_mlp['R2']:.4f}  R2_lin={m_mlp['R2_lin']:.4f}  RMSE={m_mlp['RMSE']:.4f}")

    # ── Step 3: Apply Ridge meta + bias recalibration ──────────────────────────
    y_stacked_raw = meta.predict(np.column_stack([y_xgb, y_mlp])).astype(np.float32)
    y_stacked     = (y_stacked_raw - bias_offset).astype(np.float32)
    m_raw     = compute_metrics(y_va, y_stacked_raw)
    m_stacked = compute_metrics(y_va, y_stacked)
    print(f"\n[Step 3/3] stacked_sp (raw)  R2={m_raw['R2']:.4f}  "
          f"R2_lin={m_raw['R2_lin']:.4f}  RMSE={m_raw['RMSE']:.4f}  "
          f"bias={m_raw['bias']:+.3f}  massRatio={m_raw['mass_ratio']:.3f}")
    print(f"           stacked_sp (cal)  R2={m_stacked['R2']:.4f}  "
          f"R2_lin={m_stacked['R2_lin']:.4f}  RMSE={m_stacked['RMSE']:.4f}  "
          f"bias={m_stacked['bias']:+.3f}  massRatio={m_stacked['mass_ratio']:.3f}")
    print(f"           stacked_sp (cal)  f0.1={m_stacked['frac_01']:.3f}  "
          f"f0.3={m_stacked['frac_03']:.3f}  CCC={m_stacked['CCC']:.4f}  "
          f"W1={m_stacked['W1']:.3f}  R2_mol={m_stacked['R2_mol']:.4f}  "
          f"R2_dif={m_stacked['R2_dif']:.4f}")
    print(f"  Ridge weights: XGB={meta.coef_[0]:.3f}  MLP={meta.coef_[1]:.3f}  "
          f"intercept={meta.intercept_:.3f}")

    # pred_vol is the delivered (recalibrated) prediction; the raw volume is
    # kept alongside so the correction remains inspectable after the fact.
    pred_vol     = _preds_to_volume(cubes[fold], y_stacked)
    pred_vol_raw = _preds_to_volume(cubes[fold], y_stacked_raw)

    os.makedirs('predictions', exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f'predictions/pred_g0_{g0_val}_{timestamp}.npz'
    # Full metric dicts for all four models, JSON-encoded so every recorded
    # value survives in the .npz (read back with json.loads(str(d['metrics_json'])))
    metrics_json = json.dumps({
        'xgb_sp':      m_xgb,
        'mlp_sp':      m_mlp,
        'stacked_raw': m_raw,
        'stacked_cal': m_stacked,   # delivered prediction (per recal_mode)
        'recal_mode':  recal_mode,
    })
    np.savez_compressed(
        save_path,
        pred_vol         = pred_vol,
        pred_vol_raw     = pred_vol_raw,
        metrics_json     = metrics_json,
        g0               = np.float64(g0_val),
        r2_xgb           = np.float32(m_xgb['R2']),
        r2_mlp           = np.float32(m_mlp['R2']),
        r2_stacked       = np.float32(m_stacked['R2']),
        r2_stacked_raw   = np.float32(m_raw['R2']),
        mass_ratio       = np.float32(m_stacked['mass_ratio']),
        mass_ratio_raw   = np.float32(m_raw['mass_ratio']),
        recal_mode       = np.str_(recal_mode),
        bias_offset      = np.float32(bias_offset),      # applied offset
        bias_offset_mean = np.float32(offset_mean),
        bias_offset_mass = np.float32(offset_mass),
        bias_slope       = np.float32(bias_slope),       # mean-residual fit
        bias_intercept   = np.float32(bias_intercept),
        mw_slope         = np.float32(mw_slope),         # mass-weighted fit
        mw_intercept     = np.float32(mw_intercept),
        per_cube_bias    = per_cube_bias.astype(np.float32),
        per_cube_bias_mw = per_cube_bias_mw.astype(np.float32),
        meta_coef        = np.array(meta.coef_,    dtype=np.float32),
        meta_intercept   = np.float32(meta.intercept_),
        spatial_kernels  = np.array(spatial_kernels, dtype=np.int32),
        mlp_epochs       = np.int32(mlp_epochs),
    )
    print(f"Prediction saved -> {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict nH2 with stacked_sp and compare 3D volumes against truth.')
    parser.add_argument('--g0', type=float, default=0.8,
                        help='G0 value to hold out as validation (default: 0.8). '
                             'Available: 0.1 0.2 0.4 0.8 1.6 3.2 6.4')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='MLP training epochs (default: 100; use 30 for a fast demo)')
    parser.add_argument('--spatial-kernels', nargs='+', type=int, default=[3, 5, 7],
                        help='Spatial filter kernel sizes (default: 3 5 7)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 7 G0 folds and save a prediction file for each')
    parser.add_argument('--recal-mode', choices=['mass', 'mean', 'off'],
                        default=None,
                        help="G0-linear bias recalibration: 'mass' fits the "
                             'mass-weighted per-cube OOF residual (closes the '
                             "H2 mass budget; default), 'mean' the unweighted "
                             'residual (legacy cell-mean correction), '
                             "'off' saves raw stacked predictions as pred_vol.")
    parser.add_argument('--no-recalibrate', action='store_false', dest='recalibrate',
                        help="Deprecated alias for --recal-mode off.")
    parser.set_defaults(recalibrate=True)
    add_drop_args(parser)
    args = parser.parse_args()
    recal_mode = args.recal_mode or ('mass' if args.recalibrate else 'off')

    feat_cols = get_feature_cols(build_drop_set(args))

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    # ── Baseline features ─────────────────────────────────────────────────────
    print("\nBuilding feature matrix...")
    X, y, fold_labels = get_X_y(cubes, use_log_target=True, feature_cols=feat_cols)
    print(f"  Base X shape: {X.shape}   y shape: {y.shape}")

    # ── Spatial features (needs volumes) ──────────────────────────────────────
    print(f"\nBuilding volumes and spatial features "
          f"(kernels={args.spatial_kernels})...")
    all_vols = [cube_to_volumes(df, feat_cols) for df in cubes]
    X_extra  = _compute_spatial_X(cubes, all_vols, feat_cols,
                                   kernel_sizes=tuple(args.spatial_kernels))
    n_sp = len(feat_cols) * len(args.spatial_kernels)
    X_sp = np.concatenate([X, X_extra], axis=1)
    print(f"  Spatial features: {n_sp}  ->  X_sp shape: {X_sp.shape}")

    # ── Run fold(s) ────────────────────────────────────────────────────────────
    if args.all:
        print(f"\nRunning all {len(g0_vals)} folds...")
        for fold, g0_val in enumerate(g0_vals):
            print(f"\n{'='*60}\nFold {fold+1}/{len(g0_vals)}  G0={g0_val}\n{'='*60}")
            _run_fold(fold, g0_val, X_sp, y, fold_labels, cubes, g0_vals,
                      args.mlp_epochs, args.spatial_kernels,
                      recal_mode=recal_mode)
    else:
        if args.g0 not in g0_vals:
            print(f"ERROR: G0={args.g0} not found. Available: {g0_vals}", file=sys.stderr)
            sys.exit(1)
        fold = g0_vals.index(args.g0)
        print(f"\nValidation fold: {fold}  (G0={args.g0})")
        print(f"Training on {len(cubes) - 1} cubes  |  "
              f"Validating on 1 cube (G0={args.g0})")
        _run_fold(fold, args.g0, X_sp, y, fold_labels, cubes, g0_vals,
                  args.mlp_epochs, args.spatial_kernels,
                  recal_mode=recal_mode)


if __name__ == '__main__':
    main()
