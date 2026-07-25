#!/usr/bin/env python3
"""
weight_alpha_sweep.py
=====================
Sensitivity of the density-weighting strength alpha (RUN_PLAN run 12).

The headline model trains with exponential sample weights that rise from
1x at the 99th target percentile to alpha x at the 99.99th
(model_helpers._compute_weights, alpha=100 by default).  The percentile
anchors and alpha were a designed choice, never tuned; a reviewer asks
whether the result is sensitive to alpha.  This script sweeps
alpha in {10, 100, 1000} on the XGBoost + spatial-features model under the
same leave-one-G0-out protocol as the main comparison, reporting the full
metric suite per fold and the cross-fold summary for each alpha.

XGBoost only (the tree branch is deterministic and ~10x faster than the
MLP), matching the RUN_PLAN recommendation for a fast inoculating table.
alpha=100 reproduces the xgb_standard_sp_w rows of the comparison logs
bit-for-bit (same weights, same seed, same features).

Usage
-----
  python -u weight_alpha_sweep.py                        # alpha in {10,100,1000}
  python -u weight_alpha_sweep.py --alphas 30 100 300
  python -u weight_alpha_sweep.py --log results/weight_alpha_<date>.json
"""

import argparse
import datetime
import json
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_all_cubes, cube_to_volumes, get_X_y, get_g0_values,
    get_feature_cols, add_drop_args, build_drop_set,
)
from classical_models import compute_metrics
from model_helpers import _compute_spatial_X, _XGB_CFG, _compute_weights

import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


def _fit_xgb_alpha(X_tr: np.ndarray, y_tr: np.ndarray, alpha: float):
    """xgb_standard_sp_w fit with an explicit weighting strength.

    Identical to model_helpers._fit_xgb except alpha is passed through to
    _compute_weights (which defaults to 100.0 there)."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sc = StandardScaler()
    X_s = sc.fit_transform(X_tr)
    model = xgb.XGBRegressor(**_XGB_CFG, device=device)
    model.fit(X_s, y_tr, sample_weight=_compute_weights(y_tr, alpha=alpha),
              verbose=False)
    return model, sc


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Density-weight alpha sensitivity sweep (XGBoost + spatial).')
    parser.add_argument('--alphas', nargs='+', type=float,
                        default=[10.0, 100.0, 1000.0],
                        help='Weighting strengths to sweep (default: 10 100 1000)')
    parser.add_argument('--spatial-kernels', nargs='+', type=int, default=[3, 5, 7],
                        help='Spatial filter kernel sizes (default: 3 5 7)')
    parser.add_argument('--log', type=str, default=None,
                        help='Output JSON path (default: timestamped in results/)')
    add_drop_args(parser)
    args = parser.parse_args()

    feat_cols = get_feature_cols(build_drop_set(args))

    print('Loading cubes...')
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    print(f'Building base features ({len(feat_cols)} local)...')
    X, y, fold_labels = get_X_y(cubes, use_log_target=True, feature_cols=feat_cols)

    print(f'Building spatial features (kernels={args.spatial_kernels})...')
    all_vols = [cube_to_volumes(df, feat_cols) for df in cubes]
    X_extra  = _compute_spatial_X(cubes, all_vols, feat_cols,
                                  kernel_sizes=tuple(args.spatial_kernels))
    X_sp = np.concatenate([X, X_extra], axis=1)
    print(f'  X_sp: {X_sp.shape}')

    results: dict[str, list[dict]] = {}
    for alpha in args.alphas:
        akey = f'{alpha:g}'
        print(f'\n{"="*70}\nalpha = {akey}\n{"="*70}')
        fold_recs: list[dict] = []
        for test_fold in range(len(g0_vals)):
            g0 = g0_vals[test_fold]
            te = fold_labels == test_fold
            tr = ~te
            model, sc = _fit_xgb_alpha(X_sp[tr], y[tr], alpha)
            y_pred = model.predict(sc.transform(X_sp[te])).astype(np.float32)
            m = compute_metrics(y[te], y_pred)
            fold_recs.append({'fold': test_fold, 'g0': g0, 'metrics': m})
            print(f'  G0={g0:<5.1f}  R2={m["R2"]:.4f}  RMSE={m["RMSE"]:.4f}  '
                  f'massR={m["mass_ratio"]:.3f}  R2_mol={m["R2_mol"]:.4f}  '
                  f'frac01={m["frac_01"]:.3f}')
        results[akey] = fold_recs

    # ── Summary ────────────────────────────────────────────────────────────────
    keys = ['R2', 'RMSE', 'bias', 'scatter', 'mass_ratio', 'R2_mol', 'frac_01', 'W1']
    print(f'\n{"="*90}\nCross-fold summary (mass_ratio as min-max range)\n{"="*90}')
    print(f'{"alpha":>7} ' + ' '.join(f'{k:>9}' for k in keys))
    summary: dict[str, dict] = {}
    for akey, recs in results.items():
        arr = {k: np.array([r['metrics'][k] for r in recs]) for k in keys}
        row = {}
        for k in keys:
            if k == 'mass_ratio':
                row[k] = [float(arr[k].min()), float(arr[k].max())]
            elif k == 'R2_mol':
                row[k] = float(np.nanmean(arr[k]))
            else:
                row[k] = float(arr[k].mean())
        row['R2_std'] = float(arr['R2'].std(ddof=0))
        summary[akey] = row
        cells = []
        for k in keys:
            if k == 'mass_ratio':
                cells.append(f'{row[k][0]:.2f}-{row[k][1]:.2f}')
            else:
                cells.append(f'{row[k]:>9.4f}')
        print(f'{akey:>7} ' + ' '.join(f'{c:>9}' for c in cells))

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = args.log or f'results/weight_alpha_{ts}.json'
    with open(out, 'w') as f:
        json.dump({
            'timestamp': ts,
            'alphas': args.alphas,
            'spatial_kernels': args.spatial_kernels,
            'feature_cols': feat_cols,
            'g0_values': g0_vals,
            'model': 'xgb_standard_sp_w',
            'results': results,
            'summary': summary,
        }, f, indent=2)
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
