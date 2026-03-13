#!/usr/bin/env python3
"""
single_cube_extrapolation.py
============================
Train stacked_sp on a SINGLE G0 cube and predict all 7 cubes.

Builds a 7x7 R2 matrix:  row = source (training) cube, col = target cube.
Diagonal entries (source == target) are in-sample fits.
Off-diagonal entries are fully out-of-sample extrapolations.

Question answered: how much information does one UV-field simulation contain
for predicting chemistry at other UV field strengths?

Architecture: stacked_sp — Ridge meta-learner over xgb_standard_sp + mlp_wide_sp
with multi-scale spatial neighbourhood features (3^3, 5^3, 7^3 kernels).
Ridge is fit on the in-sample base-model predictions of the source cube.

Usage
-----
  # Full 7x7 matrix (default):
  python single_cube_extrapolation.py

  # Faster demo (fewer MLP epochs):
  python single_cube_extrapolation.py --mlp-epochs 20

  # Single source cube only:
  python single_cube_extrapolation.py --train-g0 0.8

  # Skip XGBoost warm-up eval messages:
  python single_cube_extrapolation.py --quiet
"""

import argparse
import json
import datetime
import os
import sys
import numpy as np
from sklearn.linear_model import Ridge

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_loader import (
    load_all_cubes, cube_to_volumes, get_X_y,
    get_g0_values, get_feature_cols, FEATURE_COLS, LOG_TARGET_COL,
)
from classical_models import compute_metrics
from model_helpers import (
    _compute_spatial_X,
    _fit_xgb, _predict_xgb, _fit_mlp, _predict_mlp,
)


# ── Per-source experiment ──────────────────────────────────────────────────────

def run_source_cube(train_fold: int,
                    X_sp: np.ndarray, y: np.ndarray, fold_labels: np.ndarray,
                    g0_vals: list[float],
                    mlp_epochs: int,
                    quiet: bool = False,
                    ) -> list[dict]:
    """Train on cube `train_fold`, predict all 7 cubes.

    Returns a list of dicts (one per target fold), each containing:
        g0_train, g0_test, is_train (bool), xgb_r2, mlp_r2, stacked_r2,
        xgb_metrics, mlp_metrics, stacked_metrics.

    The Ridge meta-learner is fit on in-sample (source cube) predictions of the
    two base models.  For off-diagonal entries this is a valid meta-learner; for
    the diagonal (in-sample) the R2 is an optimistic upper bound.
    """
    g0_src = g0_vals[train_fold]
    mask_tr = fold_labels == train_fold
    X_tr = X_sp[mask_tr]
    y_tr = y[mask_tr]

    print(f"\n  [XGB]  source G0={g0_src:.1f}  ({X_tr.shape[0]:,} training samples)")
    xgb_model, xgb_sc = _fit_xgb(X_tr, y_tr)

    print(f"  [MLP]  source G0={g0_src:.1f}  {mlp_epochs} epochs")
    mlp_model, mlp_sc, device, y_min, y_max = _fit_mlp(
        X_tr, y_tr, epochs=mlp_epochs, quiet=quiet)

    # Fit Ridge on in-sample base-model predictions
    y_xgb_tr = _predict_xgb(xgb_model, xgb_sc, X_tr)
    y_mlp_tr = _predict_mlp(mlp_model, mlp_sc, device, X_tr, y_min, y_max)
    meta = Ridge(alpha=1.0, fit_intercept=True)
    meta.fit(np.column_stack([y_xgb_tr, y_mlp_tr]), y_tr)
    print(f"  Ridge: XGB={meta.coef_[0]:.3f}  MLP={meta.coef_[1]:.3f}  "
          f"intercept={meta.intercept_:.3f}")

    # Predict all 7 cubes
    records = []
    for tgt_fold in range(len(g0_vals)):
        mask_te = fold_labels == tgt_fold
        X_te = X_sp[mask_te]
        y_te = y[mask_te]

        y_xgb = _predict_xgb(xgb_model, xgb_sc, X_te)
        y_mlp = _predict_mlp(mlp_model, mlp_sc, device, X_te, y_min, y_max)
        y_stk = meta.predict(np.column_stack([y_xgb, y_mlp])).astype(np.float32)

        is_tr = (tgt_fold == train_fold)
        tag   = '(in-sample)' if is_tr else ''
        m_xgb = compute_metrics(y_te, y_xgb)
        m_mlp = compute_metrics(y_te, y_mlp)
        m_stk = compute_metrics(y_te, y_stk)
        print(f"    tgt G0={g0_vals[tgt_fold]:.1f} {tag:<12}  "
              f"XGB={m_xgb['R2']:.4f}  MLP={m_mlp['R2']:.4f}  "
              f"stacked={m_stk['R2']:.4f}")
        records.append({
            'g0_train':        g0_src,
            'g0_test':         g0_vals[tgt_fold],
            'is_train':        bool(is_tr),
            'xgb_r2':          m_xgb['R2'],
            'mlp_r2':          m_mlp['R2'],
            'stacked_r2':      m_stk['R2'],
            'xgb_metrics':     m_xgb,
            'mlp_metrics':     m_mlp,
            'stacked_metrics': m_stk,
        })
    return records


# ── Results display ────────────────────────────────────────────────────────────

def _r2_matrix(all_records: list[list[dict]], g0_vals: list[float],
               key: str) -> np.ndarray:
    """Build a (n_src, n_tgt) R2 matrix for model `key`."""
    n = len(g0_vals)
    mat = np.full((n, n), np.nan)
    for i, records in enumerate(all_records):
        for rec in records:
            j = g0_vals.index(rec['g0_test'])
            mat[i, j] = rec[key]
    return mat


def print_r2_table(mat: np.ndarray, g0_vals: list[float], title: str) -> None:
    """Print R2 matrix to stdout as a formatted table."""
    n   = len(g0_vals)
    hdr = ''.join(f'  G0={g:.1f}' for g in g0_vals)
    print(f"\n{title}")
    print(f"{'src\\tgt':<10}{hdr}")
    for i, g_src in enumerate(g0_vals):
        row = ''.join(
            f'  {"--" if np.isnan(mat[i, j]) else f"{mat[i, j]:6.4f}"}'
            for j in range(n)
        )
        print(f"  G0={g_src:<5.1f}{row}")


def plot_heatmaps(all_records: list[list[dict]], g0_vals: list[float],
                  save_path: str) -> None:
    """Save a 3-panel heatmap figure (XGB | MLP | stacked) as PNG."""
    keys   = ['xgb_r2', 'mlp_r2', 'stacked_r2']
    titles = ['XGBoost', 'MLP', 'Stacked (Ridge)']
    n      = len(g0_vals)
    labels = [f'G0={g:.1f}' for g in g0_vals]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif':  ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size':   8,
        'axes.linewidth': 0.6,
        'savefig.dpi':    300,
        'savefig.bbox':   'tight',
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.5))
    fig.suptitle('Single-cube extrapolation: R\u00b2 matrix\n'
                 '(row = training cube, column = target cube; '
                 'diagonal = in-sample)', fontsize=9)

    vmin, vmax = 0.0, 1.0

    for ax, key, title in zip(axes, keys, titles):
        mat = _r2_matrix(all_records, g0_vals, key)
        im  = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap='RdYlGn', aspect='equal')

        # Annotate each cell
        for i in range(n):
            for j in range(n):
                val  = mat[i, j]
                text = f'{val:.3f}' if not np.isnan(val) else ''
                col  = 'black' if 0.3 < val < 0.75 else 'white'
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=7, color=col)

        # Outline diagonal cells (in-sample)
        for k in range(n):
            ax.add_patch(plt.Rectangle(
                (k - 0.5, k - 0.5), 1, 1,
                fill=False, edgecolor='black', linewidth=1.8, zorder=5,
            ))

        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel('Training cube (source G0)', fontsize=8)
        ax.set_xlabel('Target cube (test G0)', fontsize=8)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$R^2$', fontsize=8)
        cb.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Heatmap saved -> {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Single-cube extrapolation: train on 1 G0 cube, predict all 7.')
    parser.add_argument('--train-g0', type=float, default=None,
                        help='Run only for this source G0 (default: all 7)')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='MLP training epochs (default: 100)')
    parser.add_argument('--spatial-kernels', nargs='+', type=int, default=[3, 5, 7],
                        help='Spatial filter kernel sizes (default: 3 5 7)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-epoch MLP training messages')
    parser.add_argument('--no-fh2', action='store_true',
                        help='Exclude log_fh2 from input features')
    args = parser.parse_args()

    feat_cols = get_feature_cols(args.no_fh2)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    # ── Features ──────────────────────────────────────────────────────────────
    print("\nBuilding base feature matrix...")
    X, y, fold_labels = get_X_y(cubes, use_log_target=True, feature_cols=feat_cols)
    print(f"  Base X: {X.shape}   y: {y.shape}")

    print(f"\nBuilding spatial features (kernels={args.spatial_kernels})...")
    all_vols = [cube_to_volumes(df, feat_cols) for df in cubes]
    X_extra  = _compute_spatial_X(cubes, all_vols, feat_cols,
                                   kernel_sizes=tuple(args.spatial_kernels))
    X_sp = np.concatenate([X, X_extra], axis=1)
    n_sp = len(feat_cols) * len(args.spatial_kernels)
    print(f"  Spatial features: {n_sp}  ->  X_sp: {X_sp.shape}")

    # ── Select source folds ───────────────────────────────────────────────────
    if args.train_g0 is not None:
        if args.train_g0 not in g0_vals:
            print(f"ERROR: G0={args.train_g0} not found. Available: {g0_vals}",
                  file=sys.stderr)
            sys.exit(1)
        source_folds = [g0_vals.index(args.train_g0)]
    else:
        source_folds = list(range(len(g0_vals)))

    # ── Run experiment ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Single-cube extrapolation: {len(source_folds)} source cube(s)  "
          f"x  {len(g0_vals)} target cubes")
    print(f"{'='*60}")

    all_records: list[list[dict]] = []
    for train_fold in source_folds:
        g0_src = g0_vals[train_fold]
        print(f"\n{'='*60}")
        print(f"Source cube: G0={g0_src:.1f}  (fold {train_fold+1}/{len(g0_vals)})")
        print(f"{'='*60}")
        records = run_source_cube(
            train_fold, X_sp, y, fold_labels, g0_vals,
            mlp_epochs=args.mlp_epochs, quiet=args.quiet,
        )
        all_records.append(records)

    # ── Summary tables ────────────────────────────────────────────────────────
    if len(all_records) == len(g0_vals):
        print("\n\n" + "="*60)
        print("R2 MATRICES (rows=source, cols=target; diagonal=in-sample)")
        print("="*60)
        for key, label in [('xgb_r2', 'XGBoost'), ('mlp_r2', 'MLP'),
                            ('stacked_r2', 'Stacked (Ridge meta-learner)')]:
            mat = _r2_matrix(all_records, g0_vals, key)
            print_r2_table(mat, g0_vals, label)

        # Off-diagonal OOB summary (exclude diagonal)
        print("\n\nOff-diagonal (out-of-sample) summary:")
        for key, label in [('xgb_r2', 'XGB'), ('mlp_r2', 'MLP'), ('stacked_r2', 'Stacked')]:
            vals = [r[key] for recs in all_records for r in recs if not r['is_train']]
            print(f"  {label:<12}  mean={np.mean(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir   = os.path.join('logs', 'single_cube_extrapolation')
    os.makedirs(log_dir, exist_ok=True)
    log_path  = os.path.join(log_dir, f'run_{timestamp}.json')

    log = {
        'timestamp':      timestamp,
        'mlp_epochs':     args.mlp_epochs,
        'spatial_kernels': args.spatial_kernels,
        'g0_values':      g0_vals,
        'source_folds':   source_folds,
        'results': [
            [
                {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                 for k, v in rec.items()
                 if k not in ('xgb_metrics', 'mlp_metrics', 'stacked_metrics')}
                for rec in recs
            ]
            for recs in all_records
        ],
    }
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"\nJSON log saved -> {log_path}")

    # ── Save heatmap ──────────────────────────────────────────────────────────
    if len(all_records) == len(g0_vals):
        fig_path = os.path.join(log_dir, f'heatmap_{timestamp}.png')
        plot_heatmaps(all_records, g0_vals, fig_path)


if __name__ == '__main__':
    main()
