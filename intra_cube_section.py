#!/usr/bin/env python3
"""
intra_cube_section.py
=====================
Train stacked_sp on a SPATIAL SECTION of one cube and predict the remaining cells.

For each of the 7 G0 cubes, and for each split strategy, the model is fit on a
subset of the cube's cells (the "training section") and evaluated on the
complementary held-out cells (the "test section").

Spatial features are computed from a TRAINING-SECTION-ONLY volume: test-cell
positions are zeroed before applying neighbourhood means (uniform_filter), so
the model never sees structural context leaking from the held-out region.
Raw physical features (X_flat) are still taken from all cells since those are
fully observed simulation outputs.

Question answered: how well does the model interpolate within a single cube?
How much spatial data is needed for near-lossless reconstruction?

Split strategies
----------------
  x_half   train ix < 64,   test ix >= 64          (~50% / ~50%)
  y_half   train iy < 64,   test iy >= 64          (~50% / ~50%)
  z_half   train iz < 64,   test iz >= 64          (~50% / ~50%)
  rand_1   random  1% train, 99% test               ( 1% / 99%)
  rand_5   random  5% train, 95% test               ( 5% / 95%)
  rand_10  random 10% train, 90% test              (10% / 90%)
  rand_25  random 25% train, 75% test              (25% / 75%)
  rand_50  random 50% train, 50% test              (50% / 50%)
  rand_75  random 75% train, 25% test              (75% / 25%)
  box_1    random cubic sub-region ~1% volume       ( 1% / 99%)
  box_5    random cubic sub-region ~5% volume       ( 5% / 95%)
  box_10   random cubic sub-region ~10% volume     (10% / 90%)
  box_25   random cubic sub-region ~25% volume     (25% / 75%)
  box_50   random cubic sub-region ~50% volume     (50% / 50%)

Box splits place a single axis-aligned cubic training region of the requested
volume at a random position; the remainder is held out for testing.  This
contrasts with random-voxel splits (scattered interpolation) and half-space
splits (planar extrapolation), sitting between the two in spatial locality.

Architecture: stacked_sp (xgb_standard_sp + mlp_wide_sp + Ridge meta-learner).
Ridge is fit on in-sample base-model predictions of the training section.

Usage
-----
  # Full run (all 7 G0 cubes, all 9 splits):
  python intra_cube_section.py

  # Faster demo (fewer MLP epochs):
  python intra_cube_section.py --mlp-epochs 20

  # Single cube only:
  python intra_cube_section.py --g0 0.8

  # Suppress per-epoch output:
  python intra_cube_section.py --quiet
"""

import argparse
import json
import datetime
import os
import numpy as np
from sklearn.linear_model import Ridge

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from data_loader import (
    load_all_cubes, cube_to_volumes, get_g0_values,
    get_feature_cols, add_drop_args, build_drop_set, FEATURE_COLS, LOG_TARGET_COL,
)
from classical_models import compute_metrics
from model_helpers import (
    _compute_spatial_X,
    _fit_xgb, _predict_xgb, _fit_mlp, _predict_mlp,
)

# ── Split strategy definitions ─────────────────────────────────────────────────

SPLITS = [
    ('x_half',  lambda df: df['ix'].values < 64),
    ('y_half',  lambda df: df['iy'].values < 64),
    ('z_half',  lambda df: df['iz'].values < 64),
    ('rand_1',  None),   # fraction-based: scattered random voxels
    ('rand_5',  None),
    ('rand_10', None),
    ('rand_25', None),
    ('rand_50', None),
    ('rand_75', None),
    ('box_1',   None),   # contiguous cubic sub-region
    ('box_5',   None),
    ('box_10',  None),
    ('box_25',  None),
    ('box_50',  None),
]

RAND_FRACTIONS = {
    'rand_1': 0.01, 'rand_5': 0.05, 'rand_10': 0.10,
    'rand_25': 0.25, 'rand_50': 0.50, 'rand_75': 0.75,
}

# Target volume fraction -> cubic side length (rounds to nearest integer, clamped to [1,128])
BOX_FRACTIONS = {
    'box_1': 0.01, 'box_5': 0.05, 'box_10': 0.10,
    'box_25': 0.25, 'box_50': 0.50,
}


def _make_mask(split_name: str, df, rng: np.random.Generator) -> np.ndarray:
    """Return boolean training mask for the given split strategy."""
    for name, fn in SPLITS:
        if name == split_name:
            if fn is not None:
                return fn(df)
            # Scattered random voxels
            if split_name in RAND_FRACTIONS:
                frac = RAND_FRACTIONS[split_name]
                n    = len(df)
                idx  = rng.choice(n, size=int(n * frac), replace=False)
                mask = np.zeros(n, dtype=bool)
                mask[idx] = True
                return mask
            # Contiguous cubic sub-region
            elif split_name in BOX_FRACTIONS:
                frac = BOX_FRACTIONS[split_name]
                side = max(1, min(128, round((frac * 128 ** 3) ** (1 / 3))))
                x0   = int(rng.integers(0, 128 - side + 1))
                y0   = int(rng.integers(0, 128 - side + 1))
                z0   = int(rng.integers(0, 128 - side + 1))
                ix   = df['ix'].values.astype(int) - 1
                iy   = df['iy'].values.astype(int) - 1
                iz   = df['iz'].values.astype(int) - 1
                return ((ix >= x0) & (ix < x0 + side) &
                        (iy >= y0) & (iy < y0 + side) &
                        (iz >= z0) & (iz < z0 + side))
    raise ValueError(f"Unknown split: {split_name}")


def _make_masked_vols(cube, vols: dict, mask_tr: np.ndarray) -> dict:
    """Return copies of `vols` with test-section cell positions set to 0.

    Neighbourhood means (uniform_filter) computed from these masked volumes
    reflect only the physical environment of training cells, so the model
    cannot exploit structural context leaking from the held-out region.
    """
    ix = cube['ix'].values.astype(int) - 1
    iy = cube['iy'].values.astype(int) - 1
    iz = cube['iz'].values.astype(int) - 1
    mask_te = ~mask_tr
    masked = {}
    for col, vol in vols.items():
        v = vol.copy()
        v[ix[mask_te], iy[mask_te], iz[mask_te]] = 0.0
        masked[col] = v
    return masked


# ── Per-cube, per-split experiment ─────────────────────────────────────────────

def run_cube_split(cube, g0: float, split_name: str,
                   mlp_epochs: int, quiet: bool,
                   rng: np.random.Generator,
                   feat_cols: list = FEATURE_COLS) -> dict:
    """Train stacked_sp on one spatial section, evaluate on held-out section.

    Returns a dict with g0, split, n_train, n_test, and R2 for both sections
    for each of xgb, mlp, and stacked.
    """
    # ── Create split mask ─────────────────────────────────────────────────────
    mask_tr = _make_mask(split_name, cube, rng)
    mask_te = ~mask_tr

    # ── Build feature matrix (spatial features from training section only) ────
    vols    = cube_to_volumes(cube, feat_cols)
    vols_tr = _make_masked_vols(cube, vols, mask_tr)
    X_sp    = _compute_spatial_X([cube], [vols_tr], feat_cols)
    X_flat  = cube[feat_cols].values.astype(np.float32)
    X       = np.concatenate([X_flat, X_sp], axis=1)                  # (N, 60)
    y       = cube[LOG_TARGET_COL].values.astype(np.float32)

    X_tr, y_tr = X[mask_tr], y[mask_tr]
    X_te, y_te = X[mask_te], y[mask_te]

    if not quiet:
        print(f"  G0={g0:.1f}  split={split_name:<8}  "
              f"n_train={mask_tr.sum():,}  n_test={mask_te.sum():,}")

    # ── Fit base models on training section ───────────────────────────────────
    xgb_m, xgb_sc              = _fit_xgb(X_tr, y_tr)
    mlp_m, mlp_sc, dev, lo, hi = _fit_mlp(X_tr, y_tr, epochs=mlp_epochs,
                                            quiet=quiet)

    # ── Fit Ridge on in-sample predictions of training section ────────────────
    xgb_tr = _predict_xgb(xgb_m, xgb_sc, X_tr)
    mlp_tr = _predict_mlp(mlp_m, mlp_sc, dev, X_tr, lo, hi)
    meta_tr = np.column_stack([xgb_tr, mlp_tr])
    ridge = Ridge(alpha=1.0).fit(meta_tr, y_tr)
    stk_tr = ridge.predict(meta_tr).astype(np.float32)

    # ── Predict held-out test section ─────────────────────────────────────────
    xgb_te = _predict_xgb(xgb_m, xgb_sc, X_te)
    mlp_te = _predict_mlp(mlp_m, mlp_sc, dev, X_te, lo, hi)
    meta_te = np.column_stack([xgb_te, mlp_te])
    stk_te = ridge.predict(meta_te).astype(np.float32)

    # ── Compute metrics ───────────────────────────────────────────────────────
    # Full metric suite on the held-out test cells; fast R2-only on the
    # (in-sample) training cells.
    def r2_fast(y_true, y_pred):
        return float(compute_metrics(y_true, y_pred, fast=True)['R2'])

    m_xgb_te = compute_metrics(y_te, xgb_te)
    m_mlp_te = compute_metrics(y_te, mlp_te)
    m_stk_te = compute_metrics(y_te, stk_te)

    return {
        'g0':               float(g0),
        'split':            split_name,
        'n_train':          int(mask_tr.sum()),
        'n_test':           int(mask_te.sum()),
        'xgb_train_r2':     r2_fast(y_tr, xgb_tr),
        'xgb_test_r2':      float(m_xgb_te['R2']),
        'mlp_train_r2':     r2_fast(y_tr, mlp_tr),
        'mlp_test_r2':      float(m_mlp_te['R2']),
        'stacked_train_r2': r2_fast(y_tr, stk_tr),
        'stacked_test_r2':  float(m_stk_te['R2']),
        'xgb_test_metrics':     m_xgb_te,
        'mlp_test_metrics':     m_mlp_te,
        'stacked_test_metrics': m_stk_te,
    }


# ── Pretty-print table ─────────────────────────────────────────────────────────

def print_table(results: list[dict], g0_vals: list[float]) -> None:
    split_names = [s[0] for s in SPLITS]
    col_w = 10

    header = f"{'G0':>5}  {'split':<9}" + "".join(
        f"{'xgb_te':>{col_w}}  {'mlp_te':>{col_w}}  {'stk_te':>{col_w}}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    rec_map = {(r['g0'], r['split']): r for r in results}
    for g0 in g0_vals:
        for split in split_names:
            r = rec_map.get((g0, split))
            if r is None:
                continue
            print(f"{g0:>5.1f}  {split:<9}"
                  f"  {r['xgb_test_r2']:>{col_w}.4f}"
                  f"  {r['mlp_test_r2']:>{col_w}.4f}"
                  f"  {r['stacked_test_r2']:>{col_w}.4f}")
        print()


# ── Heatmap visualization ──────────────────────────────────────────────────────

def plot_heatmaps(results: list[dict], g0_vals: list[float],
                  save_path: str) -> None:
    """3-panel heatmap: XGB | MLP | Stacked, showing held-out test R2."""
    split_names = [s[0] for s in SPLITS]
    n_g0    = len(g0_vals)
    n_split = len(split_names)

    rec_map = {(r['g0'], r['split']): r for r in results}

    def _matrix(key: str) -> np.ndarray:
        mat = np.full((n_g0, n_split), np.nan)
        for i, g0 in enumerate(g0_vals):
            for j, sp in enumerate(split_names):
                r = rec_map.get((g0, sp))
                if r is not None:
                    mat[i, j] = r[key]
        return mat

    panels = [
        ('xgb_test_r2',     'XGB test R2'),
        ('mlp_test_r2',     'MLP test R2'),
        ('stacked_test_r2', 'Stacked test R2'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 4.5))
    fig.suptitle('Intra-cube section experiment: held-out test R2', fontsize=12)

    for ax, (key, title) in zip(axes, panels):
        mat = _matrix(key)
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap='RdYlGn', aspect='auto')
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(n_split))
        ax.set_xticklabels(split_names, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(n_g0))
        ax.set_yticklabels([f'G0={g:.1f}' for g in g0_vals], fontsize=8)

        for i in range(n_g0):
            for j in range(n_split):
                v = mat[i, j]
                if not np.isnan(v):
                    color = 'k' if 0.3 < v < 0.85 else 'white'
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                            fontsize=6.5, color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nHeatmap saved: {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train stacked_sp on a spatial section of a cube, predict the rest.')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='MLP training epochs (default: 100)')
    parser.add_argument('--g0', type=float, default=None,
                        help='Run only for this G0 value (default: all)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-epoch MLP output')
    add_drop_args(parser)
    args = parser.parse_args()

    feat_cols = get_feature_cols(build_drop_set(args))

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    if args.g0 is not None:
        if args.g0 not in g0_vals:
            raise ValueError(f"G0={args.g0} not found. Available: {g0_vals}")
        cube_idx = [g0_vals.index(args.g0)]
    else:
        cube_idx = list(range(len(cubes)))

    split_names = [s[0] for s in SPLITS]
    rng = np.random.default_rng(42)

    results: list[dict] = []

    for ci in cube_idx:
        cube = cubes[ci]
        g0   = g0_vals[ci]
        print(f"\n{'='*60}")
        print(f"Cube G0={g0:.1f}  ({len(cube):,} cells)")
        print(f"{'='*60}")

        for split_name in split_names:
            rec = run_cube_split(cube, g0, split_name,
                                 mlp_epochs=args.mlp_epochs,
                                 quiet=args.quiet,
                                 rng=rng,
                                 feat_cols=feat_cols)
            results.append(rec)
            print(f"  -> xgb_test={rec['xgb_test_r2']:.4f}  "
                  f"mlp_test={rec['mlp_test_r2']:.4f}  "
                  f"stacked_test={rec['stacked_test_r2']:.4f}")

    # ── Print summary ─────────────────────────────────────────────────────────
    if len(cube_idx) > 1:
        print_table(results, g0_vals)

    # ── Save results ──────────────────────────────────────────────────────────
    log_dir = os.path.join('logs', 'intra_cube_section')
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = os.path.join(log_dir, f'run_{ts}.json')
    payload = {
        'timestamp':  ts,
        'mlp_epochs': args.mlp_epochs,
        'splits':     split_names,
        'g0_values':  g0_vals,
        'results':    results,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON log saved: {json_path}")

    # ── Heatmap (only meaningful if more than 1 G0 cube was run) ─────────────
    if len(cube_idx) > 1:
        png_path = os.path.join(log_dir, f'run_{ts}_heatmap.png')
        plot_heatmaps(results, g0_vals, png_path)


if __name__ == '__main__':
    main()
