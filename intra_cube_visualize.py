#!/usr/bin/env python3
"""
intra_cube_visualize.py
=======================
Interactive 4-panel z-slice viewer for the intra-cube spatial interpolation
experiment.

Trains stacked_sp (XGB + MLP + Ridge) on a random x% region of cells from
one G0 cube, then shows an interactive slice viewer with four panels:

  [ Ground Truth ]  |  [ Train Mask ]  |  [ Stacked Prediction ]  |  [ Test Error ]

"Train Mask" displays ground truth only for training cells; test cells appear
black (missing), making the training coverage visible at each z slice.

"Test Error" shows |pred - truth| in dex for held-out test cells only; training
cells are masked black so you can see where the model was uncertain.

Two split types are supported via --split-type:

  rand  (default)  Scattered random voxels (~pct% of all cells selected uniformly).
  box              A single axis-aligned cubic sub-region with volume ~pct% of the
                   full cube, placed at a random position.

In the box viewer, the "Train Mask" panel will show a bright cube-shaped patch
that moves along z as you slide, making the spatial locality very visible.

Spatial features use training-section-only volumes (test positions zeroed before
uniform_filter) so the model cannot exploit neighbourhood leakage.

Usage
-----
  # 5% scattered random voxels, G0=0.8:
  python intra_cube_visualize.py --pct 5

  # 5% contiguous box training region:
  python intra_cube_visualize.py --pct 5 --split-type box

  # 1% box on the hardest cube:
  python intra_cube_visualize.py --pct 1 --g0 6.4 --split-type box

  # Quick demo (fewer MLP epochs):
  python intra_cube_visualize.py --pct 10 --mlp-epochs 20 --quiet
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, TextBox
from sklearn.linear_model import Ridge

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_all_cubes, cube_to_volumes, get_g0_values,
    get_feature_cols, FEATURE_COLS, LOG_TARGET_COL,
)
from classical_models import compute_metrics
from model_helpers import (
    _fit_xgb, _predict_xgb, _fit_mlp, _predict_mlp,
    _compute_spatial_X, _preds_to_volume,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_split_mask(cube, pct: float, split_type: str,
                     seed: int) -> tuple[np.ndarray, dict]:
    """Return (mask, info) for the chosen split type.

    split_type 'rand': scatter ~pct% of cells uniformly at random.
    split_type 'box':  place a single cubic sub-region of ~pct% volume at a
                       random position; info contains side/x0/y0/z0.
    """
    rng = np.random.default_rng(seed)
    if split_type == 'rand':
        n    = len(cube)
        idx  = rng.choice(n, size=max(1, int(n * pct / 100.0)), replace=False)
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        return mask, {}
    if split_type == 'box':
        side = max(1, min(128, round((pct / 100.0 * 128 ** 3) ** (1 / 3))))
        x0   = int(rng.integers(0, 128 - side + 1))
        y0   = int(rng.integers(0, 128 - side + 1))
        z0   = int(rng.integers(0, 128 - side + 1))
        ix   = cube['ix'].values.astype(int) - 1
        iy   = cube['iy'].values.astype(int) - 1
        iz   = cube['iz'].values.astype(int) - 1
        mask = ((ix >= x0) & (ix < x0 + side) &
                (iy >= y0) & (iy < y0 + side) &
                (iz >= z0) & (iz < z0 + side))
        return mask, {'side': side, 'x0': x0, 'y0': y0, 'z0': z0}
    raise ValueError(f"Unknown split_type: {split_type!r}")


def _make_masked_vols(cube, vols: dict, mask_tr: np.ndarray) -> dict:
    """Return copies of `vols` with test-section cell positions set to 0.

    Neighbourhood means (uniform_filter) computed from these masked volumes
    reflect only the physical environment at training locations.
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


def _nan_volume(cube, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build a 128^3 float32 volume with NaN at positions where mask=False."""
    ix = cube['ix'].values.astype(int) - 1
    iy = cube['iy'].values.astype(int) - 1
    iz = cube['iz'].values.astype(int) - 1
    vol = np.full((128, 128, 128), np.nan, dtype=np.float32)
    vol[ix[mask], iy[mask], iz[mask]] = y[mask]
    return vol


# ── Train and build volumes ────────────────────────────────────────────────────

def run_and_build(cube, pct: float, split_type: str,
                  mlp_epochs: int, quiet: bool, seed: int,
                  feat_cols: list = FEATURE_COLS):
    """Fit stacked_sp on a pct% split and return display volumes.

    Returns
    -------
    truth_vol        (128,128,128)  full ground truth in log10(nH2)
    masked_truth_vol (128,128,128)  NaN for test cells, truth for train cells
    pred_vol         (128,128,128)  stacked prediction for ALL cells
    err_vol          (128,128,128)  |pred-truth| for test cells, NaN for train
    r2_xgb, r2_mlp, r2_stacked     test-set R2
    n_train, n_test, split_info
    """
    mask_tr, split_info = _make_split_mask(cube, pct, split_type, seed)
    mask_te = ~mask_tr
    n_train, n_test = int(mask_tr.sum()), int(mask_te.sum())
    actual_pct = 100.0 * n_train / len(cube)
    print(f"  split_type={split_type}  n_train={n_train:,}  n_test={n_test:,}  "
          f"({actual_pct:.2f}% train)")
    if split_info:
        print(f"  box: side={split_info['side']}  "
              f"origin=({split_info['x0']},{split_info['y0']},{split_info['z0']})")

    # ── Feature matrix (training-section-only spatial features) ───────────────
    vols    = cube_to_volumes(cube, feat_cols)
    vols_tr = _make_masked_vols(cube, vols, mask_tr)
    X_sp    = _compute_spatial_X([cube], [vols_tr], feat_cols)
    X_flat  = cube[feat_cols].values.astype(np.float32)
    X       = np.concatenate([X_flat, X_sp], axis=1)                  # (N, 60)
    y       = cube[LOG_TARGET_COL].values.astype(np.float32)

    X_tr, y_tr = X[mask_tr], y[mask_tr]
    X_te, y_te = X[mask_te], y[mask_te]

    # ── Fit base models ────────────────────────────────────────────────────────
    print("\n[xgb_standard_sp]")
    xgb_m, xgb_sc              = _fit_xgb(X_tr, y_tr)
    print(f"[mlp_wide_sp] {mlp_epochs} epochs")
    mlp_m, mlp_sc, dev, lo, hi = _fit_mlp(X_tr, y_tr,
                                            epochs=mlp_epochs, quiet=quiet)

    # ── Ridge on in-sample predictions ────────────────────────────────────────
    xgb_tr = _predict_xgb(xgb_m, xgb_sc, X_tr)
    mlp_tr = _predict_mlp(mlp_m, mlp_sc, dev, X_tr, lo, hi)
    ridge  = Ridge(alpha=1.0).fit(np.column_stack([xgb_tr, mlp_tr]), y_tr)
    print(f"Ridge: XGB={ridge.coef_[0]:.3f}  MLP={ridge.coef_[1]:.3f}  "
          f"intercept={ridge.intercept_:.3f}")

    # ── Predict on ALL cells ───────────────────────────────────────────────────
    xgb_all = _predict_xgb(xgb_m, xgb_sc, X)
    mlp_all = _predict_mlp(mlp_m, mlp_sc, dev, X, lo, hi)
    stk_all = ridge.predict(np.column_stack([xgb_all, mlp_all])).astype(np.float32)

    # ── Test R2 ───────────────────────────────────────────────────────────────
    r2_xgb     = float(compute_metrics(y_te, xgb_all[mask_te])['R2'])
    r2_mlp     = float(compute_metrics(y_te, mlp_all[mask_te])['R2'])
    r2_stacked = float(compute_metrics(y_te, stk_all[mask_te])['R2'])
    print(f"\nTest R2:  XGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}  "
          f"Stacked={r2_stacked:.4f}")

    # ── Build display volumes ──────────────────────────────────────────────────
    #   truth_vol        -- full cube, ground truth
    #   masked_truth_vol -- NaN where mask_te (only training cells visible)
    #   pred_vol         -- stacked prediction, all cells
    #   err_vol          -- |pred-truth| for test cells, NaN for train cells
    truth_vol        = _nan_volume(cube, y, np.ones(len(cube), dtype=bool))
    masked_truth_vol = _nan_volume(cube, y, mask_tr)
    pred_vol         = _preds_to_volume(cube, stk_all)

    ix = cube['ix'].values.astype(int) - 1
    iy = cube['iy'].values.astype(int) - 1
    iz = cube['iz'].values.astype(int) - 1
    err_vol = np.full((128, 128, 128), np.nan, dtype=np.float32)
    err_vol[ix[mask_te], iy[mask_te], iz[mask_te]] = np.abs(stk_all[mask_te] - y_te)

    return (truth_vol, masked_truth_vol, pred_vol, err_vol,
            r2_xgb, r2_mlp, r2_stacked, n_train, n_test, split_info)


# ── Interactive viewer ─────────────────────────────────────────────────────────

def launch_viewer(truth_vol, masked_truth_vol, pred_vol, err_vol,
                  g0: float, pct: float, split_type: str,
                  r2_xgb: float, r2_mlp: float, r2_stacked: float,
                  n_train: int, n_test: int, split_info: dict) -> None:
    """Display an interactive 4-panel z-slice viewer."""
    # Colour limits from ground truth
    valid   = truth_vol[np.isfinite(truth_vol)]
    p1, p99 = float(np.percentile(valid, 1)), float(np.percentile(valid, 99))

    cmap_main = plt.cm.magma.copy()
    cmap_main.set_bad('black')

    n_z    = 128
    init_z = n_z // 2

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 7), facecolor='black')
    gs  = gridspec.GridSpec(
        2, 4,
        figure=fig,
        height_ratios=[1, 0.0001],
        hspace=0.40, wspace=0.08,
        left=0.04, right=0.98, top=0.88, bottom=0.18,
    )
    ax_truth = fig.add_subplot(gs[0, 0])
    ax_mask  = fig.add_subplot(gs[0, 1])
    ax_pred  = fig.add_subplot(gs[0, 2])
    ax_err   = fig.add_subplot(gs[0, 3])

    ax_slider  = fig.add_axes([0.08, 0.09, 0.78, 0.03], facecolor='#2a2a2a')
    ax_textbox = fig.add_axes([0.88, 0.07, 0.08, 0.05])

    def _sl(vol, z):
        return vol[:, :, z].T   # x horizontal, y vertical

    kw_main = dict(cmap=cmap_main, vmin=p1, vmax=p99,
                   origin='lower', aspect='equal', interpolation='nearest')

    im_truth = ax_truth.imshow(_sl(truth_vol,        init_z), **kw_main)
    im_mask  = ax_mask .imshow(_sl(masked_truth_vol, init_z), **kw_main)
    im_pred  = ax_pred .imshow(_sl(pred_vol,         init_z), **kw_main)
    im_err   = ax_err  .imshow(_sl(err_vol,          init_z), **kw_main)

    for im, ax, lbl in [
        (im_truth, ax_truth, 'log10(nH2)'),
        (im_mask,  ax_mask,  'log10(nH2)'),
        (im_pred,  ax_pred,  'log10(nH2)'),
        (im_err,   ax_err,   'error (dex)'),
    ]:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(lbl, color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    if split_type == 'box':
        side = split_info['side']
        x0, y0, z0 = split_info['x0'], split_info['y0'], split_info['z0']
        mask_title = (f'Train Box ({pct:.1f}%,  {n_train:,} cells)\n'
                      f'side={side}  origin=({x0},{y0},{z0})')
    else:
        mask_title = f'Train Mask ({pct:.1f}%,  {n_train:,} cells)\nBlack = test cells'

    titles = [
        f'Ground Truth   G0={g0}',
        mask_title,
        (f'Stacked Prediction (all cells)\n'
         f'R\u00b2={r2_stacked:.4f}   XGB={r2_xgb:.4f}   MLP={r2_mlp:.4f}'),
        f'|Pred - Truth|  test cells only\n({n_test:,} test cells)',
    ]
    for ax, title in zip([ax_truth, ax_mask, ax_pred, ax_err], titles):
        ax.set_facecolor('black')
        ax.set_title(title, color='white', fontsize=9)
        ax.tick_params(colors='white')
        ax.set_xlabel('x', color='white')
        ax.set_ylabel('y', color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

    z_title = fig.suptitle(f'z slice = {init_z}', color='white',
                            fontsize=13, y=0.97)

    # ── Slider + textbox ───────────────────────────────────────────────────────
    slider  = Slider(ax_slider, 'z', 0, n_z - 1, valinit=init_z, valstep=1,
                     color='#666666')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    textbox = TextBox(ax_textbox, 'jump to z: ', initial=str(init_z),
                      color='#1e1e1e', hovercolor='#2e2e2e')
    textbox.label.set_color('white')
    textbox.text_disp.set_color('white')

    _busy = [False]

    def _update(z: int) -> None:
        im_truth.set_data(_sl(truth_vol,        z))
        im_mask .set_data(_sl(masked_truth_vol, z))
        im_pred .set_data(_sl(pred_vol,         z))
        im_err  .set_data(_sl(err_vol,          z))
        z_title.set_text(f'z slice = {z}')
        fig.canvas.draw_idle()

    def on_slider(val: float) -> None:
        if _busy[0]:
            return
        z = int(slider.val)
        _busy[0] = True
        textbox.set_val(str(z))
        _busy[0] = False
        _update(z)

    def on_textbox(text: str) -> None:
        if _busy[0]:
            return
        try:
            z = int(np.clip(int(text), 0, n_z - 1))
        except ValueError:
            return
        _busy[0] = True
        slider.set_val(z)
        _busy[0] = False
        _update(z)

    slider .on_changed(on_slider)
    textbox.on_submit(on_textbox)

    plt.show()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Visualise intra-cube spatial interpolation for a random x% split.')
    parser.add_argument('--pct', type=float, required=True,
                        help='Training percentage, e.g. 5 for 5%%. Range: 0.1-99.')
    parser.add_argument('--split-type', choices=['rand', 'box'], default='rand',
                        help='rand: scattered random voxels (default). '
                             'box: single contiguous cubic sub-region.')
    parser.add_argument('--g0', type=float, default=0.8,
                        help='G0 cube to use (default: 0.8).')
    parser.add_argument('--mlp-epochs', type=int, default=100,
                        help='MLP training epochs (default: 100).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the split (default: 42).')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-epoch MLP output.')
    parser.add_argument('--no-fh2', action='store_true',
                        help='Exclude log_fh2 from input features.')
    args = parser.parse_args()

    feat_cols = get_feature_cols(args.no_fh2)

    if not (0.1 <= args.pct <= 99.0):
        print(f"ERROR: --pct must be between 0.1 and 99. Got {args.pct}",
              file=sys.stderr)
        sys.exit(1)

    print("Loading cubes...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)

    if args.g0 not in g0_vals:
        print(f"ERROR: G0={args.g0} not found. Available: {g0_vals}",
              file=sys.stderr)
        sys.exit(1)

    cube = cubes[g0_vals.index(args.g0)]
    print(f"Cube G0={args.g0}: {len(cube):,} cells")

    print(f"\nTraining on {args.pct:.1f}% {args.split_type} split (seed={args.seed})...")
    results = run_and_build(cube, args.pct, args.split_type,
                             args.mlp_epochs, args.quiet, args.seed,
                             feat_cols=feat_cols)
    truth_vol, masked_truth_vol, pred_vol, err_vol = results[:4]
    r2_xgb, r2_mlp, r2_stacked, n_train, n_test, split_info = results[4:]

    print("\nLaunching viewer (close window to exit)...")
    launch_viewer(truth_vol, masked_truth_vol, pred_vol, err_vol,
                  args.g0, args.pct, args.split_type,
                  r2_xgb, r2_mlp, r2_stacked,
                  n_train, n_test, split_info)


if __name__ == '__main__':
    main()
