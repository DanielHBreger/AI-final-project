#!/usr/bin/env python3
"""
load_and_compare.py
===================
Select a saved prediction file (.npz) via a file dialog, automatically load
the matching ground-truth cube, and display a 3-panel 3D comparison:

    [ Ground Truth ]  |  [ stacked_sp Prediction ]  |  [ Error ]

Usage
-----
  python load_and_compare.py
  python load_and_compare.py --linear
"""

import argparse
import os
import numpy as np
import pyvista as pv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict_and_visualize import _preds_to_volume
from data_loader import load_single_cube
from viz_common import select_prediction_file, load_prediction, prepare_display


# ── Pyvista helpers ────────────────────────────────────────────────────────────

def _to_pv_grid(vol: np.ndarray, scalar_name: str = 'values') -> pv.ImageData:
    """Wrap a (128, 128, 128) numpy array in a pyvista ImageData cell grid."""
    grid = pv.ImageData()
    grid.dimensions = np.array(vol.shape) + 1   # cell-centered: dims = cells+1
    grid.origin  = (0, 0, 0)
    grid.spacing = (1, 1, 1)
    grid.cell_data[scalar_name] = vol.flatten(order='F')
    return grid


def visualize(truth_vol: np.ndarray, pred_vol: np.ndarray,
              err_vol: np.ndarray, g0: float, r2_xgb: float,
              r2_mlp: float, r2_stacked: float,
              scale_label: str = 'nH2') -> None:
    """Three-panel interactive 3D volume rendering.

    Left   - Ground truth log10(nH2)
    Centre - stacked_sp prediction (Ridge meta-learner over XGB_sp + MLP_sp)
    Right  - Prediction error (pred - truth) in log10 space

    All three panels share a linked camera so rotating one rotates all.
    """
    p1, p99 = np.percentile(truth_vol, [1, 99])
    clim    = (float(p1), float(p99))

    plotter = pv.Plotter(shape=(1, 3), window_size=(1920, 700))
    plotter.background_color = 'black'

    # ── Left: Ground truth ────────────────────────────────────────────────────
    plotter.subplot(0, 0)
    plotter.add_volume(_to_pv_grid(truth_vol), scalars='values',
                       cmap='magma', opacity='linear', clim=clim,
                       scalar_bar_args={'title': scale_label, 'color': 'white'})
    plotter.add_text(f"Ground Truth  G0={g0}", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # ── Centre: Prediction ────────────────────────────────────────────────────
    plotter.subplot(0, 1)
    plotter.add_volume(_to_pv_grid(pred_vol), scalars='values',
                       cmap='magma', opacity='linear', clim=clim,
                       scalar_bar_args={'title': scale_label, 'color': 'white'})
    plotter.add_text(
        f"stacked_sp Prediction  R2={r2_stacked:.4f}\n"
        f"XGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}",
        position='upper_edge', font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # ── Right: Absolute error — 0 transparent, errors bright orange/red ────────
    abs_err_vol = np.abs(err_vol)
    err_max     = float(np.percentile(abs_err_vol, 85))
    err_clim    = (0.0, err_max)
    t           = np.linspace(0.0, 1.0, 256)
    err_opacity = np.power(t, 0.1).tolist()   # very aggressive: t^0.1 is near-step
    err_opacity[0] = 0.0                       # ensure pure zero is fully transparent
    plotter.subplot(0, 2)
    plotter.add_volume(_to_pv_grid(abs_err_vol), scalars='values',
                       cmap='plasma', opacity=err_opacity, clim=err_clim,
                       scalar_bar_args={'title': f'|error| ({scale_label})', 'color': 'white'})
    plotter.add_text("|Prediction - Truth|", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    plotter.link_views()

    print("\nControls: left-click+drag to rotate | right-click+drag to zoom | "
          "middle-click+drag to pan")
    print("All three panels share a linked camera.")
    plotter.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Load a saved stacked_sp prediction and compare against ground truth.')
    parser.add_argument('--linear', action='store_false', dest='log_scale',
                        help='Display in linear nH2 space (default: log10 space)')
    parser.set_defaults(log_scale=True)
    args = parser.parse_args()

    npz_path = select_prediction_file()
    if not npz_path:
        print('No file selected. Exiting.')
        return

    pred_vol, g0, r2_xgb, r2_mlp, r2_stacked, kernels, epochs = load_prediction(npz_path)

    print(f"Loaded: {os.path.basename(npz_path)}")
    print(f"  G0={g0}  spatial_kernels={kernels}  mlp_epochs={epochs}")
    print(f"  XGB R2={r2_xgb:.4f}  MLP R2={r2_mlp:.4f}  stacked R2={r2_stacked:.4f}")

    print(f"\nLoading ground truth for G0={g0}...")
    cube_df   = load_single_cube(g0)
    y_va      = cube_df['log_nH2'].values.astype(np.float32)
    truth_vol = _preds_to_volume(cube_df, y_va)

    truth_display, pred_display, err_display, scale_label = prepare_display(
        truth_vol, pred_vol, args.log_scale)

    print(f"\nLaunching 3D visualizer for G0={g0}...")
    visualize(
        truth_display, pred_display, err_display,
        g0=g0,
        r2_xgb=r2_xgb, r2_mlp=r2_mlp, r2_stacked=r2_stacked,
        scale_label=scale_label,
    )


if __name__ == '__main__':
    main()
