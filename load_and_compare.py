#!/usr/bin/env python3
"""
load_and_compare.py
===================
Select a saved prediction file (.npz) via a file dialog, automatically load
the matching ground-truth cube, and display the same 3-panel comparison used
by predict_and_visualize.py.

Usage
-----
  python load_and_compare.py
  python load_and_compare.py --log-scale
"""

import argparse
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict_and_visualize import _preds_to_volume, visualize
from data_loader import load_single_cube
from viz_common import select_prediction_file, load_prediction, prepare_display


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Load a saved ens_sp prediction and compare against ground truth.')
    parser.add_argument('--log-scale', action='store_true',
                        help='Display in log10(nH2) space (default: linear nH2)')
    args = parser.parse_args()

    npz_path = select_prediction_file()
    if not npz_path:
        print('No file selected. Exiting.')
        return

    pred_vol, g0, r2_xgb, r2_mlp, r2_ens, kernels, epochs = load_prediction(npz_path)

    print(f"Loaded: {os.path.basename(npz_path)}")
    print(f"  G0={g0}  spatial_kernels={kernels}  mlp_epochs={epochs}")
    print(f"  XGB R2={r2_xgb:.4f}  MLP R2={r2_mlp:.4f}  ens R2={r2_ens:.4f}")

    # ── Load only the matching ground-truth cube ───────────────────────────────
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
        r2_xgb=r2_xgb, r2_mlp=r2_mlp, r2_ens=r2_ens,
        scale_label=scale_label,
    )


if __name__ == '__main__':
    main()
