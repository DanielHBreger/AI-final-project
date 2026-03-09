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
import glob
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import filedialog

from predict_and_visualize import _preds_to_volume, visualize


_EPS  = 1e-30
_COLS = ['ix', 'iy', 'iz', 'nH', 'nH2', 'T', 'vx', 'vy', 'vz',
         'nHp', 'ext', 'fh2', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']


def _load_cube_for_g0(g0: float, data_root: str = 'icedrive-dl-182bd/UVonly'):
    """Load a single simulation cube matching the given G0 value."""
    import pandas as pd
    dir_name = f"{g0:.1f}".replace('.', '_')
    matches  = glob.glob(os.path.join(data_root, dir_name, '*.csv'))
    if not matches:
        raise FileNotFoundError(
            f"No CSV found for G0={g0} (looked for: {data_root}/{dir_name}/*.csv)")
    df = pd.read_csv(matches[0], sep=r'\s+', header=None, skiprows=1)
    df.columns = _COLS
    df = df.drop(columns=['nH2'])
    df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Load a saved ens_sp prediction and compare against ground truth.')
    parser.add_argument('--log-scale', action='store_true',
                        help='Display in log10(fh2) space (default: linear fh2)')
    args = parser.parse_args()

    # ── File selection ─────────────────────────────────────────────────────────
    root = tk.Tk()
    root.withdraw()
    pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
    npz_path = filedialog.askopenfilename(
        title='Select prediction file',
        initialdir=pred_dir if os.path.isdir(pred_dir) else '.',
        filetypes=[('NumPy prediction files', '*.npz'), ('All files', '*.*')],
    )
    root.destroy()

    if not npz_path:
        print('No file selected. Exiting.')
        return

    # ── Load prediction ────────────────────────────────────────────────────────
    data     = np.load(npz_path)
    pred_vol = data['pred_vol']          # log10(fh2), 128^3
    g0       = float(data['g0'])         # float64 — exact
    r2_xgb   = float(data['r2_xgb'])
    r2_mlp   = float(data['r2_mlp'])
    r2_ens   = float(data['r2_ens'])
    kernels  = data['spatial_kernels'].tolist()
    epochs   = int(data['mlp_epochs'])

    print(f"Loaded: {os.path.basename(npz_path)}")
    print(f"  G0={g0}  spatial_kernels={kernels}  mlp_epochs={epochs}")
    print(f"  XGB R2={r2_xgb:.4f}  MLP R2={r2_mlp:.4f}  ens R2={r2_ens:.4f}")

    # ── Load only the matching ground-truth cube ───────────────────────────────
    print(f"\nLoading ground truth for G0={g0}...")
    cube_df   = _load_cube_for_g0(g0)
    y_va      = cube_df['log_fh2'].values.astype(np.float32)
    truth_vol = _preds_to_volume(cube_df, y_va)

    # ── Scale selection ────────────────────────────────────────────────────────
    if args.log_scale:
        truth_display = truth_vol
        pred_display  = pred_vol
        scale_label   = 'log10(fh2)'
    else:
        truth_display = np.power(10.0, truth_vol).astype(np.float32)
        pred_display  = np.power(10.0, pred_vol).astype(np.float32)
        scale_label   = 'fh2'

    err_display = (pred_display - truth_display).astype(np.float32)

    print(f"\nLaunching 3D visualizer for G0={g0}...")
    visualize(
        truth_display, pred_display, err_display,
        g0=g0,
        r2_xgb=r2_xgb, r2_mlp=r2_mlp, r2_ens=r2_ens,
        scale_label=scale_label,
    )


if __name__ == '__main__':
    main()
