"""
viz_common.py
=============
Shared helpers for viewer scripts (load_and_compare, slice_compare).
"""

import os
import numpy as np


def select_prediction_file() -> str | None:
    """Open a file dialog to select a prediction .npz file.

    Returns the selected path, or None if the user cancels.
    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
    path = filedialog.askopenfilename(
        title='Select prediction file',
        initialdir=pred_dir if os.path.isdir(pred_dir) else '.',
        filetypes=[('NumPy prediction files', '*.npz'), ('All files', '*.*')],
    )
    root.destroy()
    return path or None


def load_prediction(npz_path: str) -> tuple:
    """Load a saved prediction .npz and return its contents.

    Returns (pred_vol, g0, r2_xgb, r2_mlp, r2_ens, kernels, epochs).
    """
    data     = np.load(npz_path)
    pred_vol = data['pred_vol']
    g0       = float(data['g0'])
    r2_xgb   = float(data['r2_xgb'])
    r2_mlp   = float(data['r2_mlp'])
    r2_ens   = float(data['r2_ens'])
    kernels  = data['spatial_kernels'].tolist()
    epochs   = int(data['mlp_epochs'])
    return pred_vol, g0, r2_xgb, r2_mlp, r2_ens, kernels, epochs


def prepare_display(truth_vol: np.ndarray, pred_vol: np.ndarray,
                    log_scale: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Convert log-space volumes to display scale and compute error.

    Returns (truth_display, pred_display, err_display, scale_label).
    """
    if log_scale:
        truth_display = truth_vol.astype(np.float32)
        pred_display  = pred_vol.astype(np.float32)
        scale_label   = 'log10(fh2)'
    else:
        truth_display = np.power(10.0, truth_vol).astype(np.float32)
        pred_display  = np.power(10.0, pred_vol).astype(np.float32)
        scale_label   = 'fh2'

    err_display = (pred_display - truth_display).astype(np.float32)
    return truth_display, pred_display, err_display, scale_label
