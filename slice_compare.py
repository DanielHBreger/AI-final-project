#!/usr/bin/env python3
"""
slice_compare.py
================
Select a saved prediction file (.npz) and browse 2D z-slices of the ground
truth, prediction, and difference side-by-side.

Controls
--------
  Slider        : drag to move through z slices (0-127)
  Text box      : type a z value and press Enter to jump to that slice

Usage
-----
  python slice_compare.py
  python slice_compare.py --log-scale
"""

import argparse
import os
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, TextBox

from model_helpers import _preds_to_volume
from data_loader import load_single_cube
from viz_common import select_prediction_file, load_prediction, prepare_display


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Browse 2D z-slices of a saved ens_sp prediction vs ground truth.')
    parser.add_argument('--log-scale', action='store_true',
                        help='Display in log10(nH2) space (default: linear nH2)')
    args = parser.parse_args()

    npz_path = select_prediction_file()
    if not npz_path:
        print('No file selected. Exiting.')
        return

    pred_vol, g0, r2_xgb, r2_mlp, r2_ens, kernels, epochs = load_prediction(npz_path)

    print(f"Loaded: {os.path.basename(npz_path)}")
    print(f"  G0={g0}  XGB R2={r2_xgb:.4f}  MLP R2={r2_mlp:.4f}  ens R2={r2_ens:.4f}")

    # ── Load ground truth ──────────────────────────────────────────────────────
    print(f"\nLoading ground truth for G0={g0}...")
    cube_df   = load_single_cube(g0)
    y_va      = cube_df['log_nH2'].values.astype(np.float32)
    truth_vol = _preds_to_volume(cube_df, y_va)

    truth_display, pred_display, err_display, scale_label = prepare_display(
        truth_vol, pred_vol, args.log_scale)

    p1, p99 = np.percentile(truth_display, [1, 99])
    clim    = (float(p1), float(p99))
    n_z     = truth_display.shape[2]   # 128
    init_z  = n_z // 2

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 7), facecolor='black')
    gs  = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[1, 0.0001],   # image rows tall, controls row thin
        hspace=0.35, wspace=0.08,
        left=0.05, right=0.97, top=0.88, bottom=0.20,
    )
    ax_truth = fig.add_subplot(gs[0, 0])
    ax_pred  = fig.add_subplot(gs[0, 1])
    ax_err   = fig.add_subplot(gs[0, 2])

    # Slider and text box axes placed manually below the panels
    ax_slider  = fig.add_axes([0.10, 0.09, 0.75, 0.03], facecolor='#2a2a2a')
    ax_textbox = fig.add_axes([0.87, 0.07, 0.08, 0.05])

    # ── Initial images ─────────────────────────────────────────────────────────
    def _slice(vol, z):
        return vol[:, :, z].T   # x horizontal, y vertical

    kwargs = dict(cmap='magma', vmin=clim[0], vmax=clim[1],
                  origin='lower', aspect='equal', interpolation='nearest')

    im_truth = ax_truth.imshow(_slice(truth_display, init_z), **kwargs)
    im_pred  = ax_pred .imshow(_slice(pred_display,  init_z), **kwargs)
    im_err   = ax_err  .imshow(_slice(err_display,   init_z), **kwargs)

    # Colorbars
    for im, ax, label in [
        (im_truth, ax_truth, scale_label),
        (im_pred,  ax_pred,  scale_label),
        (im_err,   ax_err,   f'error ({scale_label})'),
    ]:
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(label, color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

    # Axis styling
    for ax, title in [
        (ax_truth, f'Ground Truth   G0={g0}'),
        (ax_pred,  f'ens_sp Prediction   R\u00b2={r2_ens:.4f}\nXGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}'),
        (ax_err,   'Prediction - Truth'),
    ]:
        ax.set_facecolor('black')
        ax.set_title(title, color='white', fontsize=10)
        ax.tick_params(colors='white')
        ax.set_xlabel('x', color='white')
        ax.set_ylabel('y', color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')

    z_title = fig.suptitle(f'z slice = {init_z}', color='white', fontsize=13, y=0.97)

    # ── Slider ─────────────────────────────────────────────────────────────────
    slider = Slider(ax_slider, 'z', 0, n_z - 1, valinit=init_z, valstep=1,
                    color='#666666')
    slider.label.set_color('white')
    slider.valtext.set_color('white')

    # ── Text box ───────────────────────────────────────────────────────────────
    textbox = TextBox(ax_textbox, 'jump to z: ', initial=str(init_z),
                      color='#1e1e1e', hovercolor='#2e2e2e')
    textbox.label.set_color('white')
    textbox.text_disp.set_color('white')

    # ── Update logic ───────────────────────────────────────────────────────────
    _busy = [False]   # mutable flag to prevent mutual recursion

    def _update(z: int) -> None:
        im_truth.set_data(_slice(truth_display, z))
        im_pred .set_data(_slice(pred_display, z))
        im_err  .set_data(_slice(err_display,  z))
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


if __name__ == '__main__':
    main()
