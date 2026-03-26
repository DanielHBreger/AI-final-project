#!/usr/bin/env python3
"""
load_and_compare.py
===================
Select a saved prediction file (.npz) via a file dialog, automatically load
the matching ground-truth cube, and display an interactive 3D comparison.

Panel layout adapts to what was saved in the .npz:
  CNN + Ensemble (4 panels): Truth | CNN | Ensemble | CNN Error
  CNN only       (3 panels): Truth | CNN | CNN Error
  Ensemble only  (3 panels): Truth | Ensemble | Error

Two independent opacity/pmax sliders (log-scale percentile):
  Slider 1 (white): shared pmax for truth and all prediction panels.
  Slider 2 (cyan):  independent pmax for the error panel.

Slider log scale: value v -> percentile = 100 - 10^(-v)
  v = -1.7  ->  50th pct  (show everything)
  v =  0    ->  99th pct
  v =  1    ->  99.9th pct
  v =  2    ->  99.99th pct
  v =  4    ->  99.9999th pct (only the very densest cores)

Usage
-----
  python load_and_compare.py
  python load_and_compare.py --log-scale
"""

import argparse
import os
import numpy as np
import vtk
import pyvista as pv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict_and_visualize import _preds_to_volume, _to_pv_grid
from data_loader import load_single_cube
from viz_common import select_prediction_file, load_prediction, prepare_display


# ── VTK transfer-function helpers ──────────────────────────────────────────────

def _update_vol_clim(actor, cmap_name: str, p_lo: float, p_hi: float,
                     n: int = 256) -> None:
    """Rewrite the colour and opacity transfer functions of a VTK volume actor.

    Linear opacity ramp: fully transparent at p_lo, fully opaque at p_hi.
    Called on every slider tick to update panels without re-creating actors.
    """
    import matplotlib.pyplot as _plt
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-3
    prop = actor.GetProperty()
    ctf  = prop.GetRGBTransferFunction(0)
    otf  = prop.GetScalarOpacity(0)
    ctf.RemoveAllPoints()
    otf.RemoveAllPoints()
    cm = _plt.get_cmap(cmap_name)
    ctf.AddRGBPoint(p_lo - 1e-3, 0.0, 0.0, 0.0)
    otf.AddPoint(p_lo - 1e-3, 0.0)
    for i in range(n):
        t   = i / (n - 1)
        val = p_lo + t * (p_hi - p_lo)
        r, g, b, _ = cm(float(t))
        ctf.AddRGBPoint(val, r, g, b)
        otf.AddPoint(val, float(t))


def _make_pct_label(renderer, color_rgb: tuple, init_pct: float) -> vtk.vtkTextActor:
    """Attach a vtkTextActor showing the current percentile cutoff to a renderer."""
    label = vtk.vtkTextActor()
    label.SetInput(f"pmax: {init_pct:.4f}%")
    label.GetTextProperty().SetColor(*color_rgb)
    label.GetTextProperty().SetFontSize(11)
    label.GetTextProperty().SetJustificationToCentered()
    label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    label.GetPositionCoordinate().SetValue(0.5, 0.12)
    renderer.AddActor2D(label)
    return label


# ── Main visualisation ─────────────────────────────────────────────────────────

def visualize(truth_vol: np.ndarray,
              pred_vol: np.ndarray | None,
              err_vol: np.ndarray | None,
              g0: float,
              r2_xgb: float, r2_mlp: float, r2_ens: float,
              pred_cnn_vol: np.ndarray | None = None,
              err_cnn_vol:  np.ndarray | None = None,
              r2_cnn: float | None = None,
              scale_label: str = 'nH2') -> None:
    """Interactive 3D volume rendering with two independent pmax sliders.

    Panel layout:
      has CNN + Ensemble -> 4 panels: Truth | CNN | Ensemble | CNN Error
      has CNN only       -> 3 panels: Truth | CNN | CNN Error
      has Ensemble only  -> 3 panels: Truth | Ensemble | Error
    """
    vtk.vtkObject.GlobalWarningDisplayOff()

    has_cnn = pred_cnn_vol is not None
    has_ens = pred_vol is not None

    # Pick the error volume to show (prefer CNN error when available)
    active_err = np.abs(err_cnn_vol if has_cnn else err_vol)

    n_panels = 4 if (has_cnn and has_ens) else 3
    win_w    = 2400 if n_panels == 4 else 1920

    _SLD_MIN  = float(-np.log10(50.0))   # ~-1.699  -> 50th pct
    _SLD_MAX  = 4.0                       # 99.9999th pct
    _SLD_INIT = 0.0                       # 99th pct
    _INIT_PCT = 99.0

    # Content volumes that share slider 1 (truth + all predictions)
    tp_vols = [truth_vol]
    if has_cnn: tp_vols.append(pred_cnn_vol)
    if has_ens: tp_vols.append(pred_vol)

    p1_tp  = float(np.percentile(truth_vol, 1))
    p1_err = 0.0
    init_pmax_tp  = float(max(np.percentile(v, _INIT_PCT) for v in tp_vols))
    init_pmax_err = float(np.percentile(active_err, _INIT_PCT))

    plotter = pv.Plotter(shape=(1, n_panels), window_size=(win_w, 820))
    plotter.background_color = 'black'

    vol_args = dict(scalars='values', cmap='magma', opacity='linear',
                    scalar_bar_args={'title': scale_label, 'color': 'white'})

    # ── Panel 0: Ground truth ─────────────────────────────────────────────────
    plotter.subplot(0, 0)
    act_truth = plotter.add_volume(_to_pv_grid(truth_vol),
                                   clim=(p1_tp, init_pmax_tp), **vol_args)
    plotter.add_text(f"Ground Truth  G0={g0}", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # Track all content actors so slider 1 updates them together
    content_acts = [act_truth]
    act_err = None
    tp_label = err_label = None

    if has_cnn and has_ens:
        # ── 4-panel layout ────────────────────────────────────────────────────
        plotter.subplot(0, 1)
        act_cnn = plotter.add_volume(_to_pv_grid(pred_cnn_vol),
                                     clim=(p1_tp, init_pmax_tp), **vol_args)
        plotter.add_text(f"CNN (unet_baseline)  R2={r2_cnn:.4f}",
                         position='upper_edge', font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        content_acts.append(act_cnn)

        plotter.subplot(0, 2)
        act_ens = plotter.add_volume(_to_pv_grid(pred_vol),
                                     clim=(p1_tp, init_pmax_tp), **vol_args)
        plotter.add_text(
            f"Ensemble (ens_sp)  R2={r2_ens:.4f}\nXGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}",
            position='upper_edge', font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        tp_label = _make_pct_label(plotter.renderer, (1, 1, 0), _INIT_PCT)
        content_acts.append(act_ens)

        plotter.subplot(0, 3)
        act_err = plotter.add_volume(
            _to_pv_grid(active_err),
            clim=(p1_err, init_pmax_err),
            **{**vol_args, 'scalar_bar_args': {'title': f'|CNN error| ({scale_label})',
                                               'color': 'white'}})
        plotter.add_text("|CNN - Truth|", position='upper_edge',
                         font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        err_label = _make_pct_label(plotter.renderer, (0, 1, 1), _INIT_PCT)

        _sld1_subplot = 2   # slider 1 anchored to ensemble panel
        _sld2_subplot = 3   # slider 2 anchored to error panel

    elif has_cnn:
        # ── 3-panel CNN-only ──────────────────────────────────────────────────
        plotter.subplot(0, 1)
        act_cnn = plotter.add_volume(_to_pv_grid(pred_cnn_vol),
                                     clim=(p1_tp, init_pmax_tp), **vol_args)
        plotter.add_text(f"CNN (unet_baseline)  R2={r2_cnn:.4f}",
                         position='upper_edge', font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        tp_label = _make_pct_label(plotter.renderer, (1, 1, 0), _INIT_PCT)
        content_acts.append(act_cnn)

        plotter.subplot(0, 2)
        act_err = plotter.add_volume(
            _to_pv_grid(active_err),
            clim=(p1_err, init_pmax_err),
            **{**vol_args, 'scalar_bar_args': {'title': f'|CNN error| ({scale_label})',
                                               'color': 'white'}})
        plotter.add_text("|CNN - Truth|", position='upper_edge',
                         font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        err_label = _make_pct_label(plotter.renderer, (0, 1, 1), _INIT_PCT)

        _sld1_subplot = 1
        _sld2_subplot = 2

    else:
        # ── 3-panel ensemble-only ─────────────────────────────────────────────
        plotter.subplot(0, 1)
        act_ens = plotter.add_volume(_to_pv_grid(pred_vol),
                                     clim=(p1_tp, init_pmax_tp), **vol_args)
        plotter.add_text(
            f"ens_sp Prediction  R2={r2_ens:.4f}\nXGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}",
            position='upper_edge', font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        tp_label = _make_pct_label(plotter.renderer, (1, 1, 0), _INIT_PCT)
        content_acts.append(act_ens)

        plotter.subplot(0, 2)
        act_err = plotter.add_volume(
            _to_pv_grid(active_err),
            clim=(p1_err, init_pmax_err),
            **{**vol_args, 'scalar_bar_args': {'title': f'|error| ({scale_label})',
                                               'color': 'white'}})
        plotter.add_text("|Prediction - Truth|", position='upper_edge',
                         font_size=12, color='white')
        plotter.add_axes(color='white')
        plotter.view_isometric()
        err_label = _make_pct_label(plotter.renderer, (0, 1, 1), _INIT_PCT)

        _sld1_subplot = 1
        _sld2_subplot = 2

    plotter.link_views()

    # ── Slider 1: shared pmax for truth + prediction panels ───────────────────
    plotter.subplot(0, _sld1_subplot)

    def _update_tp(v: float) -> None:
        pct     = float(np.clip(100.0 - 10.0 ** (-v), 50.0, 99.9999))
        new_max = float(max(np.percentile(vol, pct) for vol in tp_vols))
        for act in content_acts:
            _update_vol_clim(act, 'magma', p1_tp, new_max)
        if tp_label is not None:
            tp_label.SetInput(f"pmax: {pct:.4f}%")
        plotter.render()

    plotter.add_slider_widget(
        callback=_update_tp,
        rng=[_SLD_MIN, _SLD_MAX],
        value=_SLD_INIT,
        title='truth/pred pmax  (0=99%  1=99.9%  2=99.99%  4=99.9999%)',
        pointa=(0.05, 0.02), pointb=(0.95, 0.02),
        style='modern', color='white', title_color='white', fmt='%.2f',
    )
    _w1 = plotter.slider_widgets[-1]
    _w1.AddObserver(vtk.vtkCommand.InteractionEvent,
                    lambda obj, _: _update_tp(obj.GetRepresentation().GetValue()))

    # ── Slider 2: independent pmax for error panel ────────────────────────────
    plotter.subplot(0, _sld2_subplot)

    def _update_err(v: float) -> None:
        pct     = float(np.clip(100.0 - 10.0 ** (-v), 50.0, 99.9999))
        new_max = float(np.percentile(active_err, pct))
        _update_vol_clim(act_err, 'magma', p1_err, new_max)
        if err_label is not None:
            err_label.SetInput(f"pmax: {pct:.4f}%")
        plotter.render()

    plotter.add_slider_widget(
        callback=_update_err,
        rng=[_SLD_MIN, _SLD_MAX],
        value=_SLD_INIT,
        title='error pmax  (0=99%  1=99.9%  2=99.99%  4=99.9999%)',
        pointa=(0.05, 0.02), pointb=(0.95, 0.02),
        style='modern', color='cyan', title_color='cyan', fmt='%.2f',
    )
    _w2 = plotter.slider_widgets[-1]
    _w2.AddObserver(vtk.vtkCommand.InteractionEvent,
                    lambda obj, _: _update_err(obj.GetRepresentation().GetValue()))

    print("\nControls: left-click+drag to rotate | right-click+drag to zoom | "
          "middle-click+drag to pan")
    print("All panels share a linked camera.")
    print("Slider log scale: -1.7=50th pct  0=99th  1=99.9th  2=99.99th  4=99.9999th")
    plotter.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Load a saved prediction and compare against ground truth.')
    parser.add_argument('--log-scale', action='store_true',
                        help='Display in log10(nH2) space (default: linear nH2)')
    args = parser.parse_args()

    npz_path = select_prediction_file()
    if not npz_path:
        print('No file selected. Exiting.')
        return

    pred_vol, g0, r2_xgb, r2_mlp, r2_ens, kernels, epochs, pred_cnn_vol, r2_cnn = \
        load_prediction(npz_path)

    print(f"Loaded: {os.path.basename(npz_path)}")
    print(f"  G0={g0}  spatial_kernels={kernels}  mlp_epochs={epochs}")
    if r2_ens is not None:
        print(f"  XGB R2={r2_xgb:.4f}  MLP R2={r2_mlp:.4f}  ens R2={r2_ens:.4f}")
    if r2_cnn is not None:
        print(f"  CNN R2={r2_cnn:.4f}")

    print(f"\nLoading ground truth for G0={g0}...")
    cube_df   = load_single_cube(g0)
    y_va      = cube_df['log_nH2'].values.astype(np.float32)
    truth_vol = _preds_to_volume(cube_df, y_va)

    # Convert to display scale; truth is always present
    truth_display, _, _, scale_label = prepare_display(truth_vol, truth_vol, args.log_scale)

    pred_display = err_display = None
    if pred_vol is not None:
        _, pred_display, err_display, _ = prepare_display(truth_vol, pred_vol, args.log_scale)

    pred_cnn_display = err_cnn_display = None
    if pred_cnn_vol is not None:
        _, pred_cnn_display, err_cnn_display, _ = prepare_display(
            truth_vol, pred_cnn_vol, args.log_scale)

    print(f"\nLaunching 3D visualizer for G0={g0}...")
    visualize(
        truth_display, pred_display, err_display,
        g0=g0,
        r2_xgb=r2_xgb or 0.0, r2_mlp=r2_mlp or 0.0, r2_ens=r2_ens or 0.0,
        pred_cnn_vol=pred_cnn_display,
        err_cnn_vol=err_cnn_display,
        r2_cnn=r2_cnn,
        scale_label=scale_label,
    )


if __name__ == '__main__':
    main()
