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
import vtk
import pyvista as pv

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_helpers import _preds_to_volume
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


def _update_vol_clim(actor, cmap_name: str, p_lo: float, p_hi: float,
                     n: int = 256) -> None:
    """Rewrite the colour and opacity transfer functions of a VTK volume actor.

    Uses a linear opacity ramp (0 at p_lo, 1 at p_hi) and the requested
    matplotlib colormap.  Called on every slider tick to update all three
    panels without re-creating the actors.
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
    # Fully transparent just below the lower bound
    ctf.AddRGBPoint(p_lo - 1e-3, 0.0, 0.0, 0.0)
    otf.AddPoint(p_lo - 1e-3, 0.0)
    for i in range(n):
        t   = i / (n - 1)
        val = p_lo + t * (p_hi - p_lo)
        r, g, b, _ = cm(float(t))
        ctf.AddRGBPoint(val, r, g, b)
        otf.AddPoint(val, float(t))   # linear opacity


def _make_pct_label(renderer, color_rgb: tuple, init_pct: float) -> vtk.vtkTextActor:
    """Attach a vtkTextActor showing the current percentile to renderer."""
    label = vtk.vtkTextActor()
    label.SetInput(f"pmax: {init_pct:.4f}%")
    label.GetTextProperty().SetColor(*color_rgb)
    label.GetTextProperty().SetFontSize(11)
    label.GetTextProperty().SetJustificationToCentered()
    label.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    label.GetPositionCoordinate().SetValue(0.5, 0.12)
    renderer.AddActor2D(label)
    return label


def visualize(truth_vol: np.ndarray, pred_vol: np.ndarray,
              err_vol: np.ndarray, g0: float, r2_xgb: float,
              r2_mlp: float, r2_stacked: float,
              scale_label: str = 'nH2') -> None:
    """Three-panel interactive 3D volume rendering with two independent pmax sliders.

    Left   - Ground truth log10(nH2)
    Centre - stacked_sp prediction
    Right  - Absolute prediction error

    Slider 1 (centre panel, white): shared pmax for truth and prediction panels.
    Slider 2 (right panel,  cyan):  independent pmax for the error panel.

    Slider log scale: value v -> percentile = 100 - 10^(-v)
        v = 0   ->  99th  percentile
        v = 1   ->  99.9th
        v = 2   ->  99.99th
        v = 4   ->  99.9999th
    """
    # VTK requests a 2^20-entry 1D texture for float32 scalars then falls back to
    # the hardware max (32768).  The fallback is visually identical; suppress the
    # harmless warning so it doesn't clutter the console.
    vtk.vtkObject.GlobalWarningDisplayOff()

    abs_err_vol = np.abs(err_vol)

    p1_tp  = float(np.percentile(truth_vol, 1))   # lower bound: truth/pred panels
    p1_err = 0.0                                    # lower bound: error panel

    _INIT_PCT = 99.0
    _SLD_MIN  = float(-np.log10(50.0))   # ~ -1.699  ->  50th percentile
    _SLD_MAX  = 4.0                       # 99.9999th percentile
    _SLD_INIT = 0.0                       # 99th percentile

    init_pmax_tp  = float(max(np.percentile(truth_vol, _INIT_PCT),
                              np.percentile(pred_vol,  _INIT_PCT)))
    init_pmax_err = float(np.percentile(abs_err_vol, _INIT_PCT))
    init_clim_tp  = (p1_tp,  init_pmax_tp)
    init_clim_err = (p1_err, init_pmax_err)

    plotter = pv.Plotter(shape=(1, 3), window_size=(1920, 820))
    plotter.background_color = 'black'

    # ── Left: Ground truth ────────────────────────────────────────────────────
    plotter.subplot(0, 0)
    act_truth = plotter.add_volume(_to_pv_grid(truth_vol), scalars='values',
                                   cmap='magma', opacity='linear', clim=init_clim_tp,
                                   scalar_bar_args={'title': scale_label, 'color': 'white'})
    plotter.add_text(f"Ground Truth  G0={g0}", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()

    # ── Centre: Prediction ────────────────────────────────────────────────────
    plotter.subplot(0, 1)
    act_pred = plotter.add_volume(_to_pv_grid(pred_vol), scalars='values',
                                  cmap='magma', opacity='linear', clim=init_clim_tp,
                                  scalar_bar_args={'title': scale_label, 'color': 'white'})
    plotter.add_text(
        f"stacked_sp Prediction  R2={r2_stacked:.4f}\n"
        f"XGB={r2_xgb:.4f}  MLP={r2_mlp:.4f}",
        position='upper_edge', font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()
    tp_label = _make_pct_label(plotter.renderer, (1, 1, 0), _INIT_PCT)   # yellow

    # ── Right: Absolute error ──────────────────────────────────────────────────
    plotter.subplot(0, 2)
    act_err = plotter.add_volume(_to_pv_grid(abs_err_vol), scalars='values',
                                 cmap='magma', opacity='linear', clim=init_clim_err,
                                 scalar_bar_args={'title': f'|error| ({scale_label})',
                                                  'color': 'white'})
    plotter.add_text("|Prediction - Truth|", position='upper_edge',
                     font_size=12, color='white')
    plotter.add_axes(color='white')
    plotter.view_isometric()
    err_label = _make_pct_label(plotter.renderer, (0, 1, 1), _INIT_PCT)  # cyan

    plotter.link_views()

    # ── Slider 1: truth + prediction pmax (centre panel) ─────────────────────
    plotter.subplot(0, 1)

    def _update_tp(v: float) -> None:
        pct = float(np.clip(100.0 - 10.0 ** (-v), 50.0, 99.9999))
        new_pmax = float(max(np.percentile(truth_vol, pct),
                             np.percentile(pred_vol,  pct)))
        for act in (act_truth, act_pred):
            _update_vol_clim(act, 'magma', p1_tp, new_pmax)
        tp_label.SetInput(f"pmax: {pct:.4f}%")
        plotter.render()

    plotter.add_slider_widget(
        callback=_update_tp,
        rng=[_SLD_MIN, _SLD_MAX],
        value=_SLD_INIT,
        title='truth/pred pmax  (0=99%  1=99.9%  2=99.99%  4=99.9999%)',
        pointa=(0.05, 0.02),
        pointb=(0.95, 0.02),
        style='modern',
        color='white',
        title_color='white',
        fmt='%.2f',
    )
    # Add continuous-drag observer directly via VTK (more reliable than
    # interaction_event='always' which varies by PyVista version).
    _w1 = plotter.slider_widgets[-1]
    _w1.AddObserver(vtk.vtkCommand.InteractionEvent,
                    lambda obj, _: _update_tp(obj.GetRepresentation().GetValue()))

    # ── Slider 2: error pmax (right panel) ────────────────────────────────────
    plotter.subplot(0, 2)

    def _update_err(v: float) -> None:
        pct = float(np.clip(100.0 - 10.0 ** (-v), 50.0, 99.9999))
        new_pmax = float(np.percentile(abs_err_vol, pct))
        _update_vol_clim(act_err, 'magma', p1_err, new_pmax)
        err_label.SetInput(f"pmax: {pct:.4f}%")
        plotter.render()

    plotter.add_slider_widget(
        callback=_update_err,
        rng=[_SLD_MIN, _SLD_MAX],
        value=_SLD_INIT,
        title='error pmax  (0=99%  1=99.9%  2=99.99%  4=99.9999%)',
        pointa=(0.05, 0.02),
        pointb=(0.95, 0.02),
        style='modern',
        color='cyan',
        title_color='cyan',
        fmt='%.2f',
    )
    _w2 = plotter.slider_widgets[-1]
    _w2.AddObserver(vtk.vtkCommand.InteractionEvent,
                    lambda obj, _: _update_err(obj.GetRepresentation().GetValue()))

    print("\nControls: left-click+drag to rotate | right-click+drag to zoom | "
          "middle-click+drag to pan")
    print("All three panels share a linked camera.")
    print("Slider log scale: -1.7=50th pct  0=99th  1=99.9th  2=99.99th  4=99.9999th")
    plotter.show()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Load a saved stacked_sp prediction and compare against ground truth.')
    parser.add_argument('--log', action='store_true', dest='log_scale',
                        help='Display in log10(nH2) space (default: linear nH2)')
    parser.set_defaults(log_scale=False)
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
