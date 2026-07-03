#!/usr/bin/env python3
"""
statistical_analysis.py
========================
Load all stacked_sp prediction .npz files from a directory, compare each
against its ground-truth cube, compute statistics, and save six PNG figures.

Figures produced
----------------
  fig1_summary.png         -- R2, R2_lin, RMSE, fraction-within bars (single row)
  fig2_scatter.png         -- 2-D density scatter (truth vs prediction) per G0
  fig3_error_dist.png      -- signed-error histograms with Gaussian fits per G0
  fig4_stratified.png      -- density-stratified R2 curves + error-threshold lines
  fig5_spatial.png         -- 2-D mean|error| projections (xy / xz / yz) per G0
  fig6_distributions.png   -- truth vs prediction density histograms per G0
  fig7_massbudget.png      -- mass ratio + bias (raw vs recalibrated) and
                              phase-conditional R2 per G0
  fig8_slices.png          -- mid-plane truth | prediction | |error| slices per G0

Usage
-----
  python statistical_analysis.py
  python statistical_analysis.py --pred-dir predictions/
  python statistical_analysis.py --pred-dir . --save-dir analysis_output/
"""

import argparse
import glob
import os
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from scipy.stats import norm as sp_norm, wasserstein_distance, ks_2samp
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from model_helpers import _preds_to_volume
from classical_models import PHASE_SPLIT
from data_loader import load_single_cube
from viz_common import load_prediction


# ── Journal style ──────────────────────────────────────────────────────────────

def _apply_journal_style() -> None:
    """
    Apply ApJ / MNRAS figure conventions:
      serif STIX fonts, white background, box axes with inward ticks on all
      four sides, minor ticks visible, no grid, 300 dpi output.
    """
    plt.rcParams.update({
        # Typography — STIX matches Computer Modern (standard for physics journals)
        'font.family':           'serif',
        'font.serif':            ['STIXGeneral', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset':      'stix',
        'font.size':             8,
        'axes.titlesize':        8,
        'axes.labelsize':        8,
        'xtick.labelsize':       7,
        'ytick.labelsize':       7,
        'legend.fontsize':       7,

        # Figure / axes background
        'figure.facecolor':      'white',
        'axes.facecolor':        'white',
        'axes.linewidth':        0.6,
        'axes.grid':             False,

        # Ticks — inward on all four sides, major + minor
        'xtick.direction':       'in',
        'ytick.direction':       'in',
        'xtick.top':             True,
        'ytick.right':           True,
        'xtick.minor.visible':   True,
        'ytick.minor.visible':   True,
        'xtick.major.width':     0.6,
        'ytick.major.width':     0.6,
        'xtick.minor.width':     0.4,
        'ytick.minor.width':     0.4,
        'xtick.major.size':      3.5,
        'ytick.major.size':      3.5,
        'xtick.minor.size':      2.0,
        'ytick.minor.size':      2.0,

        # Lines and markers
        'lines.linewidth':       1.0,
        'lines.markersize':      3.5,

        # Legend
        'legend.frameon':        True,
        'legend.framealpha':     0.92,
        'legend.edgecolor':      '0.75',
        'legend.fancybox':       False,
        'legend.borderpad':      0.35,
        'legend.handlelength':   1.6,
        'legend.labelspacing':   0.30,
        'legend.columnspacing':  0.8,

        # Output
        'savefig.dpi':           300,
        'savefig.bbox':          'tight',
        'savefig.pad_inches':    0.03,
    })


_apply_journal_style()

# ── Colour palette ─────────────────────────────────────────────────────────────

_G0_LIST   = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
# Viridis is perceptually uniform, colorblind-safe, and prints well in grey.
_G0_COLORS = dict(zip(_G0_LIST,
                      plt.cm.viridis(np.linspace(0.08, 0.92, len(_G0_LIST)))))

_C_TRUTH  = '#2166ac'   # blue  (RdBu)
_C_PRED   = '#d6604d'   # red-orange (RdBu)

# Four-color scheme for threshold lines (Tableau-inspired, colorblind-safe)
_THR_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

_PANEL_ALPHA = 'abcdefghijklmnopqrstuvwxyz'


# ── Helpers ────────────────────────────────────────────────────────────────────

def _panel(ax: plt.Axes, idx: int,
           dx: float = -0.13, dy: float = 1.03) -> None:
    """Stamp bold panel letter '(a)', '(b)' ... in the top-left corner."""
    ax.text(dx, dy, f'({_PANEL_ALPHA[idx]})',
            transform=ax.transAxes,
            fontsize=8, fontweight='bold',
            va='bottom', ha='left')


def _stats_box(ax: plt.Axes, r: dict) -> None:
    """Annotate a scatter panel with R2, Pearson r, and bias."""
    txt = (rf"$R^2 = {r['r2']:.4f}$"   + '\n'
           + rf"$r = {r['corr']:.4f}$"  + '\n'
           + rf"bias $= {r['bias']:+.3f}$ dex")
    ax.text(0.04, 0.96, txt,
            transform=ax.transAxes,
            fontsize=6.5, va='top', ha='left', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='0.7', linewidth=0.5, alpha=0.92))


def _find_npz(directory: str) -> list[str]:
    """Stacked-ensemble prediction files in `directory`.

    Files without a 'pred_vol' array (e.g. CNN predictions, which store
    'pred_cnn_vol') are skipped.  When several files exist for the same G0
    (pred_g0_<G0>_<timestamp>.npz), only the most recent is kept so reruns
    don't mix old and new volumes."""
    import re
    paths = sorted(glob.glob(os.path.join(directory, '*.npz')))
    by_g0: dict[str, str] = {}
    others: list[str] = []
    skipped = 0
    for p in paths:
        try:
            with np.load(p) as d:
                if 'pred_vol' not in d.files:
                    skipped += 1
                    continue
        except Exception:
            skipped += 1
            continue
        m = re.search(r'pred_g0_([\d.]+)_\d', os.path.basename(p))
        if m:
            by_g0[m.group(1)] = p   # sorted() + timestamped names -> last = latest
        else:
            others.append(p)
    kept = sorted(by_g0.values()) + others
    n_dup = len(paths) - len(kept) - skipped
    if skipped:
        print(f'  ({skipped} non-ensemble .npz file(s) without pred_vol skipped)')
    if n_dup > 0:
        print(f'  ({n_dup} older prediction file(s) for duplicate G0 values ignored)')
    return kept


def _choose_directory() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        pred_default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'predictions')
        start = pred_default if os.path.isdir(pred_default) else '.'
        d = filedialog.askdirectory(
            title='Select directory with .npz prediction files',
            initialdir=start)
        root.destroy()
        return d or '.'
    except Exception as exc:
        print(f'  (directory chooser unavailable: {exc})')
        return '.'


def _save_fig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path)          # dpi / bbox controlled by rcParams
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Statistics ─────────────────────────────────────────────────────────────────

def _compute_record(truth_vol: np.ndarray,
                    pred_vol:  np.ndarray) -> dict:
    y_t = truth_vol.flatten().astype(np.float64)
    y_p = pred_vol.flatten().astype(np.float64)
    err = y_p - y_t

    r2   = float(r2_score(y_t, y_p))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    lo     = float(y_t.min()) - 1.0
    hi     = float(y_t.max()) + 1.0
    r2_lin = float(r2_score(10 ** y_t, 10 ** np.clip(y_p, lo, hi)))
    corr   = float(np.corrcoef(y_t, y_p)[0, 1])

    frac = {t: float(np.mean(np.abs(err) < t)) for t in (0.05, 0.1, 0.2, 0.5)}

    strat_pcts = np.linspace(0, 99, 200)
    strat_r2   = _density_stratified_r2(y_t, y_p, strat_pcts)

    e_vol = np.abs(pred_vol - truth_vol).astype(np.float32)
    pxy   = e_vol.mean(axis=2)
    pxz   = e_vol.mean(axis=1)
    pyz   = e_vol.mean(axis=0)

    n_bins    = 10
    bin_edges = np.percentile(y_t, np.linspace(0, 100, n_bins + 1))
    bin_mae   = np.zeros(n_bins, dtype=np.float32)
    bin_label = []
    for i in range(n_bins):
        mask = (y_t >= bin_edges[i]) & (y_t < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_mae[i] = float(np.mean(np.abs(err[mask])))
        bin_label.append(f'{bin_edges[i]:.1f}')

    # Mass budget (clipped, as in compute_metrics) and phase-conditional R2
    mass_ratio = float(np.sum(10.0 ** np.clip(y_p, lo, hi))
                       / np.sum(10.0 ** y_t))
    mol = y_t > PHASE_SPLIT
    r2_mol = (float(r2_score(y_t[mol], y_p[mol]))
              if mol.sum() >= 10 else np.nan)
    r2_dif = (float(r2_score(y_t[~mol], y_p[~mol]))
              if (~mol).sum() >= 10 else np.nan)
    bias_mol = float(np.mean(err[mol])) if mol.sum() >= 10 else np.nan

    # Mid-plane slices (z = N/2) for the truth | prediction | error figure
    z_mid = truth_vol.shape[2] // 2
    slice_t = truth_vol[:, :, z_mid].astype(np.float32)
    slice_p = pred_vol[:, :, z_mid].astype(np.float32)

    return dict(
        r2=r2, rmse=rmse, mae=mae, bias=bias, r2_lin=r2_lin, corr=corr,
        n_cells=len(y_t),
        truth_mean=float(y_t.mean()), truth_std=float(y_t.std()),
        pred_mean=float(y_p.mean()),  pred_std=float(y_p.std()),
        frac=frac,
        mass_ratio=mass_ratio,
        r2_mol=r2_mol, r2_dif=r2_dif, bias_mol=bias_mol,
        f_mol=float(np.mean(mol)),
        slice_t=slice_t, slice_p=slice_p,
        strat_pcts=strat_pcts, strat_r2=strat_r2,
        pxy=pxy, pxz=pxz, pyz=pyz,
        bin_mae=bin_mae, bin_label=bin_label,
        y_t=y_t.astype(np.float32),
        y_p=y_p.astype(np.float32),
        err=err.astype(np.float32),
    )


def _density_stratified_r2(y_t: np.ndarray,
                            y_p: np.ndarray,
                            pcts: np.ndarray) -> np.ndarray:
    out = np.full(len(pcts), np.nan, dtype=np.float32)
    for i, p in enumerate(pcts):
        thr  = float(np.percentile(y_t, p))
        mask = y_t >= thr
        if mask.sum() >= 10:
            out[i] = float(r2_score(y_t[mask], y_p[mask]))
    return out


# ── Figure 1: Metric summary ───────────────────────────────────────────────────

def fig1_summary(recs: list[dict], save_dir: str) -> None:
    """
    Three bar charts in a single row: R2, R2_lin, and fraction of cells
    within 0.1 dex.  Bars are coloured by G0 value using the viridis
    palette (dark = low G0, bright = high G0).
    """
    g0s    = [r['g0']  for r in recs]
    x      = np.arange(len(recs))
    colors = [_G0_COLORS[g] for g in g0s]
    xlbls  = [rf'$G_0={g:.1f}$' for g in g0s]

    metric_defs = [
        ('r2',     r'$R^2$  (log-space)',                      r'$R^2$', None, '.4f'),
        ('r2_lin', r'$R^2$  linear $n_{\rm H_2}$ (clipped)', r'$R^2$', None, '.4f'),
    ]
    frac_01 = [r['frac'][0.1] * 100 for r in recs]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))

    for idx, (ax, (key, title, ylabel, ref, fmt)) in enumerate(
            zip(axes, metric_defs)):
        vals  = [r[key] for r in recs]
        bars  = ax.bar(x, vals, color=colors, edgecolor='k',
                       linewidth=0.4, width=0.65, zorder=3)
        if ref is not None:
            ax.axhline(ref, color='0.3', ls='--', lw=0.7, zorder=2)
        mean_v = np.mean(vals)
        ax.axhline(mean_v, color='k', ls=':', lw=0.9, zorder=4,
                   label=f'mean = {mean_v:{fmt}}')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, rotation=30, ha='right')
        ax.legend(handlelength=1.2)
        # Value annotations — small text above each bar
        y_range = max(abs(v) for v in vals) if vals else 1.0
        for bar, v in zip(bars, vals):
            offset = y_range * 0.025 + abs(bar.get_height()) * 0.01
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=5.5)
        # Expand ylim so annotations don't clip the spine
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.20)
        _panel(ax, idx)

    # Fourth panel: fraction within 0.1 dex
    ax   = axes[2]
    bars = ax.bar(x, frac_01, color=colors, edgecolor='k',
                  linewidth=0.4, width=0.65, zorder=2)
    mean_f = np.mean(frac_01)
    ax.axhline(mean_f, color='k', ls=':', lw=0.9, zorder=4,
               label=f'mean = {mean_f:.1f}%')
    ax.set_title(r'Cells with $|\epsilon| < 0.1$ dex')
    ax.set_ylabel(r'Fraction (\%)')
    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, rotation=30, ha='right')
    for bar, v in zip(bars, frac_01):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f'{v:.1f}', ha='center', va='bottom', fontsize=5.5)
    ax.legend(handlelength=1.2)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + (yhi - ylo) * 0.20)
    _panel(ax, 2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save_fig(fig, os.path.join(save_dir, 'fig1_summary.png'))


# ── Figure 2: Scatter ──────────────────────────────────────────────────────────

def fig2_scatter(recs: list[dict], save_dir: str) -> None:
    """
    2-D density scatter of truth vs prediction for each G0 value.
    Points are binned into a 100x100 histogram and rendered as an image
    with a logarithmic colour scale (viridis).  The identity line y=x and
    a compact statistics annotation are overlaid.
    """
    n     = len(recs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.55, nrows * 3.65))
    axes_flat = np.array(axes).flatten()

    for idx, (r, ax) in enumerate(zip(recs, axes_flat)):
        y_t, y_p = r['y_t'], r['y_p']
        lo = float(min(y_t.min(), y_p.min())) - 0.1
        hi = float(max(y_t.max(), y_p.max())) + 0.1

        # 2-D histogram rendered with LogNorm — standard in astrophysics papers
        h, _, _ = np.histogram2d(y_t, y_p, bins=100,
                                   range=[[lo, hi], [lo, hi]])
        h_masked = np.ma.masked_where(h == 0, h)
        cmap = plt.cm.viridis.copy()
        cmap.set_bad('white')
        im = ax.imshow(h_masked.T,
                       origin='lower', aspect='equal',
                       norm=LogNorm(vmin=1, vmax=h.max()),
                       cmap=cmap,
                       extent=[lo, hi, lo, hi],
                       rasterized=True)

        # Identity line — white over viridis is visible throughout
        ax.plot([lo, hi], [lo, hi], color='white', lw=1.4, zorder=5)
        ax.plot([lo, hi], [lo, hi], color='0.2',   lw=0.6, ls='--', zorder=6)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.set_xlabel(r'Truth, $y$  (dex)')
        ax.set_ylabel(r'Prediction, $\hat{y}$  (dex)')
        ax.set_title(rf'$G_0 = {r["g0"]:.1f}$')
        _stats_box(ax, r)
        _panel(ax, idx)

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(r'$N_{\rm cells}$', fontsize=6.5)
        cb.ax.tick_params(labelsize=6)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig2_scatter.png'))


# ── Figure 3: Error distributions ─────────────────────────────────────────────

def fig3_error_dist(recs: list[dict], save_dir: str) -> None:
    """
    Probability density of the signed prediction error
    epsilon = hat{y} - y for each G0 value.
    The filled step histogram is overlaid with a Gaussian fit N(mu, sigma);
    the bias (vertical dashed) and zero (vertical solid) are indicated.
    """
    n     = len(recs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.55, nrows * 3.20))
    axes_flat = np.array(axes).flatten()

    for idx, (r, ax) in enumerate(zip(recs, axes_flat)):
        err  = r['err']
        mu   = r['bias']
        sig  = float(np.std(err))
        lo   = float(np.percentile(err, 0.3))
        hi   = float(np.percentile(err, 99.7))
        bins = np.linspace(lo, hi, 70)
        color = _G0_COLORS.get(r['g0'], 'steelblue')

        ax.hist(err, bins=bins, density=True,
                histtype='stepfilled',
                color=color, alpha=0.35, linewidth=0)
        ax.hist(err, bins=bins, density=True,
                histtype='step',
                color=color, linewidth=0.8)

        x_fit = np.linspace(lo, hi, 400)
        ax.plot(x_fit, sp_norm.pdf(x_fit, mu, sig),
                color='0.15', lw=1.2,
                label=rf'$\mathcal{{N}}({mu:+.3f},\,{sig:.3f})$')

        ax.axvline(0.0, color='0.4', lw=0.7, ls='-', zorder=3)
        ax.axvline(mu,  color='0.4', lw=0.7, ls='--', zorder=3)

        ax.set_title(rf'$G_0 = {r["g0"]:.1f}$')
        ax.set_xlim(lo, hi)
        ax.legend()
        _panel(ax, idx)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    # Shared axis labels — avoids redundant per-panel text
    fig.supxlabel('Prediction residual: predicted \u2212 truth  (dex)',
                  y=0.01, fontsize=8)
    fig.supylabel('Probability density', x=0.01, fontsize=8)
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    _save_fig(fig, os.path.join(save_dir, 'fig3_error_dist.png'))


# ── Figure 4: Density-stratified performance ───────────────────────────────────

def fig4_stratified(recs: list[dict], save_dir: str) -> None:
    """
    Four panels examining how model accuracy depends on gas density.

    (a) R^2 computed on subsets of cells above a minimum density percentile.
    (b) Fraction of cells with |error| below each threshold, vs G0.
    (c) Mean absolute error as a function of true-density decile.
    (d) R^2 restricted to the top-N% highest-density cells, vs N.
    """
    fig = plt.figure(figsize=(7.0, 6.6))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.48, wspace=0.38)
    g0_arr = np.array([r['g0'] for r in recs])

    # ── (a) R2 vs min-density percentile ─────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for r in recs:
        c = _G0_COLORS[r['g0']]
        ax1.plot(r['strat_pcts'], r['strat_r2'], color=c, lw=1.0,
                 label=rf'$G_0={r["g0"]:.1f}$')
    for p_ref, ls_ref in ((90, '--'), (99, ':')):
        ax1.axvline(p_ref, color='0.55', ls=ls_ref, lw=0.7)
        ax1.text(p_ref + 0.4, 0.01, rf'$p_{{{p_ref}}}$',
                 color='0.45', fontsize=6, va='bottom',
                 transform=ax1.get_xaxis_transform())
    ax1.set_xlabel(r'Min-density percentile of subset')
    ax1.set_ylabel(r'$R^2$')
    ax1.set_title(r'$R^2$ restricted to $n_{\rm H_2} \geq p$')
    ax1.set_xlim(0, 100)
    ax1.legend(ncol=2, fontsize=6)
    _panel(ax1, 0)

    # ── (b) Error-threshold fraction vs G0  (log G0 axis) ────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    thr_defs = [
        (0.05, _THR_COLORS[0], 'o',  r'$|\epsilon|<0.05$ dex'),
        (0.10, _THR_COLORS[1], 's',  r'$|\epsilon|<0.10$ dex'),
        (0.20, _THR_COLORS[2], '^',  r'$|\epsilon|<0.20$ dex'),
        (0.50, _THR_COLORS[3], 'D',  r'$|\epsilon|<0.50$ dex'),
    ]
    for thr, col, mk, lbl in thr_defs:
        vals = np.array([r['frac'][thr] * 100 for r in recs])
        ax2.plot(g0_arr, vals, color=col, marker=mk,
                 ms=3.5, lw=1.0, label=lbl)
    ax2.set_xscale('log')
    ax2.set_xticks(g0_arr)
    ax2.set_xticklabels([str(g) for g in g0_arr], fontsize=6.5)
    ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax2.set_xlabel(r'UV field strength $G_0$')
    ax2.set_ylabel(r'Fraction of cells (\%)')
    ax2.set_title(r'Error-threshold fractions vs.\ $G_0$')
    ax2.legend(fontsize=6)
    _panel(ax2, 1)

    # ── (c) MAE per density decile ────────────────────────────────────────────
    ax3    = fig.add_subplot(gs[1, 0])
    decile = np.arange(1, 11)
    for r in recs:
        c = _G0_COLORS[r['g0']]
        ax3.plot(decile, r['bin_mae'], color=c,
                 marker='o', ms=3.0, lw=1.0,
                 label=rf'$G_0={r["g0"]:.1f}$')
    ax3.set_xlabel(r'True-density decile  (1 = lowest, 10 = highest)')
    ax3.set_ylabel(r'Mean $|\epsilon|$  (dex)')
    ax3.set_title(r'MAE per density decile')
    ax3.set_xticks(decile)
    ax3.legend(ncol=2, fontsize=6)
    _panel(ax3, 2)

    # ── (d) Tail R2 vs top-N% (log scale) ────────────────────────────────────
    ax4      = fig.add_subplot(gs[1, 1])
    tail_pct = [0.1, 0.5, 1, 5, 10, 25, 50]
    for r in recs:
        c    = _G0_COLORS[r['g0']]
        vals = []
        for tp in tail_pct:
            thr  = float(np.percentile(r['y_t'], 100 - tp))
            mask = r['y_t'] >= thr
            v = (float(r2_score(r['y_t'][mask], r['y_p'][mask]))
                 if mask.sum() >= 10 else np.nan)
            vals.append(v)
        ax4.plot(tail_pct, vals, color=c, marker='o',
                 ms=3.0, lw=1.0,
                 label=rf'$G_0={r["g0"]:.1f}$')
    ax4.axhline(0, color='0.5', lw=0.6, ls='--')
    ax4.set_xscale('log')
    ax4.set_xticks(tail_pct)
    ax4.set_xticklabels([rf'${p}\%$' for p in tail_pct], fontsize=6)
    ax4.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax4.set_xlabel(r'Top-$N$\% highest-density cells')
    ax4.set_ylabel(r'$R^2$')
    ax4.set_title(r'$R^2$ in the high-density tail')
    ax4.legend(ncol=2, fontsize=6)
    _panel(ax4, 3)

    _save_fig(fig, os.path.join(save_dir, 'fig4_stratified.png'))


# ── Figure 5: Spatial projections ─────────────────────────────────────────────

def fig5_spatial(recs: list[dict], save_dir: str) -> None:
    """
    Mean absolute error averaged along each axis, shown as a 2-D image
    in the x-y, x-z, and y-z planes.  All panels share a common colour
    scale set at the 99th percentile of the error across all cubes.
    Colourmap: 'inferno' (monotone, perceptually uniform, prints in grey).
    """
    n   = len(recs)
    fig = plt.figure(figsize=(10.5, n * 3.1))
    gs  = gridspec.GridSpec(n, 3, figure=fig,
                            hspace=0.50, wspace=0.38)

    all_vals = np.concatenate([r[k].flatten()
                               for r in recs for k in ('pxy', 'pxz', 'pyz')])
    vmax = float(np.percentile(all_vals, 99))

    proj_cfg = [
        ('pxy', r'$\langle|\epsilon|\rangle_{z}$  ($x$–$y$ plane)', ('$i_x$', '$i_y$')),
        ('pxz', r'$\langle|\epsilon|\rangle_{y}$  ($x$–$z$ plane)', ('$i_x$', '$i_z$')),
        ('pyz', r'$\langle|\epsilon|\rangle_{x}$  ($y$–$z$ plane)', ('$i_y$', '$i_z$')),
    ]

    # One shared colorbar at the figure level
    sm = plt.cm.ScalarMappable(cmap='inferno',
                               norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])

    for row, r in enumerate(recs):
        for col, (key, col_title, (xl, yl)) in enumerate(proj_cfg):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(r[key].T,
                      origin='lower', aspect='equal',
                      cmap='inferno', vmin=0.0, vmax=vmax,
                      extent=[0, 128, 0, 128],
                      rasterized=True)
            ax.set_xlabel(xl, fontsize=8.5)
            ax.set_ylabel(
                (rf'$G_0={r["g0"]:.1f}$' + '\n' + yl) if col == 0 else yl,
                fontsize=8.5)
            if row == 0:
                ax.set_title(col_title, fontsize=9)
            # Minimal ticks (0, 64, 128)
            ax.set_xticks([0, 64, 128])
            ax.set_yticks([0, 64, 128])
            ax.tick_params(labelsize=8)

    # Single horizontal colorbar below the grid
    cax = fig.add_axes([0.15, 0.01, 0.70, 0.015])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(r'Mean $|\epsilon|$  (dex)', fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle(r'Spatial distribution of mean absolute error',
                 fontsize=10, y=1.002)

    _save_fig(fig, os.path.join(save_dir, 'fig5_spatial.png'))


# ── Figure 6: Density distribution comparison ─────────────────────────────────

def fig6_distributions(recs: list[dict], save_dir: str) -> None:
    """
    Step histograms comparing the marginal distribution of
    log10(nH2) in the ground truth (solid blue) against the model
    predictions (dashed orange-red) for each G0 value.
    The shift in mean Delta mu and spread Delta sigma are annotated.
    """
    n     = len(recs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.55, nrows * 3.20))
    axes_flat = np.array(axes).flatten()

    for idx, (r, ax) in enumerate(zip(recs, axes_flat)):
        y_t, y_p = r['y_t'], r['y_p']
        lo   = float(min(y_t.min(), y_p.min()))
        hi   = float(max(y_t.max(), y_p.max()))
        bins = np.linspace(lo, hi, 75)

        ax.hist(y_t, bins=bins, density=True,
                histtype='stepfilled', color=_C_TRUTH, alpha=0.20, lw=0)
        ax.hist(y_t, bins=bins, density=True,
                histtype='step', color=_C_TRUTH, lw=1.0,
                label='Truth')

        ax.hist(y_p, bins=bins, density=True,
                histtype='stepfilled', color=_C_PRED, alpha=0.20, lw=0)
        ax.hist(y_p, bins=bins, density=True,
                histtype='step', color=_C_PRED, lw=1.0, ls='--',
                label='Prediction')

        # KS test and Wasserstein-1 distance — more appropriate than
        # mean/std for the bimodal distributions typical of these cubes
        d_ks, p_ks = ks_2samp(y_t, y_p)
        w1 = wasserstein_distance(y_t, y_p)

        ax.set_xlabel(r'$\log_{10}(n_{\rm H_2})$  (dex)')
        ax.set_ylabel(r'$p(\log_{10}\,n_{\rm H_2})$')
        ax.set_title(rf'$G_0={r["g0"]:.1f}$')
        ax.text(0.97, 0.96,
                (rf'KS $D={d_ks:.3f}$' + '\n'
                 + rf'$p={p_ks:.1e}$'   + '\n'
                 + rf'$W_1={w1:.3f}$ dex'),
                transform=ax.transAxes, fontsize=6.5,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.7', linewidth=0.5, alpha=0.92))
        ax.legend(fontsize=6.5, framealpha=0.9)
        _panel(ax, idx)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig6_distributions.png'))


# ── Figure 7: Mass budget and phase-conditional accuracy ──────────────────────

def fig7_massbudget(recs: list[dict], save_dir: str) -> None:
    """
    (a) Total H2 mass ratio (predicted/true) per G0, raw stacked prediction
        vs recalibrated, log y-axis with the ideal ratio 1 marked.
    (b) Mean residual (bias, dex) per G0, raw vs recalibrated.
    (c) Phase-conditional R2 (molecular / diffuse split at PHASE_SPLIT) and
        overall R2 per G0.
    Raw-prediction series are drawn only for .npz files that contain
    pred_vol_raw (predictions saved after the 2026-07 recalibration fix).
    """
    g0s     = np.array([r['g0'] for r in recs])
    has_raw = any(r.get('raw') is not None for r in recs)
    # Without a raw volume (pre-2026-07 prediction files) the delivered
    # pred_vol may itself be uncalibrated — label it neutrally.
    lbl_cal = 'recalibrated' if has_raw else 'delivered prediction'

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.3))

    def _logx(ax):
        ax.set_xscale('log')
        ax.set_xticks(g0s)
        ax.set_xticklabels([f'{g:g}' for g in g0s], fontsize=6.5)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlabel(r'UV field strength $G_0$')

    # (a) mass ratio
    ax = axes[0]
    if has_raw:
        raw_mr = [r['raw']['mass_ratio'] if r.get('raw') else np.nan for r in recs]
        ax.plot(g0s, raw_mr, color='0.55', marker='o', ls='--',
                label='raw stacked')
    ax.plot(g0s, [r['mass_ratio'] for r in recs], color=_C_PRED,
            marker='D', ls='-', label=lbl_cal)
    ax.axhline(1.0, color='0.3', lw=0.7, ls=':')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    ax.set_ylabel(r'$M_{\rm H_2}^{\rm pred} / M_{\rm H_2}^{\rm true}$')
    ax.set_title('Total H$_2$ mass budget')
    ax.legend(fontsize=6.5)
    _logx(ax)
    _panel(ax, 0)

    # (b) bias
    ax = axes[1]
    if has_raw:
        raw_b = [r['raw']['bias'] if r.get('raw') else np.nan for r in recs]
        ax.plot(g0s, raw_b, color='0.55', marker='o', ls='--',
                label='raw stacked')
    ax.plot(g0s, [r['bias'] for r in recs], color=_C_PRED,
            marker='D', ls='-', label=lbl_cal)
    ax.axhline(0.0, color='0.3', lw=0.7, ls=':')
    ax.set_ylabel(r'Mean residual $\langle\hat{y}-y\rangle$  (dex)')
    ax.set_title('Log-space bias')
    ax.legend(fontsize=6.5)
    _logx(ax)
    _panel(ax, 1)

    # (c) phase-conditional R2
    ax = axes[2]
    ax.plot(g0s, [r['r2'] for r in recs], color='0.25', marker='o', ls='-',
            label='all cells')
    ax.plot(g0s, [r['r2_mol'] for r in recs], color=_C_TRUTH, marker='s',
            ls='--', label=rf'molecular ($y > {PHASE_SPLIT:g}$)')
    ax.plot(g0s, [r['r2_dif'] for r in recs], color=_C_PRED, marker='^',
            ls='--', label=rf'diffuse ($y \leq {PHASE_SPLIT:g}$)')
    ax.set_ylabel(r'$R^2$')
    ax.set_title('Phase-conditional accuracy')
    ax.legend(fontsize=6.5, loc='lower left')
    _logx(ax)
    _panel(ax, 2)

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig7_massbudget.png'))


# ── Figure 8: Mid-plane slices ─────────────────────────────────────────────────

def fig8_slices(recs: list[dict], save_dir: str) -> None:
    """
    Mid-plane (z = N/2) slices for each G0 (rows): ground truth, prediction,
    and absolute error (columns).  Truth and prediction share one colour
    scale per row; the error column has its own scale per row.
    """
    n   = len(recs)
    fig = plt.figure(figsize=(9.6, n * 2.9))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.32, wspace=0.30)

    for row, r in enumerate(recs):
        s_t, s_p = r['slice_t'], r['slice_p']
        s_e      = np.abs(s_p - s_t)
        vmin = float(min(s_t.min(), s_p.min()))
        vmax = float(max(s_t.max(), s_p.max()))
        emax = float(np.percentile(s_e, 99))

        panels = [
            (s_t, 'viridis', vmin, vmax, 'Ground truth'),
            (s_p, 'viridis', vmin, vmax, 'Prediction'),
            (s_e, 'inferno', 0.0,  emax, r'$|\epsilon|$'),
        ]
        for col, (img, cmap, lo, hi, col_title) in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(img.T, origin='lower', aspect='equal',
                           cmap=cmap, vmin=lo, vmax=hi,
                           extent=[0, img.shape[0], 0, img.shape[1]],
                           rasterized=True)
            if row == 0:
                ax.set_title(col_title, fontsize=9)
            ax.set_xticks([0, img.shape[0] // 2, img.shape[0]])
            ax.set_yticks([0, img.shape[1] // 2, img.shape[1]])
            ax.tick_params(labelsize=6.5)
            ax.set_xlabel('$i_x$', fontsize=7.5)
            ax.set_ylabel(
                (rf'$G_0={r["g0"]:.1f}$' + '\n' + '$i_y$') if col == 0 else '$i_y$',
                fontsize=7.5)
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6)
            if col < 2:
                cb.set_label(r'$\log_{10}(n_{\rm H_2})$', fontsize=6.5)
            else:
                cb.set_label(r'$|\epsilon|$ (dex)', fontsize=6.5)

    _save_fig(fig, os.path.join(save_dir, 'fig8_slices.png'))


# ── Text summary table ─────────────────────────────────────────────────────────

def _print_table(recs: list[dict]) -> None:
    HDR = (f"{'G0':>5} | {'R2':>8} | {'R2_lin':>8} | {'RMSE':>8} | "
           f"{'MAE':>8} | {'Bias':>9} | {'<0.1dex%':>9} | {'corr':>8}")
    SEP = '-' * len(HDR)
    print('\n' + SEP)
    print(HDR)
    print(SEP)
    for r in recs:
        print(f"{r['g0']:>5.1f} | {r['r2']:>8.4f} | {r['r2_lin']:>8.4f} | "
              f"{r['rmse']:>8.4f} | {r['mae']:>8.4f} | {r['bias']:>+9.4f} | "
              f"{r['frac'][0.1]*100:>8.1f}% | {r['corr']:>8.4f}")
    keys  = ('r2', 'r2_lin', 'rmse', 'mae', 'bias', 'corr')
    means = {k: np.mean([r[k] for r in recs]) for k in keys}
    print(SEP)
    print(f"{'MEAN':>5} | {means['r2']:>8.4f} | {means['r2_lin']:>8.4f} | "
          f"{means['rmse']:>8.4f} | {means['mae']:>8.4f} | {means['bias']:>+9.4f} | "
          f"{np.mean([r['frac'][0.1] for r in recs])*100:>8.1f}% | "
          f"{means['corr']:>8.4f}")
    print(SEP + '\n')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Statistical analysis: ground-truth vs stacked_sp predictions.')
    parser.add_argument('--pred-dir', default=None,
                        help='Directory with .npz prediction files '
                             '(default: opens a folder-chooser dialog)')
    parser.add_argument('--save-dir', default='analysis_output',
                        help='Output directory for figures (default: analysis_output/)')
    args = parser.parse_args()

    pred_dir  = args.pred_dir or _choose_directory()
    npz_files = _find_npz(pred_dir)
    if not npz_files:
        print(f'No .npz files found in: {pred_dir}')
        sys.exit(1)

    print(f'Found {len(npz_files)} prediction file(s) in {pred_dir}')
    os.makedirs(args.save_dir, exist_ok=True)

    recs: list[dict] = []
    for npz in npz_files:
        pred_vol, g0, r2_xgb, r2_mlp, r2_stacked, kernels, epochs = \
            load_prediction(npz)
        # Raw (pre-recalibration) volume, present in predictions saved after
        # the 2026-07 mass-budget fix
        with np.load(npz) as d:
            pred_vol_raw = (d['pred_vol_raw'] if 'pred_vol_raw' in d.files
                            else None)
        print(f'\n  G0={g0:.1f}  loading ground truth ...')
        cube_df   = load_single_cube(g0)
        y_true    = cube_df['log_nH2'].values.astype(np.float32)
        truth_vol = _preds_to_volume(cube_df, y_true)

        print(f'  G0={g0:.1f}  computing statistics ...')
        rec = _compute_record(truth_vol, pred_vol)
        rec.update(g0=g0, r2_xgb=r2_xgb, r2_mlp=r2_mlp, r2_stacked=r2_stacked)
        if pred_vol_raw is not None and not np.allclose(pred_vol_raw, pred_vol):
            y_t = truth_vol.flatten().astype(np.float64)
            y_r = pred_vol_raw.flatten().astype(np.float64)
            lo, hi = float(y_t.min()) - 1.0, float(y_t.max()) + 1.0
            rec['raw'] = {
                'bias': float(np.mean(y_r - y_t)),
                'mass_ratio': float(np.sum(10.0 ** np.clip(y_r, lo, hi))
                                    / np.sum(10.0 ** y_t)),
                'r2': float(r2_score(y_t, y_r)),
            }
        else:
            rec['raw'] = None
        recs.append(rec)
        print(f'  R2={rec["r2"]:.4f}  MAE={rec["mae"]:.4f}  '
              f'bias={rec["bias"]:+.4f}  massR={rec["mass_ratio"]:.3f}  '
              f'<0.1dex={rec["frac"][0.1]*100:.1f}%')

    recs.sort(key=lambda r: r['g0'])
    _print_table(recs)

    print(f'Generating figures in {args.save_dir}/ ...')
    fig1_summary(recs, args.save_dir)
    fig2_scatter(recs, args.save_dir)
    fig3_error_dist(recs, args.save_dir)
    fig4_stratified(recs, args.save_dir)
    fig5_spatial(recs, args.save_dir)
    fig6_distributions(recs, args.save_dir)
    fig7_massbudget(recs, args.save_dir)
    fig8_slices(recs, args.save_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
