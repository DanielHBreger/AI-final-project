#!/usr/bin/env python3
"""
statistical_analysis.py
========================
Load all stacked_sp prediction .npz files from a directory, compare each
against its ground-truth cube, compute statistics, and save six PNG figures.

Figures produced
----------------
  fig1_summary.png         -- R2, R2_lin, MAE, RMSE, Bias, fraction-within bars
  fig2_scatter.png         -- hexbin truth-vs-prediction scatter per G0
  fig3_error_dist.png      -- signed-error histograms with Gaussian fits per G0
  fig4_stratified.png      -- density-stratified R2 curves + error-fraction bars
  fig5_spatial.png         -- 2-D mean|error| projections (xy / xz / yz) per G0
  fig6_distributions.png   -- truth vs prediction density histograms per G0

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
matplotlib.use('Agg')                   # non-interactive, file-only backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm as sp_norm
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from predict_and_visualize import _preds_to_volume
from data_loader import load_single_cube
from viz_common import load_prediction


# ── Global style ───────────────────────────────────────────────────────────────

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor':   '#f7f7f7',
    'axes.edgecolor':   '#cccccc',
    'axes.grid':        True,
    'grid.color':       'white',
    'grid.linewidth':   0.9,
    'font.size':        10,
    'axes.titlesize':   11,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8.5,
})

_G0_LIST   = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
_G0_COLORS = {g: c for g, c in
              zip(_G0_LIST, plt.cm.plasma(np.linspace(0.1, 0.9, len(_G0_LIST))))}


# ── I/O helpers ────────────────────────────────────────────────────────────────

def _find_npz(directory: str) -> list[str]:
    """Return sorted list of .npz files in *directory*."""
    return sorted(glob.glob(os.path.join(directory, '*.npz')))


def _choose_directory() -> str:
    """Prompt the user to choose a directory; fall back to cwd on failure."""
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
        print('  Using current directory.  Pass --pred-dir to specify one.')
        return '.'


def _save_fig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── Statistics ─────────────────────────────────────────────────────────────────

def _compute_record(truth_vol: np.ndarray,
                    pred_vol:  np.ndarray) -> dict:
    """Return a dict of all statistics needed for plotting."""
    y_t = truth_vol.flatten().astype(np.float64)
    y_p = pred_vol.flatten().astype(np.float64)
    err = y_p - y_t

    r2   = float(r2_score(y_t, y_p))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    # linear-space R2 (clipped +-1 dex around truth range)
    lo     = float(y_t.min()) - 1.0
    hi     = float(y_t.max()) + 1.0
    r2_lin = float(r2_score(10 ** y_t, 10 ** np.clip(y_p, lo, hi)))

    corr = float(np.corrcoef(y_t, y_p)[0, 1])

    # fraction within absolute-error thresholds
    frac = {t: float(np.mean(np.abs(err) < t)) for t in (0.05, 0.1, 0.2, 0.5)}

    # density-stratified R2 (R2 on cells above percentile p)
    strat_pcts  = np.linspace(0, 99, 200)
    strat_r2    = _density_stratified_r2(y_t, y_p, strat_pcts)

    # 2-D spatial projections of mean|error| (shape 128x128 each)
    e_vol = np.abs(pred_vol - truth_vol).astype(np.float32)
    pxy   = e_vol.mean(axis=2)    # mean over z  -> x-y plane
    pxz   = e_vol.mean(axis=1)    # mean over y  -> x-z plane
    pyz   = e_vol.mean(axis=0)    # mean over x  -> y-z plane

    # error binned by true-density decile
    n_bins   = 10
    bin_edges = np.percentile(y_t, np.linspace(0, 100, n_bins + 1))
    bin_mae   = np.zeros(n_bins, dtype=np.float32)
    bin_label = []
    for i in range(n_bins):
        mask = (y_t >= bin_edges[i]) & (y_t < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_mae[i] = float(np.mean(np.abs(err[mask])))
        bin_label.append(f'{bin_edges[i]:.1f}')

    return dict(
        r2=r2, rmse=rmse, mae=mae, bias=bias, r2_lin=r2_lin, corr=corr,
        n_cells=len(y_t),
        truth_mean=float(y_t.mean()), truth_std=float(y_t.std()),
        pred_mean=float(y_p.mean()),  pred_std=float(y_p.std()),
        frac=frac,
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
    """R2 computed on cells whose true density is >= the p-th percentile."""
    out = np.full(len(pcts), np.nan, dtype=np.float32)
    for i, p in enumerate(pcts):
        thr  = float(np.percentile(y_t, p))
        mask = y_t >= thr
        if mask.sum() >= 10:
            out[i] = float(r2_score(y_t[mask], y_p[mask]))
    return out


# ── Figure 1: Metric summary ───────────────────────────────────────────────────

def fig1_summary(recs: list[dict], save_dir: str) -> None:
    """Six bar charts: R2, R2_lin, MAE, RMSE, Bias, fraction<0.1dex per G0."""
    labels = [f"G0={r['g0']:.1f}" for r in recs]
    colors = [_G0_COLORS.get(r['g0'], 'steelblue') for r in recs]
    x = np.arange(len(recs))

    metric_defs = [
        ('r2',    r'$R^2$ (log-space)',             r'$R^2$', None,  '.4f'),
        ('r2_lin',r'$R^2$ linear nH$_2$ (clipped)', r'$R^2$', None,  '.4f'),
        ('mae',   'MAE (dex)',                       'dex',    None,  '.4f'),
        ('rmse',  'RMSE (dex)',                      'dex',    None,  '.4f'),
        ('bias',  'Bias: mean(pred - truth)',         'dex',    0.0,  '+.4f'),
    ]
    frac_01 = [r['frac'][0.1] * 100 for r in recs]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('stacked_sp  Prediction Quality Summary', fontsize=14,
                 fontweight='bold')
    axes_flat = axes.flatten()

    for ax, (key, title, ylabel, ref, fmt) in zip(axes_flat, metric_defs):
        vals = [r[key] for r in recs]
        bars = ax.bar(x, vals, color=colors, edgecolor='k', linewidth=0.4)
        if ref is not None:
            ax.axhline(ref, color='k', ls='--', lw=0.8, alpha=0.45)
        mu = np.mean(vals)
        ax.axhline(mu, color='navy', ls=':', lw=1.1, alpha=0.85,
                   label=f'mean = {mu:{fmt}}')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha='right')
        for bar, v in zip(bars, vals):
            dy = abs(v) * 0.025 + 0.003
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + dy,
                    f'{v:{fmt}}', ha='center', va='bottom', fontsize=7.5)
        ax.legend()

    # 6th panel: fraction within 0.1 dex
    ax = axes_flat[5]
    bars = ax.bar(x, frac_01, color=colors, edgecolor='k', linewidth=0.4)
    mu_f = np.mean(frac_01)
    ax.axhline(mu_f, color='navy', ls=':', lw=1.1, alpha=0.85,
               label=f'mean = {mu_f:.1f}%')
    ax.set_title('Fraction of cells with |error| < 0.1 dex')
    ax.set_ylabel('%')
    ax.set_ylim(0, 107)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha='right')
    for bar, v in zip(bars, frac_01):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f'{v:.1f}%',
                ha='center', va='bottom', fontsize=7.5)
    ax.legend()

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig1_summary.png'))


# ── Figure 2: Scatter ──────────────────────────────────────────────────────────

def fig2_scatter(recs: list[dict], save_dir: str) -> None:
    """Hexbin truth-vs-prediction scatter for each G0, with y=x reference."""
    n     = len(recs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 4.2))
    axes_flat = np.array(axes).flatten()
    fig.suptitle(r'Truth vs Prediction  (log$_{10}$ nH$_2$, dex)',
                 fontsize=13, fontweight='bold')

    for r, ax in zip(recs, axes_flat):
        y_t, y_p = r['y_t'], r['y_p']
        lo = float(min(y_t.min(), y_p.min())) - 0.15
        hi = float(max(y_t.max(), y_p.max())) + 0.15
        hb = ax.hexbin(y_t, y_p, gridsize=80, bins='log',
                       cmap='plasma', mincnt=1,
                       extent=[lo, hi, lo, hi])
        ax.plot([lo, hi], [lo, hi], 'w--', lw=1.3)
        ax.set_xlabel('Truth (dex)')
        ax.set_ylabel('Prediction (dex)')
        ax.set_title(f"G0={r['g0']:.1f}   "
                     r"$R^2$" f"={r['r2']:.4f}   r={r['corr']:.4f}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        fig.colorbar(hb, ax=ax, label=r'count (log$_{10}$)')

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig2_scatter.png'))


# ── Figure 3: Error distributions ─────────────────────────────────────────────

def fig3_error_dist(recs: list[dict], save_dir: str) -> None:
    """Signed-error histograms (pred - truth) with Gaussian overlay."""
    n     = len(recs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.4))
    axes_flat = np.array(axes).flatten()
    fig.suptitle('Error Distribution: prediction - truth  (dex)',
                 fontsize=13, fontweight='bold')

    for r, ax in zip(recs, axes_flat):
        err  = r['err']
        mu   = r['bias']
        sig  = float(np.std(err))
        lo   = float(np.percentile(err, 0.5))
        hi   = float(np.percentile(err, 99.5))
        bins = np.linspace(lo, hi, 80)
        color = _G0_COLORS.get(r['g0'], 'steelblue')
        ax.hist(err, bins=bins, density=True,
                color=color, alpha=0.75, edgecolor='none')
        x_fit = np.linspace(lo, hi, 300)
        ax.plot(x_fit, sp_norm.pdf(x_fit, mu, sig), 'k-', lw=1.7,
                label=f'N({mu:+.3f}, {sig:.3f})')
        ax.axvline(mu,  color='k',   ls='--', lw=1.0, alpha=0.8)
        ax.axvline(0.0, color='red', ls=':',  lw=1.0, alpha=0.6)
        ax.set_xlabel('Error (dex)')
        ax.set_ylabel('Probability density')
        ax.set_title(f"G0={r['g0']:.1f}   "
                     f"bias={mu:+.4f}   sigma={sig:.4f}")
        ax.legend()

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig3_error_dist.png'))


# ── Figure 4: Density-stratified performance ───────────────────────────────────

def fig4_stratified(recs: list[dict], save_dir: str) -> None:
    """R2 vs density threshold; error fraction bars; MAE per density decile."""
    fig = plt.figure(figsize=(17, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    fig.suptitle('Density-Stratified Performance', fontsize=13, fontweight='bold')

    # ── (0,0) R2 vs min-density percentile ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for r in recs:
        c = _G0_COLORS.get(r['g0'], 'steelblue')
        ax1.plot(r['strat_pcts'], r['strat_r2'], color=c, lw=1.7,
                 label=f"G0={r['g0']:.1f}")
    ax1.axvline(90, color='grey', ls='--', lw=0.9, alpha=0.6)
    ax1.axvline(99, color='grey', ls=':',  lw=0.9, alpha=0.6)
    ax1.text(90.5, ax1.get_ylim()[0] if ax1.get_ylim()[0] > -5 else -2,
             '90th', fontsize=7.5, color='grey', va='bottom')
    ax1.text(99.5, ax1.get_ylim()[0] if ax1.get_ylim()[0] > -5 else -2,
             '99th', fontsize=7.5, color='grey', va='bottom')
    ax1.set_xlabel('Minimum-density percentile of subset')
    ax1.set_ylabel(r'$R^2$ (on subset)')
    ax1.set_title(r'$R^2$ restricted to cells above density threshold')
    ax1.set_xlim(0, 100)
    ax1.legend(ncol=2)

    # ── (0,1) Fraction within error thresholds ────────────────────────────────
    ax2   = fig.add_subplot(gs[0, 1])
    thrs  = [0.05, 0.1, 0.2, 0.5]
    g0s   = [r['g0'] for r in recs]
    xpos  = np.arange(len(g0s))
    width = 0.2
    pal   = plt.cm.Set2(np.linspace(0, 0.8, len(thrs)))
    for i, thr in enumerate(thrs):
        vals = [r['frac'][thr] * 100 for r in recs]
        ax2.bar(xpos + i * width, vals, width=width,
                color=pal[i], label=f'< {thr} dex', alpha=0.88, edgecolor='k',
                linewidth=0.3)
    ax2.set_xticks(xpos + width * (len(thrs) - 1) / 2)
    ax2.set_xticklabels([f'G0={g:.1f}' for g in g0s], rotation=28, ha='right')
    ax2.set_ylabel('% of cells')
    ax2.set_title('Fraction of cells within error thresholds')
    ax2.legend(ncol=2)
    ax2.set_ylim(0, 108)

    # ── (1,0) MAE per density decile (all G0 as lines) ────────────────────────
    ax3    = fig.add_subplot(gs[1, 0])
    decile = np.arange(1, 11)
    for r in recs:
        c = _G0_COLORS.get(r['g0'], 'steelblue')
        ax3.plot(decile, r['bin_mae'], marker='o', markersize=4, lw=1.7,
                 color=c, label=f"G0={r['g0']:.1f}")
    ax3.set_xlabel('True-density decile  (1 = lowest 10%, 10 = highest 10%)')
    ax3.set_ylabel('Mean |error| (dex)')
    ax3.set_title('MAE per density decile')
    ax3.set_xticks(decile)
    ax3.legend(ncol=2)

    # ── (1,1) Tail R2: R2 on top-N% cells ────────────────────────────────────
    ax4      = fig.add_subplot(gs[1, 1])
    tail_pct = [50, 25, 10, 5, 1, 0.5, 0.1]   # top-N% of cells
    x_bar    = np.arange(len(tail_pct))
    bar_w    = 0.8 / len(recs)
    for j, r in enumerate(recs):
        c    = _G0_COLORS.get(r['g0'], 'steelblue')
        vals = []
        for tp in tail_pct:
            thr  = float(np.percentile(r['y_t'], 100 - tp))
            mask = r['y_t'] >= thr
            v    = float(r2_score(r['y_t'][mask], r['y_p'][mask])) if mask.sum() >= 10 else np.nan
            vals.append(v)
        ax4.bar(x_bar + j * bar_w, vals, bar_w * 0.9,
                color=c, label=f"G0={r['g0']:.1f}", alpha=0.88, edgecolor='k',
                linewidth=0.25)
    ax4.axhline(0, color='k', ls='--', lw=0.7, alpha=0.4)
    ax4.set_xticks(x_bar + bar_w * (len(recs) - 1) / 2)
    ax4.set_xticklabels([f'top {p}%' for p in tail_pct])
    ax4.set_ylabel(r'$R^2$')
    ax4.set_title(r'$R^2$ on top-N% highest-density cells')
    ax4.legend(ncol=2)

    _save_fig(fig, os.path.join(save_dir, 'fig4_stratified.png'))


# ── Figure 5: Spatial projections ─────────────────────────────────────────────

def fig5_spatial(recs: list[dict], save_dir: str) -> None:
    """2-D mean|error| projections (xy, xz, yz planes) for every G0."""
    n   = len(recs)
    fig = plt.figure(figsize=(12, n * 2.7))
    gs  = gridspec.GridSpec(n, 3, figure=fig, hspace=0.50, wspace=0.38)
    fig.suptitle('Spatial Error Structure: mean|error| projected (dex)',
                 fontsize=12, fontweight='bold', y=1.003)

    # Shared colour scale (clipped at 99th percentile across all cubes)
    all_vals = np.concatenate([r[k].flatten()
                               for r in recs for k in ('pxy', 'pxz', 'pyz')])
    vmax = float(np.percentile(all_vals, 99))

    proj_cfg = [
        ('pxy', 'Mean over z  (x-y plane)', ('x', 'y')),
        ('pxz', 'Mean over y  (x-z plane)', ('x', 'z')),
        ('pyz', 'Mean over x  (y-z plane)', ('y', 'z')),
    ]

    for row, r in enumerate(recs):
        for col, (key, lbl, (xl, yl)) in enumerate(proj_cfg):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(r[key].T, origin='lower', aspect='equal',
                           cmap='hot', vmin=0.0, vmax=vmax,
                           extent=[0, 128, 0, 128])
            ax.set_xlabel(xl, fontsize=8)
            if col == 0:
                ax.set_ylabel(f"G0={r['g0']:.1f}\n{yl}", fontsize=8)
            else:
                ax.set_ylabel(yl, fontsize=8)
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            cb = plt.colorbar(im, ax=ax, fraction=0.044, pad=0.04)
            cb.ax.tick_params(labelsize=7)

    _save_fig(fig, os.path.join(save_dir, 'fig5_spatial.png'))


# ── Figure 6: Density distribution comparison ─────────────────────────────────

def fig6_distributions(recs: list[dict], save_dir: str) -> None:
    """Overlay truth and prediction log10(nH2) histograms per G0."""
    n     = len(recs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.4))
    axes_flat = np.array(axes).flatten()
    fig.suptitle(r'Density Distribution: Truth vs Prediction  (log$_{10}$ nH$_2$)',
                 fontsize=13, fontweight='bold')

    for r, ax in zip(recs, axes_flat):
        y_t, y_p = r['y_t'], r['y_p']
        lo   = float(min(y_t.min(), y_p.min()))
        hi   = float(max(y_t.max(), y_p.max()))
        bins = np.linspace(lo, hi, 80)
        ax.hist(y_t, bins=bins, density=True, alpha=0.62,
                color='steelblue', edgecolor='none',
                label=f"Truth  mu={r['truth_mean']:.2f}  sigma={r['truth_std']:.2f}")
        ax.hist(y_p, bins=bins, density=True, alpha=0.62,
                color='tomato', edgecolor='none',
                label=f"Pred   mu={r['pred_mean']:.2f}  sigma={r['pred_std']:.2f}")
        ax.set_xlabel(r'log$_{10}$(nH$_2$) [dex]')
        ax.set_ylabel('Probability density')
        dm = r['pred_mean'] - r['truth_mean']
        ds = r['pred_std']  - r['truth_std']
        ax.set_title(f"G0={r['g0']:.1f}   delta_mu={dm:+.3f}   delta_sigma={ds:+.3f}")
        ax.legend(fontsize=7.5)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    _save_fig(fig, os.path.join(save_dir, 'fig6_distributions.png'))


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
        print(f'\n  G0={g0:.1f}  loading ground truth ...')
        cube_df   = load_single_cube(g0)
        y_true    = cube_df['log_nH2'].values.astype(np.float32)
        truth_vol = _preds_to_volume(cube_df, y_true)

        print(f'  G0={g0:.1f}  computing statistics ...')
        rec = _compute_record(truth_vol, pred_vol)
        rec.update(g0=g0, r2_xgb=r2_xgb, r2_mlp=r2_mlp, r2_stacked=r2_stacked)
        recs.append(rec)
        print(f'  R2={rec["r2"]:.4f}  MAE={rec["mae"]:.4f}  '
              f'bias={rec["bias"]:+.4f}  <0.1dex={rec["frac"][0.1]*100:.1f}%')

    recs.sort(key=lambda r: r['g0'])
    _print_table(recs)

    print(f'Generating figures in {args.save_dir}/ ...')
    fig1_summary(recs, args.save_dir)
    fig2_scatter(recs, args.save_dir)
    fig3_error_dist(recs, args.save_dir)
    fig4_stratified(recs, args.save_dir)
    fig5_spatial(recs, args.save_dir)
    fig6_distributions(recs, args.save_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
