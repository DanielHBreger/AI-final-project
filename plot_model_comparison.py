#!/usr/bin/env python3
"""
plot_model_comparison.py
========================
Per-fold model comparison figure from an arch_comparison_*.json log:
the paper's core result (Table 1) as a picture, showing at a glance where
each model family degrades across the G0 sweep (extrapolation boundaries),
what the spatial features buy, and what stacking buys.

Panels
------
  (a) log-space R2 per held-out fold vs G0
  (b) RMSE (dex) per held-out fold vs G0
  (c) skill vs the pointwise XGBoost baseline (1 - MSE/MSE_ref) per fold
      -- only drawn when the log contains 'skill_vs_xgb' (runs made after
      the 2026-07 metrics overhaul); otherwise falls back to bias (dex)
      if available, or is omitted.

Usage
-----
  python plot_model_comparison.py                          # latest results/arch_comparison_*.json
  python plot_model_comparison.py --log results/arch_comparison_20260311_125636.json
  python plot_model_comparison.py --variants xgb_standard mlp_wide stacked_sp
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from viz_common import apply_journal_style

apply_journal_style()

# Display names and stable colors/markers per variant (colorblind-safe)
VARIANT_STYLE = {
    'xgb_standard':           ('XGBoost',                '#1f77b4', 'o', '--'),
    'mlp_wide':               ('Wide MLP',               '#ff7f0e', 's', '--'),
    'xgb_standard_sp':        ('XGBoost + spatial',      '#1f77b4', 'o', '-'),
    'mlp_wide_sp':            ('MLP + spatial',          '#ff7f0e', 's', '-'),
    'ens_sp':                 ('Equal-weight ensemble',  '#2ca02c', '^', '-'),
    'stacked_sp':             ('Stacked ensemble',       '#d62728', 'D', '-'),
    'stacked_sp_cal':         ('Stacked (recalibrated)', '#9467bd', 'v', '-'),
    'stacked_weighted':       ('Weighted stack',         '#d62728', 'D', '--'),
    'stacked_weighted_mwcal': ('Weighted stack (mass-recal.)',
                                                         '#9467bd', 'v', '-'),
    'unet_standard':          ('3D U-Net (32 ch.)',      '#8c564b', 'P', ':'),
    'unet_baseline':          ('3D U-Net (32 ch.)',      '#8c564b', 'P', ':'),
    'unet_large':             ('3D U-Net (64 ch.)',      '#8c564b', 'X', ':'),
    'xgb_standard_sp_w':      ('XGBoost + sp. (wght.)',  '#17becf', 'o', ':'),
    'mlp_wide_sp_w':          ('MLP + sp. (wght.)',      '#bcbd22', 's', ':'),
}
DEFAULT_VARIANTS = ['xgb_standard', 'mlp_wide', 'xgb_standard_sp',
                    'mlp_wide_sp', 'ens_sp', 'stacked_sp', 'unet_standard',
                    'unet_baseline']


def _latest_log() -> str | None:
    logs = sorted(glob.glob('results/arch_comparison_*.json'))
    return logs[-1] if logs else None


def _merge_cnn_log(log: dict, cnn_log_path: str) -> None:
    """Merge U-Net variants from a cnn_test_*.json into the arch log dict.

    test_cnn.py writes CNN results to a separate JSON with the same
    folds/metrics structure but no 'skill_vs_xgb' (skill is a cross-model
    quantity added at comparison level).  Because RMSE is permutation-
    invariant and both logs evaluate the same native-grid cells, the skill
    can be computed here from RMSE ratios against the arch log's
    xgb_standard row — but only when the grid sizes match (a 64³ CNN sees
    a pooled target field and is not comparable to the native reference).
    """
    with open(cnn_log_path) as f:
        cnn = json.load(f)
    if cnn.get('g0_values') != log['g0_values']:
        print(f'  (cnn log skipped: g0_values differ from {cnn_log_path})')
        return
    ref = log['variants'].get('xgb_standard')
    same_grid = cnn.get('grid_size', 128) == 128
    if not same_grid:
        print('  (cnn log is 64^3-pooled: skill vs the native XGBoost '
              'reference not computed)')
    for name, var in cnn['variants'].items():
        merged = {'folds': [
            {'fold': f['fold'], 'g0': f['g0'], 'metrics': dict(f['metrics'])}
            for f in var['folds']
        ]}
        if ref is not None and same_grid:
            for fm, fr in zip(merged['folds'], ref['folds']):
                mse_ref = fr['metrics']['RMSE'] ** 2
                fm['metrics']['skill_vs_xgb'] = \
                    1.0 - fm['metrics']['RMSE'] ** 2 / mse_ref
        log['variants'][name] = merged
        print(f'  Merged {name} from {cnn_log_path} '
              f'(grid={cnn.get("grid_size", "?")}^3)')


def _fold_values(variant: dict, key: str) -> list[float]:
    return [f['metrics'].get(key, np.nan) for f in variant['folds']]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Per-fold model comparison figure from an arch_comparison JSON.')
    parser.add_argument('--log', default=None,
                        help='arch_comparison JSON (default: latest in results/)')
    parser.add_argument('--cnn-log', default=None,
                        help='Optional cnn_test_*.json whose U-Net variants '
                             'are merged into the comparison (skill vs the '
                             'arch log\'s xgb_standard is computed from RMSE '
                             'when both are native 128^3)')
    parser.add_argument('--variants', nargs='+', default=None,
                        help=f'Variants to plot (default: those of '
                             f'{DEFAULT_VARIANTS} present in the log)')
    parser.add_argument('--save', default='figures/fig_model_comparison.png',
                        help='Output PNG path')
    parser.add_argument('--r2-min', type=float, default=None,
                        help='Lower y-limit for the R2 panel (e.g. 0.85), for '
                             'when one poor variant compresses the axis')
    args = parser.parse_args()

    log_path = args.log or _latest_log()
    if log_path is None or not os.path.exists(log_path):
        print('No results/arch_comparison_*.json found; pass one with --log', file=sys.stderr)
        sys.exit(1)
    with open(log_path) as f:
        log = json.load(f)
    print(f'Log: {log_path}')

    if args.cnn_log:
        _merge_cnn_log(log, args.cnn_log)

    g0 = np.array(log['g0_values'], dtype=float)
    wanted = args.variants or DEFAULT_VARIANTS
    variants = {n: log['variants'][n] for n in wanted if n in log['variants']}
    missing = [n for n in wanted if n not in log['variants']]
    if missing:
        print(f'  (not in log, skipped: {missing})')
    if not variants:
        print('None of the requested variants are in the log.', file=sys.stderr)
        sys.exit(1)

    # Third panel: skill if recorded, else bias, else nothing
    has_skill = any(not np.all(np.isnan(_fold_values(v, 'skill_vs_xgb')))
                    for v in variants.values())
    has_bias  = any(not np.all(np.isnan(_fold_values(v, 'bias')))
                    for v in variants.values())
    n_panels = 3 if (has_skill or has_bias) else 2

    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 3.3))
    axes = np.atleast_1d(axes)

    panel_defs = [('R2', r'$R^2$ (log space)', r'$R^2$')]
    panel_defs.append(('RMSE', r'RMSE (dex)', r'RMSE (dex)'))
    if has_skill:
        panel_defs.append(('skill_vs_xgb',
                           r'Skill vs pointwise XGBoost',
                           r'$1 - {\rm MSE}/{\rm MSE}_{\rm XGB}$'))
    elif has_bias:
        panel_defs.append(('bias', r'Bias (dex)', r'$\langle \hat{y}-y \rangle$ (dex)'))

    for idx, (ax, (key, title, ylabel)) in enumerate(zip(axes, panel_defs)):
        for name, v in variants.items():
            label, color, marker, ls = VARIANT_STYLE.get(
                name, (name, '0.4', 'x', '-'))
            vals = np.array(_fold_values(v, key), dtype=float)
            if np.all(np.isnan(vals)):
                continue
            ax.plot(g0, vals, color=color, marker=marker, ls=ls,
                    ms=3.5, lw=1.0, label=label)
        if key in ('skill_vs_xgb', 'bias'):
            ax.axhline(0.0, color='0.5', lw=0.6, ls='-')
        if key == 'R2' and args.r2_min is not None:
            ax.set_ylim(bottom=args.r2_min)
        ax.set_xscale('log')
        ax.set_xticks(g0)
        ax.set_xticklabels([f'{g:g}' for g in g0], fontsize=6.5)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax.set_xlabel(r'Held-out $G_0$ (Habing)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # Mark extrapolation boundaries
        for gb in (g0.min(), g0.max()):
            ax.axvline(gb, color='0.8', lw=4.0, zorder=0)
        ax.text(-0.13, 1.03, f'({chr(97 + idx)})', transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='bottom', ha='left')

    axes[0].legend(fontsize=6, loc='lower left')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    fig.savefig(args.save)
    plt.close(fig)
    print(f'Saved: {args.save}')


if __name__ == '__main__':
    main()
