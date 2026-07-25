#!/usr/bin/env python3
"""
plot_feature_importance.py
==========================
XGBoost feature-importance figure from an arch_comparison_*.json log that
contains 'xgb_feature_importance' blocks (runs made after the 2026-07
metrics overhaul; older logs did not record importances).

Panels
------
  (a) xgb_standard: gain importance of the local features, mean +/- std
      across the 7 leave-one-G0-out folds.
  (b) xgb_standard_sp: top-N individual features of the spatial variant
      (local features plus k=3/5/7 neighbourhood means, suffixes _k3 ...).
  (c) xgb_standard_sp aggregated per base field (local + all scales summed),
      showing which physical fields carry the signal regardless of scale.

Usage
-----
  python plot_feature_importance.py                    # latest results/arch_comparison_*.json
  python plot_feature_importance.py --log <path> --top-n 20
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from viz_common import apply_journal_style

apply_journal_style()

_C_LOCAL   = '#2166ac'   # local features
_C_SPATIAL = '#d6604d'   # neighbourhood-mean features


def _latest_log() -> str | None:
    logs = sorted(glob.glob('results/arch_comparison_*.json'))
    return logs[-1] if logs else None


def _importance(variant: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (names, mean, std) across folds for one variant block."""
    fi    = variant['xgb_feature_importance']
    per   = np.asarray(fi['per_fold'], dtype=float)   # (n_folds, n_features)
    return fi['feature_names'], per.mean(axis=0), per.std(axis=0)


def _importance_per_fold(variant: dict) -> tuple[list[str], np.ndarray]:
    """Return (names, per_fold) with per_fold of shape (n_folds, n_features),
    for aggregations that must sum across features fold-by-fold rather than
    combine already-collapsed per-feature means/stds (which would wrongly
    assume feature importances are independent across scales)."""
    fi = variant['xgb_feature_importance']
    return fi['feature_names'], np.asarray(fi['per_fold'], dtype=float)


def _barh(ax, names, mean, std, colors, title):
    y = np.arange(len(names))
    ax.barh(y, mean, xerr=std, color=colors, edgecolor='k', linewidth=0.4,
            height=0.7, error_kw=dict(lw=0.6, capsize=1.5))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.5)
    ax.invert_yaxis()                     # most important on top
    ax.set_xlabel('Gain importance')
    ax.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='XGBoost feature-importance figure from an arch_comparison JSON.')
    parser.add_argument('--log', default=None,
                        help='arch_comparison JSON (default: latest in results/)')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Features shown in the spatial-variant panel (default 20)')
    parser.add_argument('--save', default='figures/fig_feature_importance.png',
                        help='Output PNG path')
    args = parser.parse_args()

    log_path = args.log or _latest_log()
    if log_path is None or not os.path.exists(log_path):
        print('No results/arch_comparison_*.json found; pass one with --log', file=sys.stderr)
        sys.exit(1)
    with open(log_path) as f:
        log = json.load(f)
    print(f'Log: {log_path}')

    have = {n: v for n, v in log['variants'].items()
            if 'xgb_feature_importance' in v}
    if not have:
        print('This log has no xgb_feature_importance blocks — rerun '
              'compare_architectures.py (importances are recorded since the '
              '2026-07 metrics overhaul).', file=sys.stderr)
        sys.exit(1)

    base = have.get('xgb_standard')
    spat = have.get('xgb_standard_sp')
    n_panels = (1 if base else 0) + (2 if spat else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 4.2))
    axes = np.atleast_1d(axes)
    panel = 0

    # (a) local-feature variant
    if base:
        names, mean, std = _importance(base)
        order = np.argsort(mean)[::-1]
        _barh(axes[panel], [names[i] for i in order], mean[order], std[order],
              _C_LOCAL, 'XGBoost (local features)')
        axes[panel].text(-0.22, 1.02, f'({chr(97 + panel)})',
                         transform=axes[panel].transAxes,
                         fontsize=8, fontweight='bold')
        panel += 1

    if spat:
        names, mean, std = _importance(spat)
        is_spatial = np.array([bool(re.search(r'_k\d+$', n)) for n in names])

        # (b) top-N individual features
        order = np.argsort(mean)[::-1][:args.top_n]
        colors = [_C_SPATIAL if is_spatial[i] else _C_LOCAL for i in order]
        _barh(axes[panel], [names[i] for i in order], mean[order], std[order],
              colors, f'XGBoost + spatial (top {args.top_n})')
        axes[panel].text(-0.22, 1.02, f'({chr(97 + panel)})',
                         transform=axes[panel].transAxes,
                         fontsize=8, fontweight='bold')
        panel += 1

        # (c) aggregated per base field: local + all neighbourhood scales.
        # Sum importances per FOLD first (7 aggregate values per field), then
        # take mean/std of those — NOT sqrt(sum(std_i^2)) over features, which
        # would assume per-feature importances are independent. They aren't:
        # XGBoost gain importance is compositional/correlated across the
        # local feature and its k3/k5/k7 neighbourhood means of the same
        # physical field.
        names_pf, per_fold = _importance_per_fold(spat)
        roots_all = [re.sub(r'_k\d+$', '', n) for n in names_pf]
        unique_roots = sorted(set(roots_all))
        agg_per_fold = np.zeros((per_fold.shape[0], len(unique_roots)))
        for j, root in enumerate(unique_roots):
            cols = [i for i, r in enumerate(roots_all) if r == root]
            agg_per_fold[:, j] = per_fold[:, cols].sum(axis=1)
        order_c = np.argsort(agg_per_fold.mean(axis=0))[::-1]
        roots = [unique_roots[i] for i in order_c]
        vals  = agg_per_fold.mean(axis=0)[order_c]
        sds   = agg_per_fold.std(axis=0)[order_c]
        _barh(axes[panel], roots, vals, sds, '0.55',
              'Aggregated per field (all scales)')
        axes[panel].text(-0.22, 1.02, f'({chr(97 + panel)})',
                         transform=axes[panel].transAxes,
                         fontsize=8, fontweight='bold')

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
    fig.savefig(args.save)
    plt.close(fig)
    print(f'Saved: {args.save}')


if __name__ == '__main__':
    main()
