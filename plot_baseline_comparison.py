"""
plot_baseline_comparison.py
Grouped bar chart (log-space AND linear-space R2) for the three baseline
models: Linear Regression, XGBoost, MLP.
XGBoost and MLP results are loaded from the latest arch_comparison JSON.
Linear Regression is run inline (fast, deterministic).
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_all_cubes, get_X_y, get_g0_values
from classical_models import run_linear

LOG    = 'arch_comparison_20260311_231200.json'
OUTPNG = 'baseline_comparison.png'
COLORS = ['#4C72B0', '#55A868', '#C44E52']

# ── load XGB / MLP folds from JSON ────────────────────────────────────────────
with open(LOG) as f:
    log = json.load(f)

xgb_r2     = [fold['metrics']['R2']     for fold in log['variants']['xgb_standard']['folds']]
xgb_r2_lin = [fold['metrics']['R2_lin'] for fold in log['variants']['xgb_standard']['folds']]
mlp_r2     = [fold['metrics']['R2']     for fold in log['variants']['mlp_wide']['folds']]
mlp_r2_lin = [fold['metrics']['R2_lin'] for fold in log['variants']['mlp_wide']['folds']]

# ── run linear regression CV ──────────────────────────────────────────────────
print("Running linear regression CV...")
cubes   = load_all_cubes()
g0_vals = get_g0_values(cubes)
X, y, folds = get_X_y(cubes, use_log_target=True)
lr_metrics = run_linear(X, y, folds, g0_vals)
lr_r2     = [m['R2']     for m in lr_metrics]
lr_r2_lin = [m['R2_lin'] for m in lr_metrics]

# ── data ──────────────────────────────────────────────────────────────────────
models = ['Linear\nRegression', 'XGBoost', 'MLP']

log_folds = [np.array(lr_r2),     np.array(xgb_r2),     np.array(mlp_r2)]
lin_folds = [np.array(lr_r2_lin), np.array(xgb_r2_lin), np.array(mlp_r2_lin)]

log_means = [a.mean() for a in log_folds]
log_stds  = [a.std()  for a in log_folds]
lin_means = [a.mean() for a in lin_folds]
lin_stds  = [a.std()  for a in lin_folds]

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
x   = np.arange(len(models))
rng = np.random.default_rng(0)

LIN_YMIN, LIN_YMAX = -0.25, 0.60   # clip axis; MLP has extreme negatives

for ax, means, stds, fold_data, ylabel, ylim, title in [
    (axes[0], log_means, log_stds, log_folds,
     r'$R^2$  (log$_{10}(n_{\mathrm{H_2}})$ space)',
     (0.75, 1.08),
     r'Log-space $R^2$'),
    (axes[1], lin_means, lin_stds, lin_folds,
     r'$R^2$  (linear $n_{\mathrm{H_2}}$ space)',
     (LIN_YMIN, 1),
     r'Linear-space $R^2$'),
]:
    bar_means = np.clip(means, ylim[0], ylim[1])
    bars = ax.bar(x, bar_means,
                  color=COLORS, alpha=0.82, edgecolor='none', width=0.5)

    # error bars (clipped to visible range)
    for xi, mean, std in zip(x, means, stds):
        lo = max(mean - std, ylim[0])
        hi = min(mean + std, ylim[1])
        ax.plot([xi, xi], [lo, hi], color='#333333', linewidth=1.5)
        for cap_y in [lo, hi]:
            ax.plot([xi - 0.07, xi + 0.07], [cap_y, cap_y],
                    color='#333333', linewidth=1.5)

    # per-fold scatter (clipped to visible range)
    for xi, fold_vals, color in zip(x, fold_data, COLORS):
        clipped = np.clip(fold_vals, ylim[0], ylim[1])
        jitter  = rng.uniform(-0.10, 0.10, size=len(clipped))
        ax.scatter(xi + jitter, clipped, color=color, edgecolors='white',
                   linewidths=0.6, s=40, zorder=3, alpha=0.9)

    # mean value labels
    for xi, mean, std, bm in zip(x, means, stds, bar_means):
        label_y = min(bm + abs(std) + 0.012, ylim[1] - 0.07)
        ax.text(xi, label_y, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=13)

# annotate the MLP extreme values on the linear-space subplot
mlp_mean_lin = np.mean(mlp_r2_lin)
axes[1].annotate(f'MLP mean = {mlp_mean_lin:.1f}\n(clipped)',
                 xy=(1.75, LIN_YMIN + 0.02), xytext=(1.4, LIN_YMIN + 0.14),
                 fontsize=9, color="#000000",
                 arrowprops=dict(arrowstyle='->', color="#000000", lw=1.2))

fig.suptitle('Baseline models — leave-one-G0-out CV  (7 folds)', fontsize=13, y=0.97)
fig.text(0.95, -0.02,
         'dots = individual folds  |  bar = mean  |  whiskers = ±1 std  |  linear-space axis clipped',
         ha='right', va='bottom', fontsize=8, color='#666666')

plt.tight_layout()
plt.savefig(OUTPNG, dpi=150, bbox_inches='tight')
print(f"Saved: {OUTPNG}")
plt.show()
