"""
plot_slide6_ablation.py
Ablation bar chart for Slide 6: progressive feature-engineering gains.
R² values from actual JSON logs (leave-one-G0-out, log-space metric).
Saves slide6_ablation.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt


# ── data from actual runs ──────────────────────────────────────────────────────
# Step 1+2 from arch_comparison_20260309_085656  (single-scale 3³, no kernels key)
# Step 3+4 from arch_comparison_20260309_121724  (kernels=[3,5,7])

LOG1 = 'arch_comparison_20260309_085656.json'
LOG2 = 'arch_comparison_20260309_121724.json'

with open(LOG1) as f:
    d1 = json.load(f)
with open(LOG2) as f:
    d2 = json.load(f)

def fold_r2(log, variant):
    return np.array([fold['metrics']['R2']
                     for fold in log['variants'][variant]['folds']])

r2_base   = fold_r2(d1, 'xgb_standard')       # no spatial
r2_s3     = fold_r2(d1, 'xgb_standard_sp')    # +3³ only
r2_s357   = fold_r2(d2, 'xgb_standard_sp')    # +3³+5³+7³
r2_ens    = fold_r2(d2, 'ens_sp')             # ensemble

steps = [r2_base, r2_s3, r2_s357, r2_ens]
means = [a.mean() for a in steps]
stds  = [a.std()  for a in steps]

labels = [
    'XGB\n(15 features)',
    '+3³ spatial\n(30 features)',
    '+5³+7³ multi-scale\n(60 features)',
    '+MLP ensemble\n(ens_sp)',
]
COLORS = ['#4C72B0', '#55A868', '#2ca02c', '#C44E52']

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))

x    = np.arange(len(labels))
bars = ax.bar(x, means, color=COLORS, alpha=0.83, edgecolor='none', width=0.55)

# error bars
ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#333333',
            elinewidth=1.8, capsize=6, capthick=1.8, zorder=5)

# per-fold dots
rng = np.random.default_rng(1)
for xi, folds, color in zip(x, steps, COLORS):
    jitter = rng.uniform(-0.12, 0.12, size=len(folds))
    ax.scatter(xi + jitter, folds, color=color, edgecolors='white',
               linewidths=0.7, s=45, zorder=6, alpha=0.92)

# delta labels between bars
for i in range(1, len(means)):
    delta = means[i] - means[i - 1]
    mid_x = (x[i - 1] + x[i]) / 2
    mid_y = max(means[i - 1], means[i]) + max(stds[i - 1], stds[i]) + 0.005
    ax.annotate(f'+{delta:.3f}',
                xy=(x[i], means[i]), xytext=(mid_x, mid_y + 0.018),
                ha='center', va='bottom', fontsize=10, color='#444444',
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=1.2, color='#888888',
                                connectionstyle='arc3,rad=-0.15'))

# mean labels inside bars (near bottom so they don't overlap delta annotations)
for xi, mean in zip(x, means):
    ax.text(xi, 0.833, f'{mean:.3f}',
            ha='center', va='bottom', fontsize=10.5, fontweight='bold',
            color='white', zorder=7)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel(r'Mean $R^2$  (log$_{10}(n_{\mathrm{H_2}})$, 7-fold CV)', fontsize=11)
ax.set_ylim(0.82, 1.02)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.set_title('Ablation: progressive spatial feature gains', fontsize=13)

fig.text(0.99, 0.01,
         'dots = individual G0 folds  |  whiskers = ±1 std',
         ha='right', va='bottom', fontsize=8, color='#666666')

plt.tight_layout()
plt.savefig('slide6_ablation.png', dpi=150, bbox_inches='tight')
print("Saved: slide6_ablation.png")
plt.show()
