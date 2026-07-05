"""
plot_nH2_histograms.py
2x3 grid of log10(nH2) histograms for all G0 values except 0.2.
"""

import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_single_cube

G0_VALUES = [0.1, 0.4, 0.8, 1.6, 3.2, 6.4]
COLORS    = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True, sharex=True)
axes_flat = axes.flatten()

for ax, g0, color in zip(axes_flat, G0_VALUES, COLORS):
    df = load_single_cube(g0)
    vals = df['log_nH2'].values

    ax.hist(vals, bins=80, color=color, edgecolor='none', alpha=0.85)

    mean = np.mean(vals)
    ax.axvline(mean, color='black', linewidth=1.2, linestyle='--',
               label=f'mean = {mean:.1f}')

    ax.set_title(f'G$_0$ = {g0}', fontsize=13)
    ax.legend(fontsize=9, framealpha=0.7)

for ax in axes[1]:
    ax.set_xlabel(r'$\log_{10}(n_{\mathrm{H_2}})$', fontsize=11)
for ax in axes[:, 0]:
    ax.set_ylabel('Cell count', fontsize=11)

fig.suptitle(r'Distribution of $\log_{10}(n_{\mathrm{H_2}})$ across UV field strengths',
             fontsize=13, y=0.97)

plt.tight_layout()
plt.savefig('figures/nH2_histograms.png', dpi=150, bbox_inches='tight')
print("Saved: figures/nH2_histograms.png")
plt.show()
