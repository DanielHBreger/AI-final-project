"""
plot_slide6_diagram.py
Key-idea concept diagram for Slide 6: spatial neighbourhood features.
Saves slide6_diagram.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig = plt.figure(figsize=(13, 6.5))
fig.patch.set_facecolor('white')

ax_l   = fig.add_axes([0.01, 0.05, 0.44, 0.88])
ax_r   = fig.add_axes([0.55, 0.05, 0.44, 0.88])
ax_div = fig.add_axes([0.455, 0.1, 0.008, 0.8])
ax_div.set_facecolor('#cccccc')
ax_div.axis('off')

CELL_FC, CELL_EC       = '#BBDEFB', '#1565C0'
BOX3_FC, BOX3_EC       = '#FFF9C4', '#F9A825'
BOX5_FC, BOX5_EC       = '#C8E6C9', '#2E7D32'
BOX7_FC, BOX7_EC       = '#FCE4EC', '#C62828'
MODEL_FC, MODEL_EC     = '#EDE7F6', '#4527A0'
OUT_COLOR              = '#1A237E'
ARR = dict(arrowstyle='->', lw=2.0, color='#444444')


def grid_cells(ax, ox, oy, n, cell_sz, alpha=0.55, transparent_bg=False):
    for row in range(n):
        for col in range(n):
            centre = (row == n // 2 and col == n // 2)
            # transparent_bg=True: let box colors show through; only borders visible
            fc = CELL_FC if centre else ('none' if transparent_bg else '#F5F5F5')
            r = mpatches.Rectangle(
                (ox + col * cell_sz, oy + row * cell_sz), cell_sz, cell_sz,
                fc=fc,
                ec=CELL_EC if centre else '#888888',
                lw=2.0 if centre else 0.8,
                alpha=alpha if centre else 1.0,
                zorder=4)
            ax.add_patch(r)


def fbox(ax, x, y, w, h, fc, ec, lw=2, alpha=0.55, zorder=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder))


def fbar(ax, x, y, w, h, fc, ec, text, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                                fc=fc, ec=ec, lw=1.5, zorder=4))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=ec, fontweight='bold', zorder=5)


def model_box(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                fc=MODEL_FC, ec=MODEL_EC, lw=2.5, zorder=4))
    ax.text(x + w / 2, y + h / 2, 'Model\n(XGB / MLP)',
            ha='center', va='center', fontsize=10.5,
            color=MODEL_EC, fontweight='bold', zorder=5)


for ax in [ax_l, ax_r]:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

# ══════════════════════════════════════════════════════════════════════════════
#  LEFT PANEL
# ══════════════════════════════════════════════════════════════════════════════
ax_l.set_title('Traditional: per-cell model',
               fontsize=13, fontweight='bold', color='#333333', pad=8)

grid_cells(ax_l, ox=3.5, oy=6.2, n=3, cell_sz=1.0)
ax_l.text(5.0, 7.85, 'cell i', ha='center', va='center',
          fontsize=9, fontweight='bold', color=CELL_EC, zorder=6)

fbar(ax_l, x=1.5, y=5.0, w=7.0, h=0.8, fc=CELL_FC, ec=CELL_EC,
     text='15 local features   [T, nH, G\u2080, p, vx, vy, vz, ...]')

ax_l.annotate('', xy=(5.0, 5.8), xytext=(5.0, 6.2), arrowprops=dict(**ARR))
ax_l.annotate('', xy=(5.0, 3.9), xytext=(5.0, 5.0),  arrowprops=dict(**ARR))

model_box(ax_l, x=3.0, y=2.65, w=4.0, h=1.2)

ax_l.annotate('', xy=(5.0, 1.6), xytext=(5.0, 2.65), arrowprops=dict(**ARR))
ax_l.text(5.0, 1.3, r'$\hat{n}_{\mathrm{H_2}}$',
          ha='center', va='center', fontsize=16, fontweight='bold',
          color=OUT_COLOR, zorder=5)

ax_l.text(5.0, 0.4,
          'Misses shielding context — two cells\n'
          'with same T, nH can have very different nH\u2082',
          ha='center', va='center', fontsize=9, color='#B71C1C', style='italic',
          bbox=dict(fc='#FFEBEE', ec='#EF9A9A', lw=1, boxstyle='round,pad=0.4'))

# ══════════════════════════════════════════════════════════════════════════════
#  RIGHT PANEL
# ══════════════════════════════════════════════════════════════════════════════
ax_r.set_title('Ours: spatial neighbourhood features',
               fontsize=13, fontweight='bold', color='#333333', pad=8)

# ── grid + nested boxes derived from same parameters ─────────────────────────
# Grid is 7×7 so each scale (3³/5³/7³) maps to exact cell boundaries.
C    = 0.60          # cell size in data units
GX   = 2.5           # grid origin x  (7×7 → spans x: 2.5 – 9.5)
GY   = 5.5           # grid origin y  (7×7 → spans y: 5.5 – 9.7)
N    = 7

# Box corners are computed from the grid so they align perfectly with cell edges.
# 3³: inner 3×3 = rows/cols 1–3  → offset 1 cell inward from grid origin
b3x, b3y, b3s = GX+2*C,          GY+2*C,         3 * C   # (4.1, 6.9, 1.8)
# 5³: the full 5×5 grid
b5x, b5y, b5s = GX+C,         GY+C,             5 * C   # (3.5, 6.3, 3.0)
# 7³: extends 1 cell outward beyond the grid
b7x, b7y, b7s = GX,         GY,           7 * C   # (2.9, 5.7, 4.2)

# grid drawn last so cells sit on top of the colored boxes
grid_cells(ax_r, ox=GX, oy=GY, n=N, cell_sz=C, alpha=0.85, transparent_bg=True)

# center cell label (center cell is at row=2, col=2)
ax_r.text(GX + 2.5*C, GY + 3.0*C + 0.05, 'cell i',
          ha='center', va='bottom', fontsize=8.5,
          fontweight='bold', color=CELL_EC, zorder=7)

# draw largest → smallest so smaller boxes sit on top
fbox(ax_r, x=b7x, y=b7y, w=b7s, h=b7s, fc=BOX7_FC, ec=BOX7_EC, lw=2.0)
fbox(ax_r, x=b5x, y=b5y, w=b5s, h=b5s, fc=BOX5_FC, ec=BOX5_EC, lw=2.0)
fbox(ax_r, x=b3x, y=b3y, w=b3s, h=b3s, fc=BOX3_FC, ec=BOX3_EC, lw=2.0)

# box labels at bottom-left of each box, staggered so they never overlap
ax_r.text(b7x + 0.08, b7y + 0.08, '7³ neighbourhood avg',
          fontsize=8.5, color=BOX7_EC, fontweight='bold', va='bottom', zorder=6)
ax_r.text(b5x + 0.08, b5y + 0.08, '5³ neighbourhood avg',
          fontsize=8.5, color=BOX5_EC, fontweight='bold', va='bottom', zorder=6)
ax_r.text(b3x + 0.08, b3y + 0.08, '3³ neighbourhood avg',
          fontsize=8.5, color=BOX3_EC, fontweight='bold', va='bottom', zorder=6)

# ── arrow: bottom of 7³ box → feature bars ────────────────────────────────────
ax_r.annotate('', xy=(5.0, 5.3), xytext=(5.0, b7y), arrowprops=dict(**ARR))

# ── feature bars (4 bars, evenly spaced) ─────────────────────────────────────
bar_y   = [4.55, 3.8, 3.05, 2.3]
bar_dat = [
    (CELL_FC, CELL_EC, '15 local features'),
    (BOX3_FC, BOX3_EC, '15 features \u00d7 3\u00b3 avg'),
    (BOX5_FC, BOX5_EC, '15 features \u00d7 5\u00b3 avg'),
    (BOX7_FC, BOX7_EC, '15 features \u00d7 7\u00b3 avg'),
]
for (fc, ec, txt), yb in zip(bar_dat, bar_y):
    fbar(ax_r, x=1.3, y=yb, w=7.2, h=0.62, fc=fc, ec=ec, text=txt)

# "= 60 features" label centred below the bar stack
ax_r.text(5.0, 2.12, '= 60 features total',
          ha='center', va='top', fontsize=10, fontweight='bold', color='#555555')

# ── arrow: bars → model ───────────────────────────────────────────────────────
ax_r.annotate('', xy=(5.0, 1.55), xytext=(5.0, 2.1), arrowprops=dict(**ARR))

# ── model box ─────────────────────────────────────────────────────────────────
model_box(ax_r, x=3.1, y=0.3, w=3.8, h=1.2)

# ── horizontal arrow → output ─────────────────────────────────────────────────
ax_r.annotate('', xy=(8.3, 0.9), xytext=(6.9, 0.9), arrowprops=dict(**ARR))
ax_r.text(8.8, 0.9, r'$\hat{n}_{\mathrm{H_2}}$',
          ha='center', va='center', fontsize=16, fontweight='bold',
          color=OUT_COLOR, zorder=5)

# ── improvement note ──────────────────────────────────────────────────────────
# ax_r.text(4, -0.35,
#           'Neighbourhood avg encodes UV shielding depth\u2002\u00b7\u2002'
#           'deterministic\u2002\u00b7\u2002O(N)\u2002\u00b7\u2002+3.0% R\u00b2  (15 \u2192 60 features)',
#           ha='center', va='center', fontsize=9, color='#1B5E20', style='italic',
#           bbox=dict(fc='#E8F5E9', ec='#A5D6A7', lw=1, boxstyle='round,pad=0.4'))

plt.savefig('slide6_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: slide6_diagram.png")
plt.show()
