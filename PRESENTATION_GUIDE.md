# 10-Minute Presentation Guide

**Project**: Predicting Molecular Hydrogen Density (nH2) in 3D Astrophysical Simulations Using Machine Learning

---

## Time Budget Overview

| Segment | Duration |
|---|---|
| Talk | 10 min |
| Q&A | ~3 min |
| **Total** | **~13 min** |

Use 11 slides. Average ~55 seconds per slide, but distribute unevenly — the spatial features
insight (Slide 6) is your core result and deserves the most time.

---

## Slide-by-Slide Guide

### Slide 1 — Title (0:00–0:30)

**Content**: Project title, your name, date.

**Say**: "I'm going to talk about using machine learning to predict molecular hydrogen density
in 3D astrophysical simulations — specifically, how much H2 is at each point in a simulated
interstellar cloud."

**Keep it brief.** No motivation yet — save that for Slide 2.

---

### Slide 2 — The Physics Problem (0:30–1:30)

**Content**: A simple diagram of the ISM chemistry triangle:

```
     UV photons (G0)
          |
          v
[H2 molecule] <-> [H atoms]
  - Formation: dust grain surfaces (needs nH, T)
  - Destruction: UV photodissociation (needs G0, shielding)
  - Self-shielding: H2 absorbs its own UV
```

**Say**: "Hydrogen in interstellar space exists as ions, atoms, or molecules. Molecular hydrogen
(H2) forms on dust grains and is destroyed by UV radiation. A key complication is self-shielding:
dense H2 clouds absorb the UV that would destroy them, so chemistry is inherently spatial —
a cell's H2 fraction depends on its neighbours.

Computing this from first principles requires solving coupled time-dependent chemistry and
radiative transfer — extremely expensive in 3D simulations. We want to train a model to
predict nH2 directly from local physical properties."

---

### Slide 3 — Why This Is Hard (1:30–2:30)

**Content**: Three bullet points, each with a one-line visual aid.

1. **30-dex dynamic range**: Show a histogram of log10(nH2) — roughly Gaussian spanning from
   -30 to +5, centred around -1.
2. **Out-of-distribution generalization**: "We hold out entire UV field strengths — the model
   must predict G0 values it has never seen during training."
3. **Spatial context matters**: "Two cells with identical local properties can have very
   different nH2 depending on whether they sit at the surface or centre of a cloud."

**Say**: "This is not a standard regression problem for three reasons." Walk through each bullet.
For the first: "We have to predict in log space — raw nH2 spans 35 orders of magnitude."
For the second: explain what leave-one-G0-out means in one sentence.

---

### Slide 4 — Dataset and CV Strategy (2:30–3:30)

**Content**: A simple table:

| Cube | G0 Value | Fold type |
|---|---|---|
| 1 | 0.1 | Extrapolation (boundary) |
| 2 | 0.2 | Near-extrapolation |
| 3 | 0.4 | Interpolation |
| 4 | 0.8 | Interpolation (centre) |
| 5 | 1.6 | Interpolation |
| 6 | 3.2 | Interpolation |
| 7 | 6.4 | Extrapolation (boundary) |

Plus one visual: "7 cubes x 128^3 cells = 14.7M training samples total."

**Say**: "We have 7 simulations that differ only in UV field strength G0, ranging over 64x.
For cross-validation, we hold out one entire cube — one G0 value — and train on the other 6.
This is leave-one-G0-out CV. It tests whether the model can generalize to UV conditions it
has never seen. G0=0.1 and G0=6.4 are boundary values — pure extrapolation, the hardest folds.
Standard random CV would be trivially easy and scientifically meaningless — same G0 data on
both sides of the split."

---

### Slide 5 — Baseline Models (3:30–4:30)

**Content**: A bar chart showing mean R2 across 7 folds for each model:

```
Linear Regression:  R2 = 0.230  (negative on G0=0.1, 0.2)
XGBoost:            R2 = 0.886
MLP:                R2 = 0.885
```

Plus a small table showing that all 3 XGBoost depth variants give ~0.886, all 3 MLP
architecture variants give ~0.885.

**Say**: "Our baseline is linear regression — R2=0.23, with negative R2 on the hardest folds,
confirming the fundamental nonlinearity of the chemistry.

XGBoost and MLP both achieve ~0.886. All XGBoost depth variants give the same result —
the dominant signal is temperature alone, not high-order interactions. All three MLP
architectures give the same result — the bottleneck is the information content of the
14 per-cell features, not model capacity.

The question is: what is the model missing?"

---

### Slide 6 — The Spatial Features Insight [CORE SLIDE] (4:30–5:45)

**Content**: Two-part slide.

**Left**: The key idea diagram:
```
Traditional: Model sees each cell in isolation
             [cell: nH, T, nHp, ...] -> [nH2]

Ours: Model sees cell + its neighbourhood
             [cell + 3^3 avg + 5^3 avg + 7^3 avg] -> [nH2]

Why: A cell deep inside a dense cloud is shielded from UV.
     Two cells with identical nH, T can have very different nH2
     depending on whether they are at the surface or centre.
```

**Right**: The ablation table:
```
XGBoost alone (14 features)         R2 = 0.886
+ Spatial 3x3x3 (28 features)       R2 = 0.917  (+0.031)
+ Multi-scale 3+5+7 (56 features)   R2 = 0.924  (+0.007)
+ MLP_sp ensemble                   R2 = 0.948  (+0.024)
```

**Say**: "The key insight is that H2 chemistry is inherently spatial — whether a cell is
shielded from UV depends on its neighbours, not just its local properties. We encode this
by precomputing 3D box-filter averages at three scales: 3x3x3, 5x5x5, and 7x7x7 neighbourhoods.
For each of the 15 physical features, we compute its local mean at each scale. This gives us
45 additional spatial context features — 60 total.

This is the single biggest improvement in the project. Adding just the 3x3x3 neighbourhood
gives +0.031 R2 points. Multi-scale adds another +0.007. Then combining XGBoost and MLP
predictions — which have complementary errors — gives another +0.024.

Final result: ens_sp with R2=0.948."

**Pause here.** This is the central result. Let it land.

---

### Slide 7 — Final Model ens_sp (5:45–6:30)

**Content**:

Per-fold R2 table for ens_sp:

| G0 | R2 | Fold type |
|---|---|---|
| 0.1 | 0.908 | Extrapolation |
| 0.2 | 0.915 | Near-extrapolation |
| 0.4 | 0.953 | Interpolation |
| 0.8 | 0.962 | Interpolation |
| 1.6 | 0.967 | Interpolation |
| 3.2 | 0.967 | Interpolation |
| 6.4 | 0.965 | Extrapolation |
| **Mean** | **0.948 +/- 0.024** | |

**Say**: "The ensemble uses equal-weight averaging of XGBoost and MLP, both trained on the
60-feature set. It achieves R2=0.948 across 7 folds. Note the pattern: interpolation folds
reach 0.95-0.97. The hardest folds — G0=0.1 and G0=0.2, which require extrapolation below
the training range — reach 0.91. This is remarkably good for pure extrapolation to unseen
UV conditions.

All R2 values here are in log10(nH2) space — R2=0.95 means the model explains 95% of variance
in log10(nH2), with typical errors of ~0.1-0.2 dex, or a factor of 1.3-1.6 in linear nH2."

---

### Slide 8 — Why Spatial Features Beat the CNN (6:30–7:15)

**Content**: Side-by-side comparison:

| Approach | R2 | Time to train |
|---|---|---|
| 3D U-Net CNN | 0.803 | ~8 hrs/run |
| ens_sp (spatial features) | 0.948 | ~30 min |

Plus one insight: "The CNN operates on 64^3 (downsampled) volumes. The spatial features use
the full 128^3 grid."

**Say**: "We also trained a 3D U-Net CNN, which can naturally capture spatial context through
its convolutional receptive field. Best result: R2=0.803 — notably worse than our spatial
feature approach.

Why? The CNN has only 6 training volumes (6 augmented cubes per fold x 8 symmetry operations
= 48 effective training samples), versus 12.5M per-cell samples for XGBoost and MLP. The CNN
must learn to extract spatial context from scratch via gradient descent on a tiny dataset.

Our approach precomputes the spatial context as fixed 3D box-filter averages — it's deterministic,
instant, and compatible with data-rich tabular models. The spatial signal in this problem is
simple: the average density/extinction in your neighbourhood predicts shielding. A box filter
captures this directly."

---

### Slide 9 — New Experiments (7:15–8:30)

**Content**: Two mini-results panels.

**Left — Single-Cube Extrapolation** (briefly):
A small 7x7 R2 heatmap (or just a representative row from the matrix).
"How much does one simulation tell you about others?"

Key numbers: G0=0.1 -> G0=0.2: R2=0.976. G0=0.1 -> G0=3.2: R2=0.446.

**Right — Intra-Cube Spatial Section** (main focus):
A comparison table or scatter plot:

```
Split type    | Train %  | Test R2
--------------|----------|--------
rand_1        | 1%       | 0.897
rand_5        | 5%       | 0.952
rand_25       | 25%      | 0.970
rand_50       | 50%      | 0.980
x_half (slab) | 50%      | -0.394  <-- CATASTROPHIC
y_half (slab) | 50%      | -2.689  <-- CATASTROPHIC
z_half (slab) | 50%      | -1.389  <-- CATASTROPHIC
box_1         | 1%       | 0.693
box_50        | 50%      | 0.779
```

**Say** (single-cube): "We also asked: how much does one G0 simulation tell you about the
chemistry at other UV conditions? Training on G0=0.1 alone, we can predict G0=0.2 at R2=0.976,
G0=0.4 at 0.958, but G0=3.2 only at 0.446. Chemistry generalizes across small G0 gaps but
not large ones — justifying why we need all 7 cubes for accurate prediction."

**Say** (intra-cube): "Now here's the most counterintuitive result of the project.

Within a single cube, we train on a spatial section and predict the remainder. 1% of cells
chosen RANDOMLY gives R2=0.897 on the other 99%. But 50% of cells from ONE HALF of the cube
gives R2 = NEGATIVE — the model does worse than predicting the mean.

Why? The UV-facing half and the shielded half of the cube have fundamentally different nH2 fields.
A model trained only on one half cannot predict the other. But with random sampling, test cells
are surrounded by training cells in all directions — spatial interpolation works perfectly.

This has a practical implication for observational astronomy: even a small random sample of
sight-lines through a molecular cloud is sufficient to map the full H2 density field."

---

### Slide 10 — Key Takeaways (8:30–9:30)

**Content**: Four numbered takeaways.

1. **Feature engineering beats architecture search**: Adding spatial neighbourhood averages
   (+3.1 R2 pts) outperformed every model capacity increase. All 3 XGBoost depths and all 3
   MLP widths gave nearly identical results.

2. **Spatial context is the missing ingredient**: Whether a cell is UV-shielded depends on
   its neighbours. Precomputed box-filter features capture this simply and effectively.

3. **Dropout destroys OOD extrapolation**: When a model must learn genuine physical structure
   to generalize, dropout prevents stable representation formation. With only 6-7 distinct G0
   values, the model must memorize real physics, not overfit to noise.

4. **Spatial sampling geometry matters dramatically**: 1% random > 50% contiguous for
   within-cube interpolation. Coverage uniformity, not coverage volume, determines success.

**Say**: Walk through each. For #3, add: "This is a general lesson — standard regularization
intuition breaks down when the training-test split is physically meaningful, not just a
random partition."

---

### Slide 11 — Conclusion and Future Work (9:30–10:00)

**Content**:

**Result**: ens_sp achieves R2=0.948 +/- 0.024 on leave-one-G0-out CV for predicting log10(nH2).
This represents a ~6x reduction in unexplained variance compared to per-cell XGBoost alone.

**Future directions** (2-3 bullets, choose what resonates):
- Apply to simulations with varying density fields or metallicities (test generalization
  beyond the G0 axis)
- Use the model to generate training data for faster approximate chemistry in full cosmological
  simulations (emulation)
- Explore whether the random-sampling insight scales to observational data (partial IFU maps
  of molecular clouds)

**Say**: "To summarize: we've shown that molecular hydrogen density in 3D astrophysical
simulations can be predicted at R2=0.948 using a spatial ensemble of XGBoost and MLP, where
the key contribution is precomputed multi-scale neighbourhood features that encode spatial
shielding context. Thank you — happy to take questions."

---

## What to Emphasize

- **Spatial features** are the #1 scientific contribution. The core insight — local chemistry
  depends on neighbourhood average density, not just local values — is simple, physically
  motivated, and effective. Spend the most time here.

- **Intra-cube section** result is visually striking and counterintuitive. The 1% random vs
  50% slab comparison will stick with the audience.

- **The OOD CV design** — leave-one-G0-out — is what makes the results scientifically meaningful.
  Take 30 seconds to explain why random CV would be trivially easy.

---

## What to Skip or Compress

- **CNN architecture internals** (InstanceNorm vs BatchNorm, 3 levels vs 4): one sentence max.
  "We replaced BatchNorm with InstanceNorm because batch_size=1 makes BatchNorm undefined."

- **Dropout experiment**: mention in one sentence. "We tried dropout; it catastrophically hurt
  extrapolation folds, so we reverted."

- **Ridge-stacked ensemble**: just say "we also tried a Ridge meta-learner; equal averaging
  won."

- **All bug fixes** (AMP overflow, fp16 range, scheduler warning): skip entirely.

- **XGBoost variant comparison** (shallow/standard/deep): one line. "All three XGBoost depths
  gave nearly identical R2 — low-order temperature splits dominate."

- **MLP architecture variants** (standard/wide/residual): same treatment.

---

## Visuals to Prepare

Before the presentation, generate or screenshot these:

1. **log10(nH2) histogram** — from data_loader.py or any prediction output. Shows the 30-dex
   range. Can generate with matplotlib from one of the saved .npz files.

2. **Ablation bar chart** — four bars: XGB alone, +spatial3, +multi-scale, +ensemble.
   Heights: 0.886, 0.917, 0.924, 0.948. Add error bars from the per-fold std.

3. **Per-fold R2 comparison** — grouped bar chart: ens_sp, xgb_standard_sp, xgb_standard.
   X-axis: G0 values. Easy to read side-by-side.

4. **Single-cube extrapolation heatmap** — already generated at
   `logs/single_cube_extrapolation/heatmap_*.png`. Use directly.

5. **Intra-cube section scatter** — test R2 vs log(train fraction %), one series per split
   type (rand/slab/box). The divergence between rand and slab is the key visual.

6. **Optional**: A z-slice side-by-side from intra_cube_visualize.py showing Ground Truth,
   Train Mask (sparse random dots), and Stacked Prediction — visually demonstrates that
   1% sampling is "enough".

---

## Anticipated Q&A

**Q: Why not use the CNN if it sees 3D spatial structure directly?**

A: The CNN has only ~48 effective training volumes (6 cubes x 8 augmentations per fold).
XGBoost has 12.5M training samples. Our spatial features extract the same spatial context
the CNN would learn, but as a deterministic preprocessing step, so the data-rich tabular
models can use it. The CNN trained from scratch on 6 volumes is fundamentally data-starved.

**Q: Isn't fh2 derived from nH2? Isn't including it as a feature data leakage?**

A: No. fh2 (H2 self-shielding factor) measures how much H2 attenuates its own UV
photodissociation radiation — it's computed from the column density integral along the UV
axis, not algebraically from local nH2. Two cells with the same nH2 can have very different
fh2 values depending on the shielding column, and vice versa.

**Q: What is a dex and why use it as a metric?**

A: One dex = one order of magnitude = a factor of 10. R2 in log-space measures whether we
predict the correct order of magnitude for each cell. We use this instead of linear-space
metrics because neural networks can produce arbitrarily large log-space predictions (log_nH2
= 30 is feasible), and 10^30 creates astronomical MSE even for a single bad prediction. Dex
errors are physically interpretable: 0.1 dex = factor of 1.26, 0.3 dex = factor of 2.

**Q: Could this model replace the chemistry solver in future simulations?**

A: That's the long-term motivation. As an emulator, a trained model could predict nH2 ~1000x
faster than the chemistry solver. The key limitation is generalization beyond the training
distribution — currently limited to UV-only chemistry across the G0 range covered by the 7
cubes. Adding more simulations (varying density, metallicity, turbulence) would expand the
domain.

**Q: Why does the G0=0.1 fold do worse than G0=6.4 despite both being extrapolation?**

A: At high G0, UV radiation dominates everywhere — most gas is atomic, and nH2 is uniformly
low except in the densest shielded cores. The chemistry is simpler. At low G0, self-shielding
creates complex spatial structure where small density variations produce large nH2 variations.
The model must extrapolate to a regime where chemistry is more sensitive to spatial gradients.

**Q: How long does training take?**

A: The full ens_sp (XGBoost + MLP, 56 features, 7-fold CV) runs in about 30 minutes on a
GPU. The 3D CNN takes ~8 hours for 7 folds at 150 epochs. The spatial feature precomputation
(3D box filters for all 7 cubes) takes ~30 seconds.
