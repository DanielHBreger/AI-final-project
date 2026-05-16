# Project Narrative: Predicting Molecular Hydrogen Density with Machine Learning

---

## The Problem I Set Out to Solve

Modern astrophysical simulations of the interstellar medium (ISM) need to track how hydrogen transitions between its atomic and molecular forms — a process governed by complex chemistry and radiation physics. The standard approach is to solve a time-dependent chemistry network at every cell of a 3D grid, simultaneously with radiation transport. This is accurate but enormously expensive: for a 128³ grid of ~2 million cells, this computation dominates the runtime of the entire simulation.

The question I asked was: **can a machine-learning model learn to predict the molecular hydrogen density at each cell — given only local physical properties — well enough to replace this computation?** And more importantly: can it do this for UV field strengths it has never been trained on?

---

## The Dataset

I worked with seven 3D simulation cubes from a chemistry suite called *UVonly*. Each cube is a 128×128×128 grid (about 2.1 million cells), and the seven cubes are identical except for the strength of the ambient UV radiation field, G0, which takes the values {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4} in standard Habing units — a 64-fold range. Together these give roughly 14.7 million data points.

Each cell has 18 raw physical fields. I reduced these to **15 input features**: the hydrogen density, temperature, ionized hydrogen density, a self-shielding factor (which encodes how UV radiation is attenuated in that direction), dust extinction, the UV field strength itself, three velocity components, and six magnetic field components. The target is the molecular hydrogen number density, which spans roughly 35 orders of magnitude across cells — so I worked entirely in log₁₀ space.

---

## How I Designed the Evaluation

The most important methodological decision was how to split the data for testing. The naive approach would be to randomly assign cells to train/test sets. But this would be meaningless: a model that has seen millions of cells from all seven cubes could trivially interpolate to the held-out cells. It wouldn't tell me whether the model actually learned the underlying chemistry or just memorized per-cube statistics.

Instead, I used **leave-one-G0-out cross-validation**: in each of the 7 folds, I withheld an entire simulation cube (one G0 value) and trained on the remaining six. This directly tests whether the model can generalize to a UV environment it has never encountered — which is the actual use case. If it can, it's evidence that the model learned real chemical relationships, not just interpolation artifacts.

---

## What I Tried

I approached this as a systematic comparison across model families, increasing in complexity:

**Starting point — linear regression.** This immediately failed (R² as low as −0.876 on some folds), confirming that the physics is genuinely nonlinear and that a real ML approach is needed.

**Gradient-boosted trees (XGBoost).** These are the standard benchmark for tabular data. My first XGBoost model, trained on the 15 raw features, achieved an average R² of 0.886 across the 7 folds. That's decent, but with high variance — the model struggled most on the boundary folds (G0=0.1 and G0=0.2), where it was extrapolating furthest from its training range.

**Neural networks (MLPs).** I tested fully-connected networks of different widths and depths, including a residual variant with skip connections. These achieved similar performance to XGBoost (~0.886) and also showed high variance on boundary folds. Neither additional width nor residual connections helped significantly.

**The spatial insight.** The chemistry that determines molecular hydrogen is not purely local — it depends on the column of gas between a cell and the UV source. Cells near shielded molecular cores behave differently from isolated exposed cells, even if their local density is similar. I realized that the pointwise models were missing this spatial context. Rather than relying on a neural network to learn it from scratch, I **precomputed multi-scale neighbourhood averages** of all 15 features using 3D box-filter kernels at three scales (3×3×3, 5×5×5, and 7×7×7 cells). This adds 45 features per cell (3 scales × 15 features), giving a 60-dimensional feature vector. The computation takes about 30 seconds and is done once. Adding just the smallest scale (3³) to XGBoost improved mean R² from 0.886 to 0.917 — the biggest single gain in the entire project. Adding the larger scales brought it to 0.924.

**Ensembling.** XGBoost and MLPs make different kinds of predictions: tree models produce piecewise-constant outputs (sharp transitions near decision boundaries) while neural networks produce smooth continuous predictions. I combined them with equal weights (0.5 × XGBoost + 0.5 × MLP). This complementarity pushed performance to **R² = 0.948 ± 0.024** — the best result overall.

**3D convolutional networks (U-Nets).** Given that the problem is spatial, I also tested 3D U-Net architectures, which were designed precisely to learn from volumetric data. Despite this apparent advantage, the CNN results were consistently worse (R² ≈ 0.80) and far more variable than the tabular models. The reason: I only had 6 training cubes per fold, versus ~12.5 million individual cells for tabular models. The CNN is data-hungry; the spatial-feature approach captures the same structural information with orders of magnitude less data.

---

## What I Learned Along the Way

Several findings surprised me and weren't planned in advance:

**Dropout destroys extrapolation.** I tried adding dropout regularization to the neural networks. This backfired catastrophically on the out-of-distribution folds — adding 10% dropout to the CNN collapsed the G0=6.4 fold from R² = +0.22 to R² = −0.61. With only 6 distinct G0 values in training, the model needs to form a coherent internal representation of how chemistry varies with UV strength. Randomly zeroing activations breaks this. Only L2 weight decay (a much weaker regularizer) was retained.

**Feature engineering beat architecture search.** I spent considerable effort tuning XGBoost depth (4/6/8 trees), MLP width and depth, and even explored learned meta-ensembling with a Ridge meta-learner. None of these produced measurable improvements (typically < 0.002 R²). The spatial box-filter features, by contrast, gave +0.031 R² in a single change.

**Target normalization is essential for CNNs.** The CNN initially produced R² < 0. A single change — standardizing the log(nH2) target per training fold — moved it to R² = 0.775. This is a practical lesson that matters specifically for volumetric regression with small sample counts.

**How far chemistry knowledge transfers.** In a side experiment, I trained on a single cube and predicted all seven — building a 7×7 transfer matrix. Performance degrades roughly as: 1 step in G0 → R²≈0.97; 2 steps → 0.96; 3 steps → 0.88; 4 steps → 0.75; all the way to the opposite extreme (6 steps) → ~0.15. This quantifies something that wasn't known: local chemical equilibrium is approximately universal within a ~2-4× change in G0, but outside that range, a new simulation or emulator evaluation is needed.

**Coverage matters more than volume.** In another side experiment, I asked: if you can only observe part of a volume, how should you sample it? Training on 1% of cells chosen randomly gave R²=0.897 on the remaining 99%. Training on 50% as a contiguous slab gave R²=−0.4 to −2.7 — worse than useless. A contiguous slab leaves the interior chemistry regime entirely unseen; random sampling keeps the feature distribution representative everywhere. The practical implication for observational astronomy: sparse uniform IFU grids reconstruct 3D density fields far better than deep contiguous maps.

---

## The Final Result

The best model — which I called `ens_sp` — is an equal-weight ensemble of XGBoost and a wide MLP, both trained on the 60-dimensional spatial feature vector. Its performance across all 7 leave-one-G0-out folds:

| Fold withheld | R² |
|---|---|
| G0 = 0.1 (hardest: furthest extrapolation) | 0.908 |
| G0 = 0.2 | 0.915 |
| G0 = 0.4 | 0.953 |
| G0 = 0.8 | 0.962 |
| G0 = 1.6 | 0.967 |
| G0 = 3.2 | 0.967 |
| G0 = 6.4 | 0.965 |
| **Mean** | **0.948 ± 0.024** |

Training takes minutes on a single GPU. Inference on a full 128³ cube takes seconds. The typical per-cell error is 0.1–0.2 dex (a factor of 1.3–1.6 in actual nH2), across a target range of 35 orders of magnitude.

---

## The Core Lesson

The thing I found most interesting about this project is that the two most impactful decisions were **not** about model architecture. They were:

1. **How to evaluate** — leave-one-G0-out instead of random split. Without this, any model looks like it achieves R² ≈ 0.99 by memorizing per-cube statistics, and the real generalization problem is invisible.

2. **How to represent space** — precomputing neighbourhood averages instead of using a spatial neural network. The spatial CNN had the "right" inductive bias on paper, but the tabular approach with engineered spatial features achieved 18% higher R² with a fraction of the compute and an order of magnitude less variance.

The interplay between evaluation design, feature engineering, and regularization choices — rather than network architecture — drove the results.
