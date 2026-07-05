# Predicting Molecular Hydrogen Density (nH2) in 3D Astrophysical Simulations

## Executive Summary

**Authors:** Daniel H. Breger
**Date:** April 2026
**Audience:** Senior faculty reviewers

---

## 1. Problem and Scientific Motivation

The molecular hydrogen number density (nH2) is a fundamental quantity in interstellar-medium (ISM) physics, as H2 is the direct precursor of star formation. Computing nH2 self-consistently in 3D requires coupling time-dependent non-equilibrium chemistry to radiative transfer — an expense that dominates the cost of modern ISM simulations. Any reliable surrogate that maps *local physical state* to nH2 at near-interactive speed is therefore of substantial scientific value.

We investigated the following question:

> *Given 15 local physical properties at each cell of a 128³ simulation grid, can a machine-learning surrogate predict nH2 accurately, and — crucially — can it generalize to UV-field strengths (G0) never seen during training?*

The target nH2 spans roughly **30 orders of magnitude**, from UV-exposed cells (~10⁻³⁰ cm⁻³) to shielded molecular cores (~10⁵ cm⁻³), which makes regression in linear space numerically intractable. All modelling is performed in log₁₀(nH2) space.

---

## 2. Dataset

Seven uniform-grid simulations from the *UVonly* chemistry suite, each a 128×128×128 cube containing 2,097,152 cells, for a total of **14,680,064 cells**. The simulations differ only in the ambient UV field strength G0 (Habing units), which takes the geometric sequence **{0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4}**, a 64× dynamic range.

Eighteen raw fields per cell are reduced to **15 retained features**: log-transformed densities and temperatures (`log_nH`, `log_T`, `log_nHp`), the self-shielding factor (`log_fh2`), extinction (`ext`), the global `log_G0`, the three-component gas velocity, and six face-centred magnetic-field components. Note: `fh2` is *not* algebraically derived from `nH2`; it encodes directional UV attenuation along the illumination axis and is therefore a legitimate and highly informative input.

---

## 3. Evaluation Protocol

The core methodological choice of the project is **leave-one-G0-out cross-validation**. Each fold withholds an entire cube (one G0 value, ~2.1M cells) and trains on the remaining six. This directly measures *out-of-distribution generalization across UV strength* — the real-world use-case — rather than the trivial in-distribution interpolation that random-split CV would measure.

All metrics are reported in **log₁₀(nH2) space (dex)**:
- Primary metric: R² on log₁₀(nH2).
- Secondary: RMSE and MAE in dex (1 dex ≈ a factor of 10 in nH2).
- A clipped linear-space R² is reported for compatibility.

An R² of 0.95 in log-space means typical per-cell errors of ~0.1–0.2 dex, i.e. a factor of 1.3–1.6 in nH2.

---

## 4. Models and Design Rationale

We evaluated a hierarchy of models chosen to stress-test whether additional capacity, spatial context, or ensembling improves generalization:

| Family | Variants | Purpose |
|---|---|---|
| Linear Regression | — | Lower bound; tests whether the problem is nonlinear. |
| XGBoost (gradient-boosted trees) | depth 4 / 6 / 8 | Strong tabular baseline; tests feature-interaction order. |
| Fully-connected MLP | standard / wide / residual | Tests whether neural capacity or skip connections help on tabular data. |
| 3D U-Net (volumetric CNN) | base-ch 16 / 32 / 64 | Tests whether learned spatial filters beat engineered spatial features. |
| Hybrid XGBoost-guided CNN | — | Injects OOF XGBoost prediction as a 16th CNN channel. |
| Equal-weight ensemble (`ens_sp`) | XGBoost + MLP with spatial features | Combines complementary inductive biases. |
| Ridge-stacked ensemble (`stacked_sp`) | — | Learned meta-combination. |

### 4.1 Multi-Scale Spatial Neighbourhood Features (key methodological contribution)

Pointwise models see each cell in isolation, but H2 chemistry is inherently spatial (self-shielding depends on neighbour column density). Rather than relying on a CNN to learn this from scratch, we **precompute 3D box-filter averages** of all 15 features at three scales (3³, 5³, 7³ kernels) using `scipy.ndimage.uniform_filter`, yielding 45 additional features. Concatenated with the 15 baseline features, each cell carries a **60-dimensional feature vector**.

This operation is deterministic, separable (O(N) per scale), computed once for all 7 cubes in ~30 s, and directly consumable by XGBoost and MLP.

### 4.2 Physics-Aware Data Augmentation

For the CNN, we derived the 48-element octahedral symmetry group Oₕ and restricted it to the **8 z-preserving operations** consistent with a fixed UV illumination direction. Scalar fields transform by index permutation, polar vector fields (velocity) rotate as `v' = R v`, and axial vector fields (magnetic field) transform as `b' = det(R)·R b` with face-centred left/right swaps on axis negation. Applying the naïve 48-element group would inject physically invalid training data and was verified to degrade results.

### 4.3 Regularization Finding: Dropout Is Harmful Here

A consistent and empirically reproducible result across CNN and tree models: **dropout (and DART for XGBoost) degrades out-of-distribution folds catastrophically.** Adding dropout=0.10/0.20 to the CNN collapsed the G0=6.4 fold from R²=+0.22 to R²=−0.61. With only six distinct G0 values in training, the model must form coherent representations of the UV dependence; random feature-zeroing breaks the extrapolation coherence needed at the G0 boundaries. Only L2 weight decay (1e-5) was retained.

---

## 5. Results

### 5.1 Final Per-Fold Results (leave-one-G0-out, log-space R²)

| Variant | Mean R² | Std | G0=0.1 | G0=0.2 | G0=0.4 | G0=0.8 | G0=1.6 | G0=3.2 | G0=6.4 |
|---|---|---|---|---|---|---|---|---|---|
| **ens_sp (XGBoost_sp + MLP_sp, 0.5/0.5)** | **0.9482** | **0.024** | 0.908 | 0.915 | 0.953 | 0.962 | 0.967 | 0.967 | 0.965 |
| stacked_sp (Ridge meta-learner) | 0.9456 | 0.020 | 0.912 | 0.916 | 0.954 | 0.960 | 0.962 | 0.961 | 0.954 |
| xgb_standard_sp | 0.9243 | 0.026 | 0.912 | 0.874 | 0.920 | 0.934 | 0.924 | 0.948 | 0.959 |
| mlp_wide_sp | 0.9127 | 0.034 | 0.843 | 0.886 | 0.914 | 0.928 | 0.946 | 0.945 | 0.927 |
| xgb_standard (no spatial) | 0.8864 | 0.061 | 0.891 | 0.751 | 0.866 | 0.902 | 0.910 | 0.932 | 0.952 |
| mlp_standard (no spatial) | 0.8862 | 0.046 | 0.809 | 0.825 | 0.883 | 0.921 | 0.929 | 0.928 | 0.910 |
| Best 3D U-Net (standalone) | 0.803 | 0.172 | 0.497 | — | — | — | — | — | 0.591 |
| Linear Regression | 0.230 | — | −0.876 | −0.237 | — | — | — | — | — |

### 5.2 Ablation of the Final Model

| Configuration | Features | Mean R² | Δ |
|---|---|---|---|
| XGBoost alone | 15 | 0.886 | — |
| + spatial 3³ | 30 | 0.917 | **+0.031** |
| + multi-scale 3³+5³+7³ | 60 | 0.924 | +0.007 |
| + MLP_sp equal-weight ensemble | 60 | **0.948** | +0.024 |

The largest single-step gain came from adding spatial neighbourhood means (+0.031). Neural architecture search (MLP width, residual blocks) and tree depth search produced negligible improvements (< 0.002).

### 5.3 Per-Fold Difficulty

The two boundary folds, **G0=0.1 (pure downward extrapolation)** and **G0=0.2 (near-boundary)**, are consistently the hardest, confirming the a-priori prediction. Interior folds (0.4–3.2) all exceed R²=0.95. The upper boundary (G0=6.4) is easier than the lower boundary, consistent with the physical picture that at high UV, photodissociation dominates uniformly and the chemistry is simpler.

### 5.4 The CNN Lesson

The 3D U-Net — despite having roughly the "right" inductive bias for a spatial problem — was consistently **worse** than pointwise models and significantly **more variable** (std up to 0.52). Reasons:

1. Only 6 training cubes × 8 augmentations = 48 effective volumetric samples, versus ~12.5M per-cell samples for tabular models.
2. Memory constraints force 64³ input (down-sampled from 128³), losing fine-scale gradients.
3. Instability on extrapolation folds — the very folds where the CNN's spatial advantage should matter most.

Including the CNN in any ensemble lowered mean R². The spatial-feature tabular approach dominates on every metric: accuracy, variance, runtime, and reproducibility.

---

## 6. Supplementary Experiments

### 6.1 Single-Cube Extrapolation (7×7 R² Matrix)

Training `stacked_sp` on a single G0 cube and predicting all 7 yields a clean quantitative picture of how far chemical knowledge transfers across UV strength:

| G0 distance from training cube | Typical R² |
|---|---|
| 1 step (e.g., 0.1→0.2) | ~0.97 |
| 2 steps | ~0.96 |
| 3 steps | ~0.88 |
| 4 steps | ~0.75 |
| 5 steps | ~0.45 |
| 6 steps (0.1→6.4 or vice-versa) | ~0.15 |

The local chemical equilibrium is approximately universal within ~2–4× changes in G0 but degrades rapidly beyond. This empirically justifies leave-one-G0-out CV and quantifies the information content of a single simulation.

### 6.2 Intra-Cube Spatial Section — An Observational-Design Result

Within a single cube, we asked: *if only part of the volume is observed, how well can the rest be reconstructed?* Three geometries were tested (random, contiguous slab, contiguous box) at varying fractions.

| Training geometry | Fraction | Test R² |
|---|---|---|
| Random sample | **1%** | **0.897** |
| Random sample | 5% | 0.952 |
| Random sample | 25% | 0.970 |
| Contiguous box | 25% | 0.823 |
| Contiguous **slab** (x/y/z half) | 50% | **−0.4 to −2.7** |

**The central finding:** 1% random coverage outperforms 50% contiguous-slab coverage by more than a full R²-unit. **Coverage uniformity matters more than coverage volume.** A slab leaves an entire spatial regime (e.g. the shielded interior) unseen, and the learned feature distribution does not extend there. Random sampling keeps every test cell statistically close to training data in all three dimensions.

**Practical consequence for observational astronomy:** surveys that target molecular-cloud sight-lines should prioritize *uniform spatial coverage* (e.g., sparse IFU-style grids) over *deep contiguous maps* when the goal is to reconstruct a 3D density field via ML.

---

## 7. Reproducibility and Code Artefacts

The full pipeline is deterministic and runs end-to-end on a single GPU:

| Component | File |
|---|---|
| Data loading, log-transforms, volume reshaping | `data_loader.py` |
| Octahedral symmetry augmentation | `augmentation.py` |
| Linear / XGBoost / MLP with `compute_metrics` | `classical_models.py` |
| 3D U-Net architecture | `cnn_model.py`, `train_cnn.py` |
| Shared utilities (FlexMLP, spatial-feature computation) | `model_helpers.py` |
| Architecture comparison driver (main experiments) | `compare_architectures.py` |
| Production ensemble trainer + `.npz` predictions | `predict_and_visualize.py` |
| Single-cube extrapolation (7×7 matrix) | `single_cube_extrapolation.py` |
| Intra-cube spatial section study | `intra_cube_section.py` |
| 3D volume viewer (PyVista, linked panels) | `load_and_compare.py` |
| 2D slice browser (matplotlib, slider + textbox) | `slice_compare.py` |

All experiment runs are logged to timestamped JSON (`results/arch_comparison_*.json`, `results/cnn_training_*.json`, `logs/**`). Predictions are stored as per-fold `.npz` files with embedded R² metadata.

---

## 8. Key Takeaways

1. **Feature engineering beat architecture search.** The single largest gain (+0.031 R²) came from adding three scales of 3D box-filter averages to the feature set. Neither MLP width/depth nor XGBoost depth produced measurable improvements.
2. **Pointwise model + engineered spatial features > 3D CNN.** The tabular approach achieved R²=0.948 with ~20× less training cost and an order of magnitude less variance than the best CNN (R²=0.80, std=0.17).
3. **Simple averaging beat learned stacking.** Equal-weight ensembling of XGBoost and MLP outperformed a Ridge meta-learner; with only 7 folds, learned combination weights overfit the meta-training distribution.
4. **Dropout is harmful for out-of-distribution extrapolation** when training signal comes from only a handful of parameter values. L2 weight decay is preferable.
5. **Target normalization is essential for the CNN.** Standardizing log(nH2) per fold moved CNN results from R² < 0 to R² = 0.775 in a single change.
6. **Leave-one-G0-out CV is the correct evaluation protocol.** Random-split CV would produce R² ≈ 0.99 by letting the model memorize per-cube statistics, masking the real generalization problem.
7. **Observational implication.** Sparse uniform sampling reconstructs 3D H2 density fields far better than deep contiguous sampling — an actionable guideline for IFU survey design.

---

## 9. Final Model Summary

**ens_sp** — 60-feature equal-weight XGBoost + MLP ensemble

- **Input:** 15 physical features + 45 multi-scale (3³, 5³, 7³) box-filter means
- **XGBoost:** depth=6, 400 trees, lr=0.1, subsample=0.3, tree_method="hist" (GPU)
- **MLP:** [512, 512, 256, 128], BatchNorm+ReLU, 100 epochs, Adam, CosineAnnealingLR, weight_decay=1e-5
- **Ensemble:** `y = 0.5·y_xgb + 0.5·y_mlp`
- **Target:** log₁₀(nH2)

**Performance (7-fold leave-one-G0-out):** R² = **0.948 ± 0.024** in log-space, with per-fold R² ∈ [0.908, 0.967]. Training completes in minutes on a single consumer GPU; inference on a full 128³ cube takes seconds.
