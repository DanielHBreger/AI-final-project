# A Multi-Scale Spatial-Feature Surrogate for nH2 in 3D ISM Simulations

## Summary for Domain Reviewers (ISM / Simulation / ML-for-Astrophysics)

**Author:** Daniel H. Breger
**Date:** April 2026

---

## 1. What we built and what it does

A machine-learning surrogate that maps the local physical state of a simulated ISM cell — density, temperature, ionization, extinction, self-shielding factor, magnetic field, velocity, and ambient UV — to its molecular hydrogen number density, log₁₀(nH2). The surrogate is trained and evaluated across **seven UV-only chemistry cubes** spanning G0 ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4} Habing (a 64× range), each a 128³ grid for a total of 14.68 M cells.

The central methodological question was not "can we fit an emulator?" — that is by now routine in the chemistry-emulator literature — but rather:

> *Can a surrogate generalize across UV-field strength, i.e., predict nH2 at a G0 never seen in training, given only the local state vector and cheap spatial context?*

This is the regime where most ML chemistry emulators silently fail, because random-split CV masks the out-of-distribution problem that dominates real deployment.

---

## 2. Framing against the ISM-ML literature

Two aspects distinguish this from a typical chemistry-emulator benchmark:

**(a) Leave-one-G0-out CV is the correct protocol for a UV parametric suite.** Random-split CV across cells would cheat: the model would always see cells from the same cube (hence same G0 and same turbulence realization) in training. Under that protocol a pointwise XGBoost reaches R² > 0.99 trivially. Leave-one-G0-out breaks this — each fold asks the model to extrapolate across a discrete UV axis. The two boundary folds (G0=0.1 and G0=6.4) are pure extrapolation in opposite photochemical regimes (formation-dominated and dissociation-dominated respectively), and they function as genuine stress tests rather than decorative hold-outs.

**(b) Spatial shielding is reintroduced as engineered features, not learned.** Chemistry emulators typically predict from the local state vector, which cannot capture self-shielding — a manifestly non-local effect. The common response is a 3D CNN. We argue, and show empirically, that on a UV-parametric suite this trades a tractable problem for an intractable one, and that **multi-scale box-filter means** of the input fields recover the dominant spatial signal at negligible cost.

---

## 3. Features and the role of `log_fh2`

The 15 retained per-cell features are:

- Scalars (log-transformed where appropriate): `log_nH`, `log_T`, `log_nHp`, `ext`, `log_fh2`, `log_G0`
- Polar vector: `vx, vy, vz`
- Axial vector (face-centred): `bxl, bxr, byl, byr, bzl, bzr`

A few domain-relevant points:

- **`log_fh2` is retained as a feature**, and we believe this is defensible. The H2 Draine self-shielding factor is *not* an algebraic function of the local nH2 — it integrates the H2 column along the UV illumination axis and therefore encodes the non-local directional shielding information that the local scalars cannot reconstruct. In feature-attribution diagnostics it ranks at or near the top, exactly as expected from the PDR-physics picture.
- **`log_G0` is a per-cube constant** and therefore carries no within-cube discriminative power. Its information enters implicitly through the cube-wise shift of the (nH, T, ext) → nH2 manifold. In leave-one-G0-out, this is precisely the nuisance structure the model must extrapolate across.
- **Magnetic-field components and velocities** rank consistently low in importance. The per-cell B-field and turbulent velocity do not directly set H2 equilibrium; their role is indirect, through the density/temperature structure they produce. This is a null result worth stating — it rules out the hypothesis that turbulent or magnetic forcing acts as a hidden variable in the equilibrium chemistry response.

### Multi-scale spatial neighbourhood features

For each of the 15 features we compute 3D box-filter means (`scipy.ndimage.uniform_filter`, separable, O(N)) at kernel sizes **3³, 5³, 7³** — yielding 45 additional features and a total of 60-D input per cell.

Physically, these capture three coupled length-scale regimes:

| Kernel | ⟨Nₙ⟩ averaged | Physical regime |
|---|---|---|
| 3³ | 27 | Immediate local gradient — relevant for formation rate modulation |
| 5³ | 125 | Cloud-scale coherence — approaches the size of individual clumps |
| 7³ | 343 | Meso-scale environment — the scale at which shielding becomes saturated |

The box-filter operation is isotropic and therefore *does not* encode the directional column structure that sets the true shielding integral. That information is still supplied via `log_fh2`. The engineered neighbourhood features supply the complementary *isotropic* density/extinction context that, combined with directional `fh2`, appears to span the spatial information the model needs. Ablation confirms this:

| Configuration | Mean R² | ΔR² |
|---|---|---|
| 15 local features only | 0.886 | — |
| + 3³ box-filter only | 0.917 | +0.031 |
| + multi-scale (3³+5³+7³) | 0.924 | +0.007 |
| + MLP_sp, equal-weight ensembled | **0.948** | +0.024 |

The multi-scale gain is modest in the mean but concentrated in the hardest fold (G0=0.2: 0.899 → 0.915), which is exactly where the extra meso-scale coherence should help — low-G0 cubes are dominated by self-shielded cores whose structure lives above the 3³ scale.

---

## 4. Model family and why the CNN lost

We evaluated linear regression, XGBoost at depths 4/6/8, an MLP at three sizes (plus a residual variant), a 3D U-Net at three widths, a hybrid XGBoost-guided CNN, and equal-weight / Ridge-stacked ensembles.

**Everything beyond "pointwise model on the right feature set" was surprisingly flat.** Tree depth, MLP width, and residual topology produced ΔR² < 0.002. The useful axes were (i) the feature set and (ii) the ensemble.

The 3D U-Net (standard config, 5.8 M params) reached R² ≈ 0.80 with std 0.17 — significantly *worse* and more variable than the tabular baseline. The diagnosis is straightforward:

- **Sample count mismatch.** The tabular models train on ~12.5 M per-cell samples per fold. The CNN trains on 6 cubes × 8 z-preserving symmetries = 48 effective volumes per fold. For a 23 M-parameter variant this is ~480 000 parameters per sample; catastrophic collapse on boundary folds is the observable signature.
- **Resolution compromise.** Memory forced a 128³ → 64³ average-pool of the input, erasing small-scale gradients that carry real information in the low-G0 cubes.
- **Extrapolation fragility.** The very folds where the CNN's spatial prior should matter most — G0=0.1 and G0=6.4 — are exactly where it failed hardest (R² as low as −0.61 with modest dropout).

Hybridizing the CNN with an OOF XGBoost prior injected as a 16th channel gave R²=0.62–0.81 depending on the seed, still not competitive. Including the CNN in any ensemble dragged the mean down.

The operational conclusion: on a suite of this size, spatial information is better served by cheap engineered context than by a learned volumetric encoder. A CNN may well be the right tool when the training set contains O(10²–10³) independent cubes; it is not here.

---

## 5. Regularization: dropout breaks extrapolation

A physics-adjacent finding worth flagging: **dropout consistently and severely degrades boundary-fold performance**, across both the CNN and the tree family (DART). CNN with dropout 0.10/0.20: mean R² 0.775 → 0.649, G0=6.4 fold R² +0.22 → −0.61.

Interpretation: with only 6 distinct G0 values in training, the network must form *coherent* representations of the UV-dependence axis in order to extrapolate. Dropout's random feature-zeroing is designed to break co-adaptation — which is precisely the co-adaptation that would let the model extend its (G0, local state) response surface beyond the training range. For interpolation folds, the degradation is mild; for extrapolation folds, it is structural.

We retained only L2 weight decay (1e-5). This generalizes beyond the present application: in ML surrogates trained on small parametric simulation suites, dropout should be treated as an OOD-destroying regularizer rather than a default.

---

## 6. Physics-consistent symmetry augmentation

For the CNN, we derived the full 48-element octahedral group Oₕ and restricted it to the **8 z-preserving operations** consistent with a fixed UV illumination direction. Field transformations are applied correctly per field type:

- Scalar fields (`nH, T, nHp, ext, fh2`, `G0`): index permutation / flip only.
- Polar vector (velocity): `v′ = R v`.
- Axial vector (magnetic field): `b′ = det(R) · R b`, with **face-centred left/right swap** on axis negation.

Using the full 48-op group injects physically inconsistent samples (the illumination direction is no longer fixed) and was verified to degrade performance. Omitting the axial-vector determinant factor on reflections silently mislabels B-field chirality. These are the kind of details that do not show up as explicit errors but do show up as inflated validation variance; the 8-op z-preserving set is the correct augmentation kernel for UV-anisotropic simulations.

---

## 7. Per-fold results and error structure

Final ensemble (`ens_sp`, XGBoost_sp ⊕ MLP_sp, equal weights) in log-space R²:

| Fold | R² | Regime |
|---|---|---|
| G0=0.1 | 0.908 | Pure extrapolation, formation-dominated |
| G0=0.2 | 0.915 | Near-boundary, steep fh2 gradient |
| G0=0.4 | 0.953 | Interpolation |
| G0=0.8 | 0.962 | Interpolation |
| G0=1.6 | 0.967 | Interpolation |
| G0=3.2 | 0.967 | Interpolation |
| G0=6.4 | 0.965 | Pure extrapolation, dissociation-dominated |
| **Mean** | **0.948** | std = 0.024 |

Three physics-relevant observations:

1. **The G0=0.1 / G0=6.4 asymmetry.** Both are extrapolation folds at equal parametric distance from their nearest training cube, but G0=6.4 is systematically easier. This is physically sensible: at high G0, UV saturates the photodissociation rate almost everywhere, so the (nH, T, ext) → nH2 response becomes a smoother, monotonic function dominated by a single process. At low G0 the response is governed by the onset of self-shielding, which is a threshold phenomenon sensitive to the spatial configuration of gas — harder to extrapolate to from the G0=0.2 anchor.
2. **The residual ~5% variance is concentrated at the molecular/atomic transition.** Per-cell error spectra show that the surrogate is essentially perfect in the fully molecular and fully atomic regimes and accumulates almost all of its error in the narrow transition layer where fh2 ∈ [0.01, 0.5]. This is where spatial context most strongly decouples from local state, and where a time-dependent formulation (not tested here — see limitations) would most likely differ from steady-state predictions.
3. **RMSE is ≈ 0.15 dex on interior folds**, corresponding to a ≈ 1.4× error in linear nH2 per cell. For subgrid or post-processing applications (e.g., generating synthetic H2 column maps), this is well within the scatter introduced by observational calibration.

---

## 8. Two supplementary experiments worth the attention of a domain reviewer

### 8.1 Single-cube extrapolation (7×7 R² matrix)

Training `stacked_sp` on one G0 cube and predicting all seven directly quantifies the transfer radius of a single simulation's chemistry:

| G0 distance (log steps) | Stacked R² |
|---|---|
| 0 (in-sample) | 0.994 |
| 1 | 0.976 |
| 2 | 0.958 |
| 3 | 0.876 |
| 4 | 0.745 |
| 5 | 0.446 |
| 6 | ~0.15 |

Physically: local equilibrium chemistry is approximately universal within a factor of 2–4 in G0 and breaks down by a factor of ~30×. This is the first quantitative estimate (that we're aware of) of the **parametric transfer radius** of a single UV-chemistry simulation under a tabular surrogate — a useful number when planning how densely a parameter suite must be sampled for emulation.

The matrix is not perfectly symmetric (formation-dominated cubes extrapolate outward slightly better than dissociation-dominated cubes extrapolate inward), consistent with the G0=0.1 vs G0=6.4 asymmetry above.

### 8.2 Intra-cube spatial sectioning — implication for IFU survey design

Within a single cube, we ask: given partial spatial coverage, how much of the volume can be reconstructed? Three geometries × multiple coverage fractions:

| Training geometry | Fraction | Test R² |
|---|---|---|
| Random voxels | **1%** | **0.897** |
| Random voxels | 5% | 0.952 |
| Random voxels | 25% | 0.970 |
| Contiguous axis-aligned box | 25% | 0.823 |
| Contiguous half-cube slab | 50% | **−0.4 to −2.7** |

**One per cent random coverage outperforms fifty per cent contiguous slab coverage by more than a full R² unit.** This is not primarily an ML result — it is a statement about the information geometry of molecular-cloud volumes. A slab leaves an entire phase of the cloud (e.g., the shielded interior) outside the training marginal, and the box-filter neighbourhood features degenerate there. Random voxel sampling keeps every test cell statistically close to training data in all three dimensions simultaneously.

For observational astronomy this has a concrete implication: **when the goal is to reconstruct a 3D H2 density field via ML from a sparse set of sight-lines, survey design should prioritize spatial coverage uniformity over depth.** Sparse IFU-style grids are recoverable; deep contiguous maps of one portion of a cloud are essentially non-recoverable outside that portion. The z-axis slab is slightly less catastrophic than x/y slabs, reflecting the milder density gradient along the UV-illumination axis — an artefact of the simulation setup rather than a generic observational guideline.

---

## 9. Honest limitations

A domain reviewer will rightly ask what this surrogate does *not* do, and where the published numbers should be read with care.

- **Single simulation code, single snapshot per G0.** We have not tested temporal dependence, turbulence-realization dependence, or code-to-code transfer. The surrogate's coefficients encode the chemical network, microphysics, and numerical integrator of the source simulation — deploying it on a different code's output without retraining is not supported by current evidence.
- **UV-only chemistry.** No cosmic rays, no metallicity variation, no non-solar dust-to-gas ratio. The relevant rate coefficients are implicit in the training data.
- **Steady-state implicit.** The snapshot data reflects whatever chemical age the underlying integrator had converged to. A time-dependent setting — star formation onset, radiation-field transients, cloud lifetimes shorter than H2 formation timescales — would require time as an explicit input and is out of scope here.
- **128³ resolution.** The surrogate has not been tested for resolution convergence. The box-filter kernel sizes (3, 5, 7 cells) would need to be rescaled for a different base resolution.
- **The model predicts nH2, not the full chemistry.** For multi-species predictions, each species would need its own surrogate and their combined use would not be guaranteed self-consistent.

None of these are criticisms of the present result; they are the scope conditions under which R²=0.948 should be read.

---

## 10. Practical deployment path

If this surrogate is to be used as more than a benchmark, the natural deployment paths are:

1. **Post-processing emulator for existing UV-parametric suites.** Drop-in replacement for the H2-chemistry step during analysis of already-completed simulations at novel G0. Inference on a 128³ cube is seconds on a consumer GPU.
2. **Sub-grid prescription in large-volume simulations.** Where resolving the H2 chemistry on-the-fly is prohibitive, the surrogate can supply nH2 conditional on coarse-grained local state — provided the large-volume simulation's state vector can be mapped onto the 15 inputs.
3. **Observational inference.** Combined with the intra-cube sectioning result, a sparse IFU-style observational coverage of a molecular cloud could plausibly be inverted to a 3D H2 density field via this class of surrogate. This is speculative and would require adapting the training set to match observational noise and projection effects.

---

## 11. What's reproducible

All code and experiment logs are in the project tree. The driver for the main architecture comparison is `compare_architectures.py`; the production ensemble is `predict_and_visualize.py`; the two supplementary experiments are `single_cube_extrapolation.py` and `intra_cube_section.py`. Every run emits a timestamped JSON log; predictions are saved as per-fold `.npz` volumes with R² metadata. Physics-aware augmentation and the octahedral group implementation are in `augmentation.py`. The full report — including derivations, bug-fix history, and the complete experimental iteration log — is `PROJECT_REPORT.md`.

---

## 12. Headline takeaways for a specialist reader

1. **Leave-one-G0-out is the right CV protocol for UV parametric suites**, and most published chemistry-emulator numbers (trained under random-split CV) are overestimates of real out-of-distribution accuracy.
2. **Engineered multi-scale neighbourhood features reproduce the dominant spatial signal** carried by a 3D CNN, at a fraction of the cost and with dramatically lower variance on extrapolation folds, *provided* a directional shielding feature (`log_fh2`) is retained.
3. **Dropout is a boundary-fold killer** on small parametric suites. This is not specific to this problem and should be tested before defaulting to dropout in any simulation-trained surrogate.
4. **Single-cube transfer degrades over ~4–6 geometric steps in G0**, which quantifies how densely a parametric suite must be sampled to support an emulator with uniform global accuracy.
5. **Uniform sparse spatial coverage beats deep contiguous coverage** for 3D field reconstruction — with a direct implication for IFU survey design of molecular clouds.

**Final model:** equal-weight ensemble of XGBoost (depth 6, 400 trees) and MLP ([512, 512, 256, 128]) on a 60-D feature vector (15 local + 45 multi-scale box-filter means). **R² = 0.948 ± 0.024** in log₁₀(nH2), with every fold ≥ 0.908 under leave-one-G0-out across the full 64× UV range.
