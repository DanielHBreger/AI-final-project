# Design decisions & audit log

Written 2026-07-04, after a full code review (all scripts) and a
design-decision audit performed while runs 3–4 trained. This is the
canonical record of every non-obvious scientific/engineering decision in
the pipeline, its assessment, and how it is addressed. Cross-references:
runs live in `RUN_PLAN.md`; paper-text actions live in
`paper/PAPER_UPDATE_INSTRUCTIONS.md` ("Audit-mandated disclosures").

Each item: **Decision** → **Assessment** → **Disposition** (one of:
KEEP — sound, no action; DISCLOSE — keep, but the paper must state it;
VERIFY — factual check needed before the wording is finalised;
OPTIONAL RUN — worth new compute if time allows; FUTURE — post-paper).

---

## 1. Prediction clipping

### 1.1 Metric-level clip (truth range ± 1 dex)
**Decision:** `compute_metrics` (and `statistical_analysis`,
`merit_metrics` since 2026-07-04) clips predictions to
`[true_min − 1, true_max + 1]` before exponentiating for `R2_lin` and
`mass_ratio`.

**Assessment:** For `R2_lin` it is a necessary numerical guard (one wild
voxel destroys linear-space R²). For `mass_ratio` it is scientifically
questionable: (i) it uses the held-out cube's truth range — an oracle
unavailable in deployment, so the reported mass is not what a user would
get; (ii) it understates the failure of models that over-predict beyond
truth + 1 dex — exactly the U-Net's OOD mode (the ×172 at G0=6.4 is a
lower bound). Mitigations: for the classical models the clip essentially
never binds (bias ≤ 0.3 dex, scatter ≈ 0.25), and it only softens the
model the paper argues *against* — conservative for the headline claim.
Runs 1b/2b/3 recorded clipped values, and `test_cnn` saves neither
volumes nor weights, so unclipped CNN numbers are unrecoverable without
a rerun.

**Disposition:** KEEP frozen for this paper + DISCLOSE (state the clip in
the metrics section; call the U-Net's extrapolation-fold mass ratios
lower bounds). FUTURE: switch the clip bound to the *training*-target
range (removes the oracle, deployable) and record `mass_ratio_raw`
alongside — definition change, requires rerunning everything quoted.

### 1.2 Model-level clip (training range ± 2 dex, MLP only)
**Decision:** MLP predictions are clipped to the training-target range
± 2 dex (`run_mlp_cv`, `_predict_mlp`, `_train_mlp`).

**Assessment:** Justified. Training-information only (deployable);
encodes the prior that a surrogate should not invent densities outside
training support; XGBoost has the property inherently (leaf values are
bounded by training targets). The ± 2 dex margin is arbitrary but
permissive. Asymmetry: the U-Net is the only unbounded model.

**Disposition:** KEEP + DISCLOSE (describe as part of the model
definition; note the U-Net is unbounded).

---

## 2. Data & features

### 2.1 Target dynamic range — "~30 orders of magnitude" is stale
**Decision/claim:** older docs (CLAUDE.md; possibly paper.tex) say the
target spans ~30 orders of magnitude.

**Assessment:** Verified against the raw CSVs 2026-07-04: there are NO
exact nH2 zeros; range is 4.7×10⁻¹² – 9.2×10³ (checked at both G0
extremes) → **~16 dex**. The `_EPS = 1e-30` guards never bind (min
nonzero fh2 ≈ 8.5×10⁻²⁵). The "+eps" vs "clip(lower=eps)" inconsistency
between columns is cosmetic.

**Disposition:** VERIFY nothing further; fix the number wherever quoted
(CLAUDE.md fixed 2026-07-04; paper must say ~16 dex). KEEP `_EPS` as a
harmless guard.

### 2.2 f_H2 (self-shielding factor) as a feature
**Decision:** `log_fh2` is a model input; code comment claims it is "an
independent physical quantity — not algebraically derived from nH2".

**Assessment:** Too strong. Self-shielding factors are computed from the
H2 *column density* — an integral of the target field. Not row-wise
leakage, but it imports target information through the simulation's own
solver. The Table 2 ablation (runs 2/2b) is exactly the right defense
and quantifies the dependence (R2_mol 0.991 → 0.840 without it).

**Disposition:** DISCLOSE — the paper must call f_H2 "a solver-internal
quantity, available only where the host code already tracks H2", present
the no-f_H2 configuration as the deployable one, and never claim
independence. (Code comment in `data_loader.py` should be softened at
the next touch; not urgent.)

### 2.3 log_G0 as a feature
**Decision:** constant-per-cube UV strength as input.

**Assessment:** Justified — the only channel through which models
distinguish regimes; makes leave-one-G0-out a genuine covariate-shift
test. Side effect worth a sentence: trees cannot extrapolate in this
coordinate (splits saturate at the training range), partially explaining
XGBoost's edge behaviour; the MLP/stack can extrapolate the trend.

**Disposition:** KEEP + DISCLOSE (one sentence on the tree-extrapolation
point in §5).

### 2.4 Velocity + B-field features
**Decision:** vx/vy/vz + 6 face-centred B components included "because
available"; weak direct physical prior for equilibrium chemistry.

**Assessment:** Harmless; feature-importance figure will show their
weight. Run 8 (`--no-vel --no-B`) is the designed test and doubles as
the "minimal deployable feature set" result.

**Disposition:** OPTIONAL RUN (run 8, already in RUN_PLAN) — until it is
run, the paper should not imply physical necessity.

### 2.5 Spatial features: kernels (3,5,7), means only
**Decision:** multi-scale neighbourhood means via `uniform_filter`.

**Assessment:** Kernel choice arbitrary but spans small/medium/large;
multi-scale beat single-scale in earlier runs. Means-only (no std/
gradients) is a scope decision.

**Disposition:** KEEP; future-work sentence at most.

---

## 3. CV & data design

### 3.1 Leave-one-G0-out
**Assessment:** The right protocol for the OOD claim; the interior/edge
dichotomy (run 3's headline) is its natural structure. KEEP.

### 3.2 One realization per G0; shared initial conditions
**Assessment:** Fold-to-fold differences conflate UV-strength difficulty
with realization noise (n = 7); all cubes share initial conditions, so
the OOD claim is specifically about G0, not unseen clouds. §2.1 wording
was already corrected (cubes are NOT identical in nH/T/v — they evolved
apart).

**Disposition:** DISCLOSE (scope the claims to G0 generalisation) +
OPTIONAL RUN (run 13 in RUN_PLAN: a different-seed test cube — the
single highest-value robustness addition; blocked on data from the
simulation collaborators).

---

## 4. Training decisions

### 4.1 Density weighting (1× at p99 → α=100 at p99.99, exponential)
**Assessment:** Physically motivated (H2 mass lives in the dense tail)
but three arbitrary hyperparameters with no recorded sensitivity test —
and the weighting now sits inside the headline model
(`stacked_weighted_mwcal`).

**Disposition:** OPTIONAL RUN (run 12 in RUN_PLAN: XGB-only α sweep;
needs a small code change to expose α) + DISCLOSE the chosen values.

### 4.2 Ridge(α=1.0) meta-learner
**Assessment:** 2 meta-features → insensitive to α; intercept absorbs the
pooled OOF bias (interacts with recalibration by design, documented).
KEEP.

### 4.3 G0-linear bias recalibration (mean and mass-weighted)
**Assessment:** 6 points, assumes a smooth monotone trend; run 1b
confirms it for the stack; run 3 shows it structurally cannot rescue the
U-Net (phase-localised, non-linear failure) — which preempts the
fairness question. The mass-vs-mean comparison is the paper's
methodological contribution (smoke test 10 proves the identity). KEEP;
the fairness argument goes in the paper (finding 10).

### 4.4 Fixed-epoch training for MLP/XGB (no early stopping, no selection)
**Assessment:** Clean protocol — no selection of any kind. KEEP.

### 4.5 CNN best-epoch checkpoint selected on the held-out cube
**Assessment:** Test-based model selection, optimistic for the U-Net.
Quantified from run-3 histories: best epoch 61–128/150, final/best
val-MSE ratio 1.01–1.14 → ≤ 7 % RMSE advantage. Conservative for the
paper's claim (the favoured baseline still loses OOD).

**Disposition (superseded):** KEEP for this paper + DISCLOSE with the
quantified bound ("upper bound for the U-Net"). FUTURE: report
final-epoch metrics or select on a training-cube holdout.
**REVISED 2026-07-05:** upgraded to FIX before submission — both
external reviews (paper/REVIEW3_DISPOSITIONS.md #4) independently flag
test-set checkpoint selection as unacceptable for the primary benchmark
table even with the bound disclosed. Now RUN_PLAN run 16:
inner-validation-cube selection (+ record final-epoch metrics), rerun
the run-3 (and run-4) configs, then replace the U-Net rows in
Tables 2--3 and the §5.4 numbers.
**FIXED 2026-07-06/07 (runs 16b + 16c, central inner-val rule):** all
U-Net numbers in both papers are now leakage-free (15-input
0.963 ± 0.047, RMSE 0.348; 11-input 0.851 ± 0.121). The ≤12 % disclosure
is replaced by the inner-val protocol description plus a rule-sensitivity
disclosure (nearest-rule 0.914 vs central 0.963; the nearest rule turns
edge folds into two-step extrapolation — RUN_PLAN run 16 entry).
Selection vs final epoch is a no-op (0.9627 vs 0.9625): the old
test-based selection carried the U-Net's apparent edge advantage, but
the interior-champion result survives (interior R² 0.992 vs stack
0.984).

### 4.6 CNN trained unweighted (vs density-weighted stack)
**Assessment:** The mass-budget comparison has an asymmetry: the headline
stack is mass-aware by training, the U-Net is cell-mean-trained. A
weighted U-Net would be the fully fair mass comparison; the
interior/edge structural argument does not depend on it.

**Disposition:** DISCLOSE as a limitation; FUTURE (run 14 candidate,
post-paper).

### 4.7 InstanceNorm in the U-Net
**Assessment:** Each test volume is normalised by its own per-channel
statistics — per-cube test-time input adaptation the pointwise models
don't get. Inputs only, so not leakage; together with 4.5 it makes the
CNN comparison generous, which strengthens the conclusion.

**Disposition:** KEEP + DISCLOSE (footnote).

---

## 5. Boundary conditions & augmentation

### 5.1 Boundary treatment: `reflect` filters, zero-padded convolutions
**Decision:** `uniform_filter(mode='reflect')` for spatial features;
Conv3d zero-padding in the U-Net.

**Assessment:** Turbulence boxes are typically *periodic*; if these are,
`wrap`/circular padding would be physically correct. ~13 % of cells sit
within 3 cells of a face (contaminated k=7 means). Changing now would
invalidate every recorded number for a likely-marginal effect.

**Disposition:** VERIFY the simulations' actual BCs with the
collaborators → then DISCLOSE as an approximation (or state it is
correct). OPTIONAL RUN (run 15 in RUN_PLAN: one-model `mode='wrap'`
robustness check; needs a small code flag).

### 5.2 Augmentation group: 8 z-preserving ops (C4v)
**Assessment:** Chosen on the assumption UV enters along z. If the field
is isotropic the restriction is merely conservative (fine); if
directional, it is required. Either way the choice is safe; it must just
be stated correctly.

**Disposition:** VERIFY UV geometry with the collaborators → DISCLOSE.
If boxes are periodic, translations would be valid, cheap augmentation
the CNN never got — FUTURE note, not an error.

### 5.3 Held-out cube never augmented; no test-time augmentation
**Assessment:** Correct and conservative. KEEP.

---

## 6. Pipelines & reproducibility

### 6.1 Two stacking protocols
**Assessment:** `compare_architectures` reuses the outer CV's OOF
predictions (meta-training rows produced by base models that saw the
held-out cube — standard shortcut, tiny indirect effect with a
2-coefficient Ridge); `predict_and_visualize` is fully nested. Numbers
from the two pipelines are close but not identical. Documented in both
files 2026-07-04.

**Disposition:** DISCLOSE — describe the nested protocol in Methods; do
not claim "fully nested" for Table 1's stacked rows.

### 6.2 Seed inconsistency
**Assessment:** Table-1 MLPs use GLOBAL_SEED+fold; §5.4's `_train_mlp`
uses seed 0. Harmless, deterministic each, but §5.4 MLPs are not
seed-identical to Table 1 MLPs.

**Disposition:** DISCLOSE only if per-model §5.4 numbers are quoted next
to Table 1; otherwise ignore.

### 6.3 `test_cnn` saves neither volumes nor weights
**Assessment:** No post-hoc CNN analysis possible (phase decomposition of
the mass failure, unclipped mass, slices) without retraining. Fine for
this paper (only metrics are quoted).

**Disposition:** FUTURE — add `--save-preds` if the CNN ever becomes more
than a baseline.

### 6.4 intra_cube zero-masking; 1-based `ix<64` split
**Assessment:** Masking test cells out of the spatial-feature volumes is
over-conservative (boundary training cells also degraded; log-space
zeros injected) — biases §6.2 *downward*, not a leak. The 1-based index
makes "halves" 63:65 — cosmetic.

**Disposition:** KEEP + one DISCLOSE clause in §6.2 ("conservative
masking").

### 6.5 Removed no-op `eval_set` (2026-07-04)
**Assessment:** XGBoost `fit(eval_set=...)` without early stopping never
affected the fitted trees; removed from all three call sites for
code-release hygiene. Fitted models are unchanged (rerun bit-match
expectations unaffected).

---

## 7. Metrics

### 7.1 PHASE_SPLIT = −4.0
**Assessment:** Presumably the minimum between the bimodal log-nH2
populations, but nothing in code or notes justifies it; sensitivity
unexamined.

**Disposition:** DISCLOSE — one sentence tying the value to the Fig 1
histograms (verify it lands in the density minimum when writing that
sentence).

### 7.2 Per-fold R², averaged
**Assessment:** Documented pathology (R² normalises by each fold's
variance; rankings can invert vs pooled MSE — finding 8 has a concrete
example from the table). Kept for literature comparability with
RMSE + skill as primary. KEEP + DISCLOSE (already in the metric
hierarchy).

### 7.3 mass_ratio = Σ10^pred / Σ10^true
**Assessment:** Equals the H2 mass ratio only on a uniform grid with
equal cell volumes and identical box sizes across G0 — presumably true.

**Disposition:** VERIFY box metadata once (with C3 physical-units work,
same source) → state the assumption.

### 7.4 MAE_mw weighted by TRUE nH2; skill vs pointwise XGB
**Assessment:** Truth-weighting is correct (predicted weights would be
gameable); the skill reference is a meaningful strong-simple baseline.
KEEP.

---

## Summary of open actions

| # | Item | Type | Where |
|---|------|------|-------|
| 1 | Clip disclosure + "CNN mass ratios are lower bounds" | paper text | PAPER_UPDATE_INSTRUCTIONS §Audit |
| 2 | f_H2 wording (solver-internal, not independent) | paper text | same |
| 3 | ~16 dex, not ~30 | paper text (CLAUDE.md done) | same |
| 4 | Tree-vs-MLP G0-extrapolation sentence | paper text | same |
| 5 | Checkpoint disclosure with ≤7 % bound; InstanceNorm footnote; unbounded U-Net | paper text | same |
| 6 | Unweighted-CNN limitation sentence | paper text | same |
| 7 | Nested-vs-shortcut stacking wording | paper text | same |
| 8 | PHASE_SPLIT justification vs Fig 1 | paper text (+quick check) | same |
| 9 | Scope claims to G0-generalisation (one realization, shared ICs) | paper text | same |
| 10 | Verify sim BCs + UV geometry (+ box metadata w/ C3) | ask collaborators | RUN_PLAN "Verification" |
| 11 | Density-weight α sensitivity sweep | optional run 12 | RUN_PLAN |
| 12 | Different-seed test cube | optional run 13 (data-blocked) | RUN_PLAN |
| 13 | Weighted U-Net | future (run 14 candidate) | RUN_PLAN |
| 14 | `mode='wrap'` robustness check | optional run 15 | RUN_PLAN |
| 15 | Training-range clip + mass_ratio_raw; final-epoch CNN option; --save-preds | future code revision | this file |
| 16 | Leakage-free U-Net checkpoints (disposition change of §4.5, 2026-07-05) | **DONE 2026-07-06/07** (runs 16b+16c, central rule; papers updated) | RUN_PLAN + REVIEW3_DISPOSITIONS #4 |
| 17 | Mass-cal functional stated exactly in §4.5 (mass-weighted residual ≠ exact log-mass-ratio closure; kept, validated OOS at 0.985–1.090; optional check A4) | paper text DONE 2026-07-05 | REVIEW3_DISPOSITIONS #6 |
| 18 | Deployed nested rows added to Tables 2–3 below a rule, excluded from bolding; shortcut comparison kept in main text (uniform protocol; nesting all 17 variants ≈ 6× compute for no inferential gain) | paper DONE 2026-07-05 | REVIEW3_DISPOSITIONS #3/#5 |
