# Scientific Merit Review (Reviewer 2, second round)

**Focus:** significance, validity, and astrophysical meaning of the findings —
not editorial/technical issues (covered in `REVIEWER2_REPORT.md`).
**Date:** 2026-07-02. Two new analyses were performed on your archived
prediction volumes for this review (`merit_metrics.py`, `check_fields.py`);
their outputs are quoted below and are reproducible.

---

## Verdict

The paper is a competent, honest, internally consistent **methodology case
study**. Its strongest scientific asset is the evaluation philosophy; its
weakest point is that the headline metric (log-space R²) does not measure the
quantity astrophysics actually consumes, and on that quantity the model
currently fails: **the saved final-model predictions over-estimate the total
H₂ mass of every held-out cube by 63–85 per cent.** This is fixable (it is a
calibration problem, not a ranking problem), but it must be fixed and
reported before the paper can claim a usable surrogate. Below, in decreasing
order of scientific weight.

---

## 1. New result of this review: the surrogate's H₂ mass budget is wrong

From the saved 128³ prediction volumes of the Section 5.4 run
(`predictions/pred_g0_*_20260312_*.npz`), comparing predicted and true
total H₂ mass per held-out cube, plus phase-conditional accuracy
(molecular = true log n(H₂) > −4):

| G0 | M_pred/M_true | R² (molecular) | RMSE mol [dex] | bias mol [dex] | R² (diffuse) | mass-weighted MAE [dex] | molecular cell fraction |
|----|---------------|----------------|----------------|----------------|--------------|------------------------|--------------------------|
| 0.1 | **1.63** | 0.945 | 0.24 | +0.19 | 0.79 | 0.21 | 67.7% |
| 0.2 | **1.66** | 0.956 | 0.22 | +0.20 | 0.84 | 0.23 | 44.2% |
| 0.4 | **1.67** | 0.960 | 0.23 | +0.20 | 0.82 | 0.22 | 31.3% |
| 0.8 | **1.85** | 0.953 | 0.25 | +0.22 | **0.60** | 0.26 | 16.7% |
| 1.6 | **1.73** | 0.968 | 0.22 | +0.19 | 0.88 | 0.23 | 8.0% |
| 3.2 | **1.76** | 0.964 | 0.24 | +0.21 | 0.92 | 0.25 | 3.7% |
| 6.4 | **1.83** | 0.968 | 0.24 | +0.21 | 0.85 | 0.25 | 0.8% |

Interpretation:

- The +0.2 dex bias the paper honestly reports in Section 5.4 is not a
  cosmetic residual-plot feature: it sits in the *mass-carrying* molecular
  phase and compounds exponentially, producing a systematic ×1.6–1.85 error in
  total H₂ mass — the number that feeds X_CO calibrations, molecular mass
  functions, and star-formation laws. R² = 0.97–0.99 conceals this entirely
  because R² is insensitive to a uniform offset relative to a 16-dex spread.
- The consistency of the bias across folds (all positive, +0.19 to +0.22 in
  the molecular phase) is actually good news: a single affine recalibration
  (report item D3) should remove most of it. But the paper cannot go out
  claiming a deployable surrogate while its mass budget is off by ~70 per cent
  and the text calls this "a property of this prediction run".
- **Caveat:** these volumes come from the 30-epoch analysis run, not the
  100-epoch headline configuration. That makes report item D2 (regenerate the
  volumes with the full schedule) scientifically urgent, not cosmetic: either
  the bias shrinks (good — say so) or it persists (then recalibrate and show
  the corrected mass ratios). Either way, add a mass-budget column/panel.

**Required for publication, in my view:** a "total H₂ mass per cube,
predicted vs true" panel or table, mass-weighted error statistics, and either
a recalibration step or a demonstration that the full-schedule model does not
carry the bias.

---

## 2. What R² = 0.988 actually measures on a bimodal 16-dex target

Most of the target variance is the separation between the UV-exposed
(~10⁻¹⁰ cm⁻³) and shielded (~1 cm⁻³) phases. A model earns most of the
available R² by placing each cell in the right phase; precision *within* the
molecular phase — the astrophysically relevant part — is a second-order
contribution. The numbers above quantify this: within the molecular phase R²
is 0.945–0.968 with ~0.23 dex scatter, and within the diffuse phase as low as
0.60. Neither number is 0.988. The paper's own Fig. 4(a,d) gestures at this
(R² falls toward the dense tail) but frames it as a metric artefact
("R² is a harsh statistic") rather than as the correct reading: **headline R²
overstates the precision of the interesting gas.** Recommend reporting
phase-conditional metrics in Table 1 or the error-analysis section, and
letting the abstract quote molecular-phase RMSE (~0.23 dex) alongside global
R². This is more honest and, frankly, still a good result.

---

## 3. The one-realisation design: what leave-one-G0-out does and does not test

A verification performed for this review sharpens Section 2.1: the evolved
cubes are **not** identical in their hydrodynamic fields (median T rises from
~1.0×10³ K at G0 = 0.1 to ~1.1×10⁴ K at G0 = 6.4; median n_H from 0.21 to
1.09 cm⁻³; velocity fields differ). The paper's original claim that the runs
"share the same … density field" was wrong and has been corrected in the
manuscript. Two consequences:

- *In the paper's favour:* the held-out cube's input fields are genuinely
  unseen, so leave-one-G0-out is a slightly stronger test than the old text
  implied — the model must cope with a shifted density–temperature
  distribution, not just a new G0 label.
- *Against over-reading:* the large-scale structure is still one realisation.
  The learned mapping may exploit realisation-specific correlations — most
  suspiciously through the velocity and magnetic-field features, which have
  no causal role in equilibrium UV-driven H₂ chemistry. If the model uses
  them, it is using them as fingerprints of the local dynamical state of
  *this* cloud. A `--no-vel --no-B` run (flags already exist) would show
  whether accuracy survives on causally defensible inputs; if it does, the
  paper's story tightens, and if it does not, that is a finding about
  realisation leakage that must be disclosed.

**The single most valuable experiment this paper could add** is one
simulation with a different turbulence seed (even at a single interior G0),
used purely as a test set. It converts every claim from "internal to one
cloud" to "demonstrated transfer", and would move the paper from a
methodology note to a citable surrogate. If no such simulation can be
obtained, the title/abstract should not resist the limitation — they
currently don't overclaim badly, but "surrogates for H₂ density in ISM
simulations" reads more general than "in seven re-illuminations of one
cloud".

---

## 4. The motivation is written for a problem the surrogate does not solve

The Introduction motivates via the cost of stiff *time-dependent* chemistry
inside dynamical simulations. But the surrogate maps local+neighbourhood
state → *quasi-equilibrium* H₂ density. In a live hydrodynamical run the
chemistry is out of equilibrium (H₂ formation timescales are long compared
with turbulent times at low density — the eq-vs-noneq comparison PDFs
shipped alongside your own simulation data suggest the simulators study
exactly this). The legitimate use cases are: post-processing snapshots for
synthetic observations, filling the G0 parameter axis without re-running
chemistry, initialising chemical fields, and the hybrid scheme of point 6
below. Branca & Pallottini's time-stepping emulator addresses the live-run
problem; this paper addresses the equilibrium-map problem. Say so in the
Introduction, or a referee who works on non-equilibrium chemistry will say it
for you, less kindly.

Related: **the speed-up is never quantified.** "Predicts 2,097,152 cells in
seconds" is only meaningful against the solver's cost for the same cube,
which the paper never states. Get the CPU-hours per cube from the simulators
and print the ratio. If the equilibrium solve is cheap (equilibrium chemistry
sometimes is), the honest pitch shifts further toward the
G0-interpolation/parameter-coverage use case.

---

## 5. The G0 = 0.8 anomaly deserves a physical sentence, and you already have the pieces

The interior fold G0 = 0.8 is the worst fold for the MLP and both ensembles,
carries the largest bias (+0.5 dex), and — from this review's analysis — has
by far the lowest diffuse-phase R² (0.60 vs 0.79–0.92 elsewhere). It is also
the cube where the atomic-to-molecular transition sweeps the largest volume
fraction (molecular fraction drops from 31 to 8 per cent between G0 = 0.4 and
1.6). Your own Fig. 4(c) shows errors peak at transition densities. The
likely story: at G0 ≈ 0.8 the largest share of cells sits on the steep part of
the transition, where equilibrium n(H₂) is exponentially sensitive to
shielding-column errors; interpolation difficulty here is physics, not an ML
pathology. One connecting sentence in Section 5.4 or the Discussion would
turn an unexplained blemish into an insight. (It also predicts where the
method will struggle in any new application: wherever the phase boundary
occupies the most volume.)

---

## 6. The buried headline: the within-cube experiment is a hybrid solver–surrogate scheme

Section 6.2 is framed defensively ("not a claim about observational
surveys"), which leaves it reading like a curiosity. Its real scientific
content is a deployment recipe: **run the expensive chemistry on a uniformly
random ~10 per cent of cells, train, and emulate the remaining 90 per cent at
R² ≈ 0.98** — an immediate ~10× reduction in solver cost within a single
snapshot, no cross-simulation generalisation required, and therefore immune
to the one-realisation caveat of point 3. This is arguably the most robust
and most usable result in the paper and it is presented as a supplementary
experiment. Recommend: promote it in the abstract/conclusions as a
solver-subsampling scheme, and note the practical corollary the data already
support (random beats contiguous because contiguous coverage misses physical
regimes — a genuinely useful warning for anyone subsampling simulations).

---

## 7. Mechanism test the ablation now begs for: directional column features

The completed Table 2 shows isotropic neighbourhood averages recover about
half of the f_H2 information. But f_H2 is a *z*-column quantity (illumination
is along +z); an isotropic box mean is the wrong symmetry. A cumulative sum
of density/extinction along z (a discrete column integral, O(N), one line
with `np.cumsum`) is the physically correct feature. If adding it closes most
of the remaining 0.977 → 0.988 gap, you will have demonstrated *mechanistic*
understanding — that the surrogate needs exactly the shielding column and
nothing else from the solver — which elevates the ablation from "graceful
degradation" to an explanation. If it does not close the gap, that is
interesting too (f_H2 carries non-column information, e.g. spectral
hardening). Cheapest high-value experiment available; strongly recommended.

---

## 8. Missing context: how much better than a formula is this?

Without any physics baseline, "0.977 without f_H2" floats free: nobody knows
whether a two-parameter Gnedin & Kravtsov-style prescription evaluated
cell-by-cell would score 0.5 or 0.95 on this task. If a fitted formula gets
close, the ML contribution shrinks to a convenience; if it fails badly (my
expectation, at cell scale with this dynamic range), the ML case becomes
compelling. Either outcome makes the paper stronger and the comparison is a
day's work. (This is report item D8 restated as a merit issue rather than a
formatting one: it is the difference between "accurate" and "accurate
*compared to what*".)

---

## 9. What genuinely holds up (credit where due)

- **The protocol reversal result** — random cell splits give R² > 0.99 and
  mean nothing; grouped splits reveal the real difficulty — is the paper's
  most transferable lesson, applies to half the ML-in-simulations literature,
  and is demonstrated rather than asserted.
- **The transfer matrix (Section 6.1)** is a real physical measurement: the
  equilibrium mapping is locally universal within a factor ~2–4 in G0 and
  decays smoothly beyond. That number (how far one simulated UV field
  carries) is usable by others planning simulation grids, independent of any
  ML details.
- **The f_H2 ablation, now complete,** answers the sceptic's first question
  quantitatively, and the spatial-features-recover-half-the-gap result is a
  clean, falsifiable piece of evidence for the shielding-column
  interpretation.
- **The U-Net instability finding** (mean R² varying 0.80–0.97 across
  configurations with 48 augmented volumes vs 1.5–23M parameters) is a useful
  negative result for the field, honestly framed as regime-specific rather
  than architectural doom.

---

## 10. Significance and venue, scientifically speaking

As it stands the contribution is: a careful OOD evaluation template + a
practical recipe (engineered neighbourhood features + stacking) + several
honest domain findings, on one private equilibrium suite along one physical
axis. That is a solid RASTI/MNRAS-methods paper. With three additions —
(a) mass-budget metrics + recalibration (point 1), (b) one unseen-realisation
test cube (point 3), and (c) the directional-column mechanism test
(point 7) — it becomes a strong MNRAS paper whose surrogate other groups
could actually adopt. I would prioritise in exactly that order; (a) is
mandatory, (b) is what referees will ask for, (c) is what makes the paper
memorable.

---

## Priority list (scientific, this round)

| # | Action | Cost | Effect |
|---|--------|------|--------|
| 1 | Mass-budget table/panel + mass-weighted metrics; recalibrate or show full-schedule run is unbiased (with D2/D3) — **fix implemented in code (see below), not yet run** | hours | removes a hidden ×1.7 systematic; mandatory |
| 2 | Phase-conditional metrics (molecular/diffuse) in results; quote molecular RMSE in abstract | hours | honest headline |
| 3 | One different-seed test simulation (ask simulators) | external | transforms scope of claims |
| 4 | z-column cumulative features vs f_H2 gap | ~3 h compute | mechanism demonstrated |
| 5 | `--no-vel --no-B` run | ~2.5 h compute | causal-input check / realisation-leakage probe |
| 6 | Quantify solver vs surrogate cost; rescope Introduction to equilibrium use cases | ask simulators | motivation matches capability |
| 7 | Reframe Section 6.2 as hybrid solver–surrogate subsampling; promote to conclusions | writing | strongest deployable result surfaces |
| 8 | Analytic-formula baseline (GK11-style) | ~1 day | "compared to what" answered |
| 9 | Physical sentence on the G0=0.8 anomaly (transition-volume argument) | minutes | anomaly → insight |

Manuscript changes already applied during this round: Section 2.1 corrected
(the evolved cubes do differ in T, density, and velocity — verified directly
from the data; medians quoted), and the corresponding limitation rephrased.
Analysis scripts `merit_metrics.py` and `check_fields.py` are in the repo
root for reproduction.

### Mass-budget fix: implemented in code (2026-07-02), awaiting a run

The recalibration is now wired into the pipeline; nothing has been executed
yet. Design: the Ridge meta-learner's intercept already zeroes the *pooled*
OOF residual, so a constant offset would be a no-op — the surviving bias is
G0-dependent. The fix computes the stacked model's mean OOF residual on each
*training* cube, fits it as a linear function of log10(G0)
(`fit_g0_bias_correction` / `predict_bias` in `model_helpers.py`), and
subtracts the fitted value at the held-out G0. Training-cube quantities only:
leakage-free.

Where it lives:

- `classical_models.compute_metrics` now also returns `bias` (mean residual,
  dex) and `mass_ratio` (predicted/true total H2 mass, clipped like R2_lin),
  so every pipeline and JSON log reports the mass budget automatically.
- `compare_architectures.run_stacked_ensemble_cv` reports each stacked
  ensemble twice: raw (`stacked_sp`) and recalibrated (`stacked_sp_cal`),
  with the per-fold offset printed; JSON summaries now aggregate all metric
  keys including `mass_ratio`.
- `predict_and_visualize.py` applies the correction by default
  (`--no-recalibrate` to disable); saved `.npz` files now contain the
  recalibrated `pred_vol`, the raw `pred_vol_raw`, the fitted
  slope/intercept/offset, the per-cube OOF biases, and raw+calibrated
  R²/mass-ratio. (The script's dead CNN code paths, which had left it
  syntactically broken, were repaired in passing.)

To produce the paper numbers: rerun `predict_and_visualize.py --all`
(full-schedule volumes + recalibration, addresses D2+D3 together) and rerun
`compare_architectures.py` to get the `stacked_sp_cal` row for Table 1/2.
