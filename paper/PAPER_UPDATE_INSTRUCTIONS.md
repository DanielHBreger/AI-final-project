# Instructions: updating or rewriting the paper with the new results

> **2026-07-05 addendum:** a dual-AI review pass was dispositioned in
> `REVIEW3_DISPOSITIONS.md` (same directory) — read it before further
> paper edits. Text fixes are already in paper.tex; open items: run 16
> (leakage-free U-Net checkpoints, blocking), runs 8/12/17, analysis
> items A1–A5 (RUN_PLAN). Bib TODO (verify on ADS before citing, like
> KMT09/Sternberg2014/Bialy2016): Palud et al. 2023, A&A (Meudon PDR
> neural emulation); Janssen, Branca & Buck 2026, A&A (surrogate
> benchmarking/model selection). Provenance deliverables for C1 now have
> concrete templates: a simulation-metadata table (code, box, resolution,
> BCs, driving, B-field, metallicity, dust, radiation geometry, network,
> heating/cooling, equilibrium criterion) and a per-feature provenance
> column for Table 1 (MHD primitive / chemistry output / column integral
> / imposed parameter / derived), plus an exact definition of `ext`
> (formula, direction, normalisation, boundary convention).

Audience: a future Claude Code session asked to take the rerun results and
figures (see `../docs/RUN_PLAN.md`) and either revise `paper/paper.tex` in place
or write a new version. Written 2026-07-03 by the session that did the
metrics/figure overhaul.

## What the paper is

`paper/paper.tex` — "Machine-learning surrogates for molecular hydrogen
density in three-dimensional ISM simulations using multi-scale spatial
features", single author (D. H. Breger), **RASTI format** (`rasti.cls` +
`rasti.bst` live in `paper/`; build with `latexmk -pdf paper.tex` from
`paper/`; the only acceptable warning is the benign `Hfootnote.1`).
It evaluates tabular ML surrogates (XGBoost, wide MLP, spatial box-filter
features, Ridge-stacked ensemble) against a 3D U-Net for predicting
log10(nH2) per cell, under leave-one-G0-out CV over 7 simulation cubes.

## Hard constraints (user requirements — do not violate)

1. **Cohesive single work.** No references to "early runs", "development
   runs", "corrected pipeline", "we later found". The paper describes one
   finished study. Bias/recalibration is presented as a *methodological
   audit step*, not as a fix applied after a mistake.
   This extends to method/evaluation versions: the paper must NOT narrate
   the project's evolution — no 64³-pooled vs native-128³ evaluation
   history (there is only ONE evaluation protocol in the paper: every model
   scored on the same native 128³ cells), no mean-recal "replaced by"
   mass-recal story (the two corrections are presented side by side as a
   designed comparison of which functional to unbias), no old-vs-new table
   numbers. Present the final methods and results as if designed that way
   from the start; the findings below are inputs to that narrative, not
   events to recount.
2. **Every number traceable.** Each quoted value must come from a named,
   archived JSON/npz log. Verify by opening the log, never from memory.
   Record the mapping in the "Result sources" section below as you go.
3. **Honest reporting.** Failures and caveats stay in (extrapolation
   degradation, diffuse-phase weakness, CNN sensitivity). The bias is
   reported alongside its correction, not hidden by it.
4. Keep the RASTI class and structure (firstpage/pagerange/keywords).

## Metric conventions (agreed with the user, 2026-07-03)

The metric hierarchy — reflect it in text, tables, and abstract:

- **Primary:** RMSE (dex) decomposed into bias + scatter; total H2 mass
  ratio (predicted/true). These answer "can you trust the predicted
  densities?".
- **Secondary:** log-space R² (keep for literature comparability, with ONE
  sentence noting its fold-variance denominator caveat — the paper's §5.4
  stratified analysis already demonstrates the pathology); fraction within
  0.1/0.3 dex; phase-conditional metrics (molecular vs diffuse at
  log nH2 = −4); clipped linear-space R² stays demoted.
- **Optional/supporting:** CCC (Lin's concordance — penalises bias R²
  forgives), W1 (Wasserstein distance between marginals, dex; replaces the
  old KL annotation), skill vs the pointwise-XGBoost baseline
  (1 − MSE/MSE_ref — a meaningful reference, unlike R²'s fold-mean).
- Aggregate over the 7 folds with mean ± fold std, `nanmean` where
  phase-conditional values are NaN. Model-ranking claims ("X beats Y")
  should note per-fold consistency (e.g. "wins in all seven folds"), not
  just means — 7 folds is the inferential unit, never the 14.7M cells.

Metric keys in the JSONs: `R2, RMSE, MAE, bias, scatter, R2_lin,
mass_ratio, MAE_mw, frac_01, frac_03, frac_05, CCC, W1, R2_mol, RMSE_mol,
bias_mol, R2_dif, RMSE_dif, bias_dif, f_mol, skill_vs_xgb`.

## Result sources — superseded

This planning table is superseded: all runs completed 2026-07-05 and the
definitive file → table/figure mapping is in the section "Result sources
(final — core run plan completed 2026-07-05)" below. One addition made
during the rewrite: the Discussion's evaluation-scale sentence ("scoring
the same stacked predictions against a 2×2×2 box-averaged target raises
R² from 0.962 to 0.991") pairs the raw stacked_sp row of
`arch_comparison_20260703_234017.json` (grid-scale, 0.9623) with the same
row of `arch_comparison_20260703_185744.json` (box-averaged evaluation of
the identical predictions, 0.991).

## Findings from the 2026-07-03/04 runs (write the paper around these)

0. **Evaluation-resolution fix (affects every ensemble/stacked number ever
   quoted).** Until 2026-07-03 the ensemble/stacked predictions were pooled
   to 64³ before metric computation (`_normalize_preds_to_64`), while the
   pointwise rows used native 128³ cells — the paper's Table 1 mixed the
   two. Since `_align_preds`, ensembles are evaluated on the same 2.1M
   native cells as everything else. Consequences: the paper's stacked_sp
   0.988 ± 0.013 (a 64³-pooled number) is superseded by native-resolution
   values that are LOWER (best ≈ 0.985) — not a regression, a consistent
   table. NEVER mix pre- and post-`_align_preds` ensemble numbers.
1. **At native resolution the raw stacked models over-predict mass — the
   §5.4 story and the OOF comparison now agree.** Run 1b raw stacked_sp:
   bias +0.32 dex, mass ratio 1.08–1.64 — matching the ×1.6–1.85
   over-prediction the saved §5.4 volumes always showed (both pipelines are
   128³). The earlier "unbiased stacked, mass 0.89" reading came from run
   1's pooled evaluation. So recalibration IS needed after all — but the
   right flavour matters (finding 2).
2. **Mean-recal vs mass-recal, the methodological point (fig7 evidence):**
   the unweighted `_cal` centres the cell-mean bias (−0.01) but *overshoots*
   the mass budget (stacked_sp_cal mass 0.50–0.89): the cell-mean offset is
   dominated by the diffuse cells and is larger than the mass-weighted
   offset, so dense cells get pushed too low. The mass-weighted `_mwcal`
   closes the budget: **`stacked_weighted_mwcal` is the headline candidate**
   — RMSE 0.294 (best of all 17 variants), mass ratio 0.93–1.07 in every
   fold, R² 0.9847 ± 0.0147 (within noise of stacked_sp_cal 0.9848),
   R2_mol 0.991, bias_mol −0.01, frac_01 0.68, W1 0.13. Residual trade-off
   to report honestly: diffuse-phase bias +0.10 dex (vs −0.00 for _cal) and
   overall bias +0.09; skill_vs_xgb 0.55 vs 0.60 for _cal (skill uses
   unweighted MSE, which favours the cell-mean correction by construction).
   Present both corrections; primary-metric hierarchy (RMSE + mass) picks
   mwcal.
3. **The f_H2 ablation costs far more than the R² comparison admits** —
   sharpen §5.3 with the new metric columns (run 2, pointwise rows:
   molecular-phase R² falls with a −0.16 dex molecular bias for XGB+sp,
   frac_01 collapses). Run 2b (`ablation_nofh2_128eval.json`, 2026-07-04)
   supplies the native-128³ ensemble/stacked rows — Table 2 must use it,
   not run 2 (64³-pooled; its pooled stacked 0.977 is superseded).
   Run-2b results, same variant with → without f_H2 (from run 1b → 2b):
   - `stacked_weighted_mwcal`: R² 0.985→0.964, RMSE 0.294→0.469,
     **R2_mol 0.991→0.840**, R2_dif 0.857→0.744, bias_mol −0.01→−0.10;
     mass budget survives (0.93–1.16 all folds). Headline sentence: losing
     f_H2 costs 60 per cent in RMSE and most of the molecular-phase
     fidelity, while aggregate R² conceals it (0.985→0.964 "looks fine").
   - Degradation is concentrated where H2 matters most: at G0=0.1 (the most
     molecular cube) R2_mol drops to 0.70 and R2_dif to 0.49, vs ≈0.88 at
     high G0.
   - Two run-1b conclusions are *reinforced* without f_H2: mean-recal
     overshoots mass even harder (stacked_sp_cal mass 0.38–0.63,
     stacked_weighted_cal 0.63–0.82, vs mwcal 0.93–1.16), and the weighted
     stack beats the sp stack outright (mwcal R² 0.964 vs 0.935) — the
     headline-model choice is robust to the feature set.
   - Determinism: all 6 pointwise rows of run 2b bit-match run 2.
4. **Phase-conditional story (run 1b):** the recalibrated stacks fix the
   diffuse phase relative to XGB+sp (bias_dif +0.20 → −0.00/_cal or
   +0.10/_mwcal; R2_dif 0.81 → 0.86–0.87); molecular-phase R² ~0.99
   throughout. Density-weighted training alone already closes most of the
   pointwise mass gap: weighted XGB+sp mass 0.94 vs 0.72 unweighted at
   bit-identical R².
5. **Numbers that moved for non-ensemble reasons:** mlp_wide_sp
   0.968 ± 0.041 → 0.9821 ± 0.0169 (the paper's per-fold 0.871 collapse at
   G0=0.8 is gone — now 0.945); every abstract/§5/conclusions sentence
   quoting 0.988/0.25 dex or the 0.871 fold must be rewritten from
   `arch_comparison_20260703_234017.json`. XGB rows are bit-identical to
   the March run — worth a one-line reproducibility remark (deterministic
   pipeline, same seed).
6. **Bias/scatter decomposition attributes what each method ingredient
   does** (use to structure the methods→results narrative): spatial
   features reduce scatter only (XGB 0.357 → 0.253, MLP 0.458 → 0.289 dex;
   bias nearly unchanged), while recalibration cannot touch scatter by
   construction (all three stacked_sp variants share scatter = 0.2518
   exactly) and only relocates bias. Non-local context buys precision;
   calibration decides where accuracy is centred; density weighting decides
   *which* functional is unbiased (cell-mean vs mass). The gap between the
   fitted mean and mass offsets (~0.1–0.2 dex) is a per-model one-number
   diagnostic of density-dependent (compression-to-the-mean) error — worth
   reporting.
7. **W1 + frac_01/03 expose a model-family difference R² hides:** mlp_wide
   has worse RMSE than xgb_standard (0.518 vs 0.486) but better W1 (0.217
   vs 0.300) and far better frac_03 (0.726 vs 0.498) — trees are moderately
   wrong everywhere (compressed distribution), the MLP is right for most
   cells with a heavy error tail. Best marginal realism in the table:
   weighted MLP+sp (W1 ≈ 0.109), slightly better than the stacks (Ridge
   averaging narrows the predicted distribution). Discussion sentence: for
   generating statistically realistic nH2 fields (synthetic observations,
   sub-grid realizations) the weighted MLP+sp may be preferable to the
   stacked model.
8. **Concrete R²-pathology example for the metrics-caveat sentence:**
   mlp_wide beats xgb_standard on mean R² (0.958 vs 0.950) while its pooled
   MSE is 18 per cent worse (skill −0.18) — per-fold R² normalises by each
   fold's variance, so winning high-variance folds buys R² cheaply. A model
   pair whose ranking inverts between R² and MSE, from the table itself.
9. **Scientific interpretations to weave into §5/§7:**
   - The out-of-distribution error is dominated by a global, G0-smooth
     normalisation offset, not by structure: a 2-parameter (log G0) linear
     correction removes ~30 per cent of RMSE (0.42 → 0.29) and closes the
     mass budget. The surrogate gets the morphology right two octaves
     outside its training range; what it misses is the overall H2
     normalisation vs UV field strength — the most interpretable and
     correctable failure mode. Raw over-prediction is worst at LOW G0
     (mass ×1.64 at G0=0.1), i.e. extrapolating toward weak-UV, H2-rich
     conditions.
   - Accuracy is resolution-dependent: identical predictions score
     R² 0.991 when evaluated on 2×2×2-averaged fields vs 0.962 at the grid
     scale — roughly half the error variance lives at the smallest scales
     (cell-level structure partly unpredictable from local features).
     Present as a designed "accuracy vs evaluation scale" result
     (applications needing the smoothed field get the higher figure); per
     constraint 1, do NOT present it as an evaluation-protocol history.
   - The diffuse phase is the persistent hard regime (R2_dif 0.81–0.87 vs
     R2_mol ≈ 0.99) and is where the recalibration trade-off lands. Benign
     for mass budgets (diffuse gas holds negligible H2 mass) but a real
     caveat for diffuse-gas chemistry applications.
10. **U-Net at native 128³ (run 3, `cnn_test_20260704_122206.json` —
    Table 1 U-Net row): best interpolator, worst extrapolator.**
    Headline: R² 0.970 ± 0.039, RMSE 0.310, skill_vs_xgb +0.48 (computed
    against run 1b xgb_standard; same native cells so the comparison is
    exact). The split that matters:
    - Interior folds (G0 0.2–3.2): R² 0.9946, RMSE 0.187 — the U-Net
      BEATS every classical model including the stack (0.9836/0.311).
      Non-local context genuinely helps in-distribution.
    - Edge folds (G0 0.1, 6.4): R² 0.907, RMSE 0.618 vs stack 0.988/0.253.
      Per-fold skill vs XGB: −1.14 at G0=0.1 (worse than raw pointwise
      XGB), +0.23 at 6.4.
    - Mass budget uncontrolled OOD: mass ratio 0.81/3.9/0.94/1.8/2.8/
      16.7/171.6 across folds (computed on clipped predictions → lower
      bounds). At G0=6.4: bias_mol +1.09 dex, R2_mol 0.05. At G0=0.1:
      bias −0.63 dex, R2_dif 0.16. MAE_mw 0.90 dex vs 0.06 for
      stacked_weighted_mwcal — 15× worse where the H2 mass lives.
    - Preempt the fairness question ("you recalibrated the stack but not
      the CNN"): the stack's OOD bias is a global, G0-smooth offset —
      exactly what a 2-parameter log-G0 fit corrects. The U-Net's OOD
      failure is phase-localised (bias_mol +1.09 vs small diffuse bias at
      6.4) and wildly non-linear in log G0 (mass ratio spans 0.8–172,
      super-linear at the edge), so the same correction cannot fix it.
      Supported directly by run-3 numbers, no extra run needed.
    - Checkpoint disclosure (code-review finding): all CNN results use
      the best-epoch checkpoint selected on the held-out cube's loss.
      Run-3 histories bound the advantage: best epoch 61–128 of 150,
      final/best val-MSE ratio 1.01–1.14 → ≤7 % RMSE. Disclose as "upper
      bound for the U-Net"; conservative for the paper's claim since the
      favoured baseline still loses OOD.
    - Native-vs-64³: per-fold pattern matches the archived
      cnn_test_20260323 log (means 0.970 vs 0.974) — the resolution
      change does not alter any conclusion; archived-log claims stay
      valid with the "at 64³" caveat.
    - Mean mass ratio (28 ± 59) is meaningless for this row — quote the
      per-fold range or median (2.8) in Table 1, never the mean.
11. **11-input U-Net (run 4, `cnn_test_20260704_170833.json` — the §5.2
    "upper bound" source, replaces the archived 64³ 0.874):** with only
    log_T + log_G0 + velocity + B (no chemistry-local inputs), the U-Net
    reaches R² 0.893 ± 0.088 at native 128³. Two-sided reading:
    - The headline sentence stays: ~90 per cent of the log-nH2 variance
      is recoverable from dynamics, temperature and UV strength alone —
      the H2 morphology is largely inferable without any chemistry
      inputs.
    - But the new metrics show what that 0.89 hides: chemistry
      *precision* is gone — scatter 0.61 dex, frac_01 0.17, R2_mol
      0.09 ± 1.18 (negative at both edges: −1.22 at G0=0.1, −2.24 at
      6.4), molecular bias −1.08/+2.14 dex at the edges, mass ratio
      0.27–×800 (clipped → lower bounds). Phrase §5.2 as "morphology
      yes, chemistry no", not as 90 per cent of the problem solved.
    - Comparability guard: run 4 (11 inputs) is NOT the CNN analogue of
      Table 2's no-fh2 rows (those keep nH/nHp/ext — 14 features); it
      pairs only with the archived 11-input 64³ log
      (`cnn_test_20260322_221728.json`) for config-sensitivity claims.
    - Checkpoint disclosure update: run 4's final/best val-MSE ratios
      are 1.10–1.25 (≤ 12 % RMSE), larger than run 3's ≤ 7 % — the
      disclosure sentence (audit item 5) should quote ≤ 12 % to cover
      both CNN runs.

## Result sources (final — core run plan completed 2026-07-05)

Every quoted number must trace to one of these. Do not mix ensemble
numbers across the pooled/native eval boundary (finding 0) or across the
two stacking protocols (disclosure 7).

| Paper element | Source | Notes |
|---|---|---|
| Table 1 (all classical + ensemble rows) | `results/arch_comparison_20260703_234017.json` (run 1b) | native-128³ eval; mwcal variants; importance blocks |
| Table 1 U-Net row | `results/cnn_test_20260704_122206.json` (run 3) | R² 0.970±0.039; skill vs xgb computed cross-log (valid: same native cells); quote mass ratio as range/median, never mean |
| Table 2 (f_H2 ablation) | `results/ablation_nofh2_128eval.json` (run 2b) | pointwise rows bit-match run 2; NEVER quote run 2's pooled ensemble rows |
| §5.2 11-input upper bound | `results/cnn_test_20260704_170833.json` (run 4) | R² 0.893±0.088; "morphology yes, chemistry no" |
| §5.4 error analysis + D2/D3 | `predictions/pred_g0_*_20260704_*.npz` (run 5) | nested weighted stack + mwcal; R² 0.9901±0.0064, mass 0.985–1.090; independent check = merit_metrics table 2026-07-05 |
| §6.1 single-cube matrix | `logs/single_cube_extrapolation/run_20260705_114057.json` (run 10, 2026-07-05) | consistent with March matrix; full metric dicts |
| §6.2 intra-cube section | `logs/intra_cube_section/run_20260705_121037.json` (run 11, 2026-07-05) | ⚠ March log irreproducible (uncommitted script variant); §6.2 text rewritten from this log — do NOT quote the March slab/rand_1 numbers |
| U-Net config sensitivity | archived `results/cnn_test_20260322_221728.json`, `results/cnn_test_20260323_010315.json`, `results/cnn_training_*` | 64³ — say so explicitly |
| Fig 1 (histograms) | `plot_nH2_histograms.py` | data unchanged |
| Fig 2 (method diagram) | static | |
| Fig 3 replacement (model comparison) | `figures/fig_model_comparison.png` | run 1b + run 3 via `--cnn-log`; regenerate with `python plot_model_comparison.py --log results/arch_comparison_20260703_234017.json --cnn-log results/cnn_test_20260704_122206.json --variants ...` |
| Figs 4–8 (scatter, residuals, stratified, marginals, slices) | `figures/fig2_scatter, fig3_error_dist, fig4_stratified, fig6_distributions, fig8_slices` (2026-07-05, from run-5 volumes) | fig6 caption must cite KS/W1, not KL |
| Mass-budget evidence (new) | `figures/fig7_massbudget.png` | raw ×1.65–1.90 vs recal ≈1; phase panel |
| Feature importance (new, D4) | `figures/fig_feature_importance.png` | from run 1b importance blocks |
| Figs 9–10 (§6 heatmaps) | archived heatmaps in `logs/` | update timestamped paths in paper.tex if runs 10/11 happen |

## Audit-mandated disclosures (2026-07-04 design audit)

Full rationale per item in `docs/DESIGN_DECISIONS.md`; this is the
checklist the rewritten paper must satisfy. All are text-only — none
require new runs. Items marked (VERIFY) depend on answers from the
simulation collaborators (see RUN_PLAN "Verification items"); draft the
sentence with a placeholder if the answer is pending.

1. **Prediction clipping.** Metrics section must state: linear-space
   quantities (`R2_lin`, mass ratio) are computed on predictions clipped
   to the truth range ± 1 dex; for the classical models the clip
   essentially never binds; for the U-Net's extrapolation folds the mass
   ratios are therefore **lower bounds** (say so wherever ×172 etc. are
   quoted). Also describe the MLP's training-range ± 2 dex output clamp
   as part of the model definition and note the U-Net is the only
   unbounded model. [DESIGN_DECISIONS §1]
2. **f_H2 wording.** Never "independent quantity". Correct framing: a
   solver-internal quantity (self-shielding derives from the H2 column —
   an integral of the target field), available in deployment only where
   the host code already tracks H2; the no-f_H2 configuration (Table 2)
   is the deployable one. (VERIFY exact computation with collaborators.)
   [§2.2]
3. **Dynamic range.** The target spans **~16 dex** (4.7×10⁻¹² to
   9.2×10³ cm⁻³, no zeros in the data) — replace any "~30 orders of
   magnitude" claim. [§2.1]
4. **Tree extrapolation in G0.** One §5 sentence: trees cannot
   extrapolate in the log G0 coordinate (splits saturate at the training
   range), partially explaining XGBoost's edge-fold behaviour, while the
   MLP/stack extrapolate the trend. [§2.3]
5. **U-Net protocol disclosures** (can share one footnote/paragraph):
   (a) best-epoch checkpoint selected on the held-out cube's loss —
   quantified ≤ 12 % RMSE advantage across runs 3–4 (run 3: final/best
   val-MSE 1.01–1.14, ≤ 7 %; run 4: 1.10–1.25, ≤ 12 %), so U-Net numbers
   are an upper bound — which is conservative for the paper's claim;
   (b) InstanceNorm gives the CNN
   per-cube test-time input normalisation the pointwise models don't
   get (inputs only, not leakage). [§4.5, §4.7]
6. **Unweighted-CNN limitation.** The headline stack is density-weighted
   (mass-aware), the U-Net is cell-mean-trained: one limitations sentence
   that a weighted U-Net is future work; the interior/edge structural
   result does not depend on it. [§4.6]
7. **Stacking protocol.** Methods describe the fully nested §5.4
   procedure; the comparison-table stacked rows use the standard
   reuse-the-CV-OOF-predictions shortcut — do NOT write "fully nested"
   or "no leakage" for Table 1's stacked rows; the two pipelines' numbers
   are close but not identical. [§6.1]
8. **PHASE_SPLIT = −4.0.** Justify against the Fig 1 bimodality (check
   the split lands in the density minimum when writing the sentence);
   note it matches merit_metrics. [§7.1]
9. **Scope of the OOD claim.** One realization per G0, shared initial
   conditions → generalisation claim is over UV field strength, not over
   unseen clouds; phrase §2/§7 accordingly (different-seed cube listed as
   future work / optional run 13). [§3.2]
10. **Boundary conditions.** (VERIFY) State the cubes' BCs; if periodic,
    disclose that spatial-feature filters (`reflect`) and CNN
    zero-padding are approximations affecting ~13 % of cells (within 3
    cells of a face at k=7). [§5.1]
11. **Augmentation subgroup.** (VERIFY UV geometry) State why the
    8-element z-preserving subgroup (C4v) is used — required if the UV
    field is directional along z, conservative if isotropic. [§5.2]
12. **Mass-ratio interpretation.** (VERIFY box metadata, same source as
    C3/units) mass_ratio = H2 mass ratio assumes uniform grid, equal
    cell volumes, identical box size across G0 — state it. [§7.3]
13. **Density-weighting hyperparameters.** State the chosen scheme
    (1× at p99 → α=100 at p99.99, exponential, mean-normalised); if
    optional run 12 (α sweep) is done, cite it as the sensitivity check;
    if not, flag the values as a designed choice. [§4.1]
14. **§6.2 masking clause.** One clause: spatial features for the
    intra-cube experiment are computed from training-section-only
    volumes (conservative — boundary training cells are also degraded),
    so §6.2 slightly understates interpolation performance. [§6.4]

## Table redesign

- **Table 1**: keep the per-fold R² columns (they carry the
  interpolation-vs-extrapolation story) but change the summary columns to:
  mean R² ± std, mean RMSE, mean bias, mean mass ratio, mean skill. If too
  wide for RASTI two-column even as `table*`, split: Table 1a per-fold R²,
  Table 1b summary metrics per model.
- **Table 2**: same treatment (with/without f_H2 rows now get
  RMSE/bias/mass-ratio columns, not just R² ± std).
- **New Table 3 (if run 8 done)**: minimal-feature-set comparison
  (all 15 / no vel+B / no f_H2 / no vel+B+f_H2 as available).
- The stacked ensemble now has `stacked_sp` and `stacked_sp_cal` variants
  in the logs. The **recalibrated** variant is the headline model IF its
  R² is within noise of raw and its mass ratio is ≈1 — check, then present
  recalibration as part of the method (§ Ensembles), with fig7 as evidence.

## Figure plan (10 → 12, net)

| Paper slot | File | Action |
|---|---|---|
| Fig 1 | `figures/nH2_histograms.png` | keep as-is |
| Fig 2 | `fig_method_diagram.png` | keep as-is |
| Fig 3 | `figures/fig1_summary.png` | **REMOVE** — replace with `fig_model_comparison.png` (per-fold R²/RMSE/skill curves; caption: grey bands = extrapolation folds) |
| Fig 4 | `fig2_scatter.png` | keep, regenerate |
| Fig 5 | `fig3_error_dist.png` | keep, regenerate; caption must reflect post-recalibration biases |
| Fig 6 | `fig4_stratified.png` | keep, regenerate |
| Fig 7 | `fig6_distributions.png` | keep, regenerate; **caption bug**: currently cites KL divergence, the figure annotates KS D, p, and W1 — rewrite caption |
| Fig 8 | `figures/fig8_slices.png` | **path change** (was a stale PNG the old code couldn't regenerate); caption already matches this layout |
| NEW | `figures/fig7_massbudget.png` | insert in §5.4: mass ratio + bias raw-vs-recalibrated + phase-conditional R² |
| NEW | `figures/fig_feature_importance.png` | insert in Methods or Discussion (closes review D4) |
| Fig 9 | `logs/single_cube_extrapolation/heatmap_20260705_114057.png` | updated 2026-07-05 (run 10) |
| Fig 10 | `logs/intra_cube_section/run_20260705_121037_heatmap.png` | updated 2026-07-05 (run 11; March heatmap shows irreproducible numbers — do not revert) |

`\graphicspath` in paper.tex is `{./}{../}` — figures/ and logs/
paths resolve via `../` from `paper/`. Verify every `\includegraphics`
target exists before building.

## Narrative changes required

1. **§5.4 becomes "audit + recalibration".** The old text reports a
   +0.2–0.5 dex bias and says recalibration "would largely remove" it. The
   new prediction run applies the G0-linear OOF recalibration
   (leakage-free, described in `model_helpers.fit_g0_bias_correction`).
   Present: R² alone missed a ×1.6–1.85 mass-budget error (fig7 raw
   series); the recalibration, fitted on training-cube quantities only,
   brings the mass ratio to ≈1 (fig7 cal series). Verify the actual
   post-cal numbers from the new npz before writing "≈1".
2. **Metrics section (§3.2) expands** to define the hierarchy above
   (bias/scatter decomposition, mass ratio, phase split at −4, CCC, W1,
   skill). One paragraph, with the R² caveat sentence.
3. **Phase-conditional results** enter §5.4: molecular-phase R² high,
   diffuse-phase R² as low as ~0.6 at G0=0.8 in the old run — check new
   values; report both phases.
4. **Abstract**: rewrite the results sentences around the primary metrics
   (e.g. "RMSE of X dex with |bias| < Y dex and total H2 mass recovered to
   Z per cent" alongside the R² values, which stay for comparability).
   Update every number from the new logs.
5. **Conclusions**: add the mass-budget/recalibration conclusion; update
   all numbers; if run 8 (no-vel/no-B) was done, add the minimal-feature
   conclusion; if the z-cumsum experiment was done, integrate it into the
   feature-engineering contribution (it may become contribution (ii)).
6. **U-Net resolution (changed 2026-07-03).** New CNN runs train at
   **native 128³** — no downsampling (the code's `--downsample` /
   `--cnn-downsample` flags are legacy-only; JSONs record
   `grid_size`/`cnn_grid`, check it when citing a log). Consequences:
   - The old "GPU memory forces 64³ input, discarding fine-scale
     gradients" limitation paragraph must be REMOVED or rewritten — it no
     longer applies to the new runs. This also closes the REVIEWER2 point
     that the U-Net trained at 64³ while tabular models saw 128³: CNN and
     pointwise models are now evaluated on the same 128³ cells.
   - The archived config-sensitivity logs (Result sources table) are 64³
     runs. If cited alongside new 128³ numbers, state the resolution
     difference explicitly; never mix them in one table without a column
     or footnote saying so.
   - Ensembles no longer pool pointwise predictions to 64³
     (`_align_preds` is a no-op when everything is 128³).
7. **Do not change** the §2 data description, §3.1 CV protocol, or the
   limitations structure except where numbers changed. The §2.1 wording
   about evolved cubes differing in T/velocity was carefully negotiated —
   leave it.

## Alter vs rewrite decision

- Only core reruns (1–7) done → **alter in place**: numbers, tables,
  figure swaps, §3.2 + §5.4 rewrites, abstract/conclusions. ~1 session.
- Runs 8–9 also done (new experiments) → restructure §4–§7 (new methods
  subsection for feature variants, new results subsection, updated
  contributions list). Rewriting those sections is cleaner than patching.
- Either way keep title, author block, §1 structure, and the RASTI class.

## Verification checklist before finishing

1. Every number in abstract/tables/text grep-matched against its source
   JSON (`python -c "import json; ..."` spot checks, not eyeballing).
2. Every `\includegraphics` file exists; no stale timestamped paths.
3. `latexmk -gg -pdf paper.tex` clean (only Hfootnote.1 warning); page
   count sane; check rendered first page with Ghostscript
   (`rungs -q -sDEVICE=txtwrite ...` — the Read tool may falsely report
   the PDF as password-protected; it is not).
4. No forbidden framing (grep the .tex for "early run", "development",
   "corrected", "originally", "we later").
5. Captions consistent with what the current scripts draw (esp. Fig 7
   KS/W1 and Fig 8 slices).
6. Remaining pre-submission blockers from `REVIEWER2_REPORT.md` Parts B–E
   that this update does NOT close: C1 (simulation provenance table),
   C2 (code/data release), C3 (physical units of velocity/B in code
   units). Flag them to the user at the end; do not invent provenance.
