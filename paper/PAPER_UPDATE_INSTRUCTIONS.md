# Instructions: updating or rewriting the paper with the new results

Audience: a future Claude Code session asked to take the rerun results and
figures (see `../RUN_PLAN.md`) and either revise `paper/paper.tex` in place
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

## Result sources (fill in as runs complete)

| Content | Source file | Status |
|---|---|---|
| Table 1 (main comparison) | `arch_comparison_20260703_234017.json` (run 1b) | DONE. Do NOT take ensemble/stacked rows from `..._185744.json` — that run evaluated them 64³-pooled (see finding 0) |
| Table 2 (f_H2 ablation) | `arch_comparison_20260703_200316.json` | DONE, but its ensemble/stacked rows are ALSO 64³-pooled (pre-`_align_preds` launch) — quote only its pointwise rows, or rerun `--no-fh2` for consistent ensemble rows |
| Feature-importance figure | `arch_comparison_20260703_234017.json` | DONE (xgb_standard, _sp, _sp_w blocks) |
| U-Net best-config row | `cnn_test_<NEW>.json` | pending rerun |
| 11-input U-Net bound | `cnn_test_<NEW>.json` | pending rerun |
| U-Net config sensitivity | archived: `cnn_test_20260322_221728.json`, `cnn_training_2026030*.json`, `arch_comparison_20260310_224154.json` | valid, keep |
| §5.4 error analysis | `predictions/pred_g0_*_<NEW>.npz` (`metrics_json` key) | pending rerun |
| §6.1 transfer matrix | `logs/single_cube_extrapolation/run_20260313_022129.json` | valid unless rerun |
| §6.2 sampling geometry | `logs/intra_cube_section/run_20260313_142443.json` | valid unless rerun |
| Optional: no-vel/no-B ablation | `ablation_novelB_<NEW>.json` | optional |

If a table's source is still "pending", STOP and tell the user which runs
are missing rather than reusing old numbers for new-metric columns.

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
   frac_01 collapses). ⚠ Run 2's ensemble/stacked rows are 64³-pooled —
   take no-fh2 ensemble numbers from run 2b only (see RUN_PLAN).
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
| Fig 1 | `nH2_histograms.png` | keep as-is |
| Fig 2 | `fig_method_diagram.png` | keep as-is |
| Fig 3 | `analysis_output/fig1_summary.png` | **REMOVE** — replace with `fig_model_comparison.png` (per-fold R²/RMSE/skill curves; caption: grey bands = extrapolation folds) |
| Fig 4 | `fig2_scatter.png` | keep, regenerate |
| Fig 5 | `fig3_error_dist.png` | keep, regenerate; caption must reflect post-recalibration biases |
| Fig 6 | `fig4_stratified.png` | keep, regenerate |
| Fig 7 | `fig6_distributions.png` | keep, regenerate; **caption bug**: currently cites KL divergence, the figure annotates KS D, p, and W1 — rewrite caption |
| Fig 8 | `analysis_output/fig8_slices.png` | **path change** (was a stale PNG the old code couldn't regenerate); caption already matches this layout |
| NEW | `analysis_output/fig7_massbudget.png` | insert in §5.4: mass ratio + bias raw-vs-recalibrated + phase-conditional R² |
| NEW | `analysis_output/fig_feature_importance.png` | insert in Methods or Discussion (closes review D4) |
| Fig 9 | `logs/single_cube_extrapolation/heatmap_20260313_022129.png` | keep; update path only if rerun |
| Fig 10 | `logs/intra_cube_section/run_20260313_142443_heatmap.png` | keep; update path only if rerun |

`\graphicspath` in paper.tex is `{./}{../}` — analysis_output/ and logs/
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
