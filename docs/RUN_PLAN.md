# Run plan — regenerating all paper data and figures

Written 2026-07-03, after the metrics overhaul (full metric suite recorded by
`compute_metrics`: RMSE = bias + scatter, mass_ratio, frac_01/03/05,
phase-conditional R²/RMSE/bias, MAE_mw, CCC, W1, skill_vs_xgb) and the figure
overhaul (fig7/fig8 + two standalone figure scripts). Old JSON logs remain
*valid* but lack the new metric keys — that is why the reruns are needed.

## Before starting

- Data: `data/UVonly/<G0>/`; loading all 7 cubes takes several
  minutes per script invocation.
- Python stdout is fully buffered when redirected — always use `python -u`
  and redirect to a log file for the long runs.
- GPU: all timings below are for the RTX 3090. Runs 1–2 and 5 can NOT share
  the GPU with runs 3–4; run them sequentially.
- Sanity check first: `python smoke_test_metrics.py` (seconds, no data
  needed) — all 9 checks must pass.
- **CNN resolution change (2026-07-03):** `test_cnn.py`, `train_cnn.py` and
  the CNN paths of `compare_architectures.py` now train at **native 128³**
  by default (`--downsample` / `--cnn-downsample` restores the old 64³
  average-pooling). VRAM verified on the 3090: peak 2.9/5.7/11.4 GiB for
  base_ch 16/32/64 at batch=1 (`vram_test_128.py`). Consequences:
  - Runs 3–4 process 8× more voxels per volume — expect roughly 4–8× the
    old per-epoch time; budget accordingly.
  - New CNN numbers are **not comparable** to archived 64³ logs
    (`cnn_test_20260322_*`, `cnn_training_*`); every JSON now records
    `grid_size`/`cnn_grid`. Config-sensitivity claims citing the archived
    logs must say they were obtained at 64³.
  - Ensembles mixing CNN and pointwise predictions no longer pool the
    pointwise models down to 64³ (`_align_preds` replaces
    `_normalize_preds_to_64`; alignment only happens for legacy 64³ runs).

## Core runs (required, in this order)

| # | Command | Time | Output | Feeds |
|---|---------|------|--------|-------|
| 1 | `python -u compare_architectures.py > run_main.log` | ~2.5 h | `results/arch_comparison_<ts>.json` | Table 1 (all metrics + skill), model-comparison figure, feature-importance figure |
| 2 | `python -u compare_architectures.py --no-fh2 --log results/ablation_nofh2_<date>.json > run_nofh2.log` | ~2.5 h | ablation JSON | Table 2 (f_H2 ablation) |
| 3 | `python -u test_cnn.py --variants unet_baseline > run_cnn.log` | hours | `results/cnn_test_<ts>.json` | U-Net row of Table 1 |
| 4 | `python -u test_cnn.py --no-fh2 --no-nH --no-nHp --no-ext --variants unet_baseline > run_cnn11.log` | hours | `results/cnn_test_<ts>.json` | 11-input U-Net upper-bound number (§5.2) |
| 5 | `python -u predict_and_visualize.py --all > run_pred.log` | ~2–3 h | `predictions/pred_g0_*_<ts>.npz` (raw + recalibrated volumes, `metrics_json`) | §5.4 error analysis; closes review items D2 (full 100-epoch schedule) and D3 (bias recalibration) |
| 6 | `python statistical_analysis.py --pred-dir predictions --save-dir figures` | ~10 min | `figures/fig1..fig8` | Figures (see below) |
| 7 | `python merit_metrics.py > merit_table.txt` | minutes | console table | independent mass-budget / phase check |

### Status (updated 2026-07-03 evening)

- **Run 1 DONE** → `arch_comparison_20260703_185744.json` (~1 h, not 2.5 h).
  ⚠ Launched minutes before the importance-recording code landed, so it has
  **no `xgb_feature_importance` blocks** and predates the `*_mwcal` variants.
- **Run 2 DONE** → `arch_comparison_20260703_200316.json` (~1 h). This IS the
  Table 2 source despite the generic filename (`--log` wasn't used). Has
  importance blocks (14-feature set, no log_fh2). Bit-identical to the 7/1
  `ablation_nofh2_spatial.json` — determinism verified.
- **Run 1b DONE** → `arch_comparison_20260703_234017.json` (~1 h). All
  sanity checks passed: every pointwise row bit-matches 185744 on every
  metric key; importance blocks present (xgb_standard, _sp, _sp_w); mwcal
  variants present. **This is the Table 1 source.**
  ⚠ Ensemble/stacked rows are NOT comparable to run 1 or any earlier log:
  the old `_normalize_preds_to_64` pooled all ensemble evaluation to 64³;
  `_align_preds` (since 2026-07-03) keeps native 128³ when no CNN is mixed
  in. Run 1b is the first run where every model is evaluated on the same
  2.1M native cells. At native resolution the raw stacked models show the
  same +0.2–0.3 dex / ×1.4–1.7 mass over-prediction the §5.4 volumes
  always showed (the two pipelines now agree — run 1's "unbiased stacked,
  mass 0.89" picture was an artifact of the pooled 64³ evaluation).
  Headline candidate per the metric hierarchy: `stacked_weighted_mwcal`
  (RMSE 0.294 best-of-table, mass 0.93–1.07 in all folds, R² 0.985 within
  noise of `stacked_sp_cal`; unweighted `_cal` fixes cell-mean bias but
  overshoots mass down to 0.50–0.89).
- **Run 2b DONE** (2026-07-04) → `ablation_nofh2_128eval.json` (~1 h).
  All 6 pointwise rows bit-match run 2 on every metric key (determinism
  verified); ensemble/stacked rows now native 128³ with the full mwcal set.
  **This is the Table 2 source** (run 2/`ablation_nofh2_spatial.json` are
  64³-pooled — do not quote their ensemble rows). Headline no-fh2 numbers:
  `stacked_weighted_mwcal` R² 0.964 ± 0.014, RMSE 0.469, mass 0.93–1.16,
  R2_mol 0.840, R2_dif 0.744. Ablation cost vs run 1b on the same variant:
  R² 0.985→0.964, RMSE 0.294→0.469, **R2_mol 0.991→0.840**,
  bias_mol −0.01→−0.10; degradation is worst at low G0 (g0=0.1 R2_mol 0.70,
  R2_dif 0.49). Mean-recal pathology is *stronger* without f_H2
  (stacked_sp_cal mass 0.38–0.63) and the weighted stack now clearly beats
  the sp stack (mwcal 0.964 vs 0.935) — both reinforce the run-1b headline
  choice. Details in PAPER_UPDATE_INSTRUCTIONS.md finding 3.
- **Run 3 DONE** (2026-07-04, ~4.6 h) → `cnn_test_20260704_122206.json`
  (unet_baseline, native 128³, 150 epochs). **This is the U-Net row of
  Table 1**: R² 0.970 ± 0.039, RMSE 0.310, skill_vs_xgb +0.48 (computed
  against run 1b xgb_standard; valid — same native cells). Headline
  pattern: best-in-table interpolator, worst extrapolator —
  interior folds (0.2–3.2) R² 0.9946 / RMSE 0.187 (beats the stack's
  0.9836/0.311); edge folds (0.1, 6.4) R² 0.907 / RMSE 0.618 (stack:
  0.988/0.253). Mass budget uncontrolled OOD: mass ratio per fold
  0.81 / 3.9 / 0.94 / 1.8 / 2.8 / 16.7 / **171.6** (clipped values →
  lower bounds); at G0=6.4 bias_mol +1.09 dex, R2_mol 0.05; at G0=0.1
  bias −0.63 dex. MAE_mw 0.90 dex vs 0.06 for stacked_weighted_mwcal.
  Native-vs-64³ sanity: per-fold pattern matches the archived
  cnn_test_20260323 log (0.970 vs 0.974 mean R²); conclusions unchanged.
  Checkpoint note (see review): best epochs 61–128/150; final/best
  val-MSE ratio 1.01–1.14 → best-checkpoint selection on the held-out
  cube is worth ≤7 % RMSE — quantify this in the disclosure sentence.
- **Run 4 DONE** (2026-07-04, ~4.5 h) → `cnn_test_20260704_170833.json`
  (unet_baseline, 11 inputs: log_T + log_G0 + v + B — no chemistry-local
  inputs — native 128³, 150 epochs). **§5.2 upper-bound source**:
  R² 0.893 ± 0.088 (archived 64³ number was 0.874 — same story, cite the
  native value now), RMSE 0.684, scatter 0.610. Reading: dynamics +
  temperature + UV alone recover ~90 % of log-nH2 variance (morphology),
  but chemistry precision is gone — R2_mol 0.09 ± 1.18 (negative at both
  edges: −1.22 at G0=0.1, −2.24 at 6.4), frac_01 0.17, bias_mol −1.08 /
  +2.14 dex at the edges, mass ratio 0.27–×800 (clipped → lower bound).
  Same edge collapse as run 3, amplified (edge R² 0.75). Checkpoint gap
  larger than run 3: final/best val-MSE 1.10–1.25 → ≤ 12 % RMSE (update
  the disclosure bound to cover both runs). Feature-set caveat: NOT
  comparable to run 2b's no-fh2 rows (those keep nH/nHp/ext, 14
  features); comparable only to the archived 11-input 64³ log.
- **Run 5 DONE** (2026-07-04 22:01–23:50, ~2 h) →
  `predictions/pred_g0_*_20260704_*.npz` (7 folds, mass recal default,
  metrics_json uses the new `xgb_sp_w`/`mlp_sp_w` keys). **§5.4 source;
  closes review items D2 (full schedule) and D3 (recalibration).**
  Delivered (nested stacked_weighted + mwcal): R² 0.9901 ± 0.0064,
  RMSE 0.250, bias +0.017, **mass ratio 0.985–1.090 every fold**,
  R2_mol 0.990, R2_dif 0.905 (worst 0.74 at G0=0.1), frac_01 0.73,
  W1 0.082, CCC 0.995. Raw stacks over-predict mass ×1.65–1.90 —
  consistent with both the old §5.4 volumes (×1.63–1.85) and run 1b.
  Protocol note: these NESTED numbers are *better* than Table 1's
  shortcut `stacked_weighted_mwcal` row (0.9847/0.294) — mostly because
  the G0=0.8 fold doesn't dip (0.9866 vs 0.9516); the stricter protocol
  scoring higher is a good look (disclosure item 7 wording).
  ⚠ Two stale partial files from 2026-07-03 remain for G0=0.1/0.2 —
  selectors correctly ignore them (latest-per-G0).
- **Run 7 DONE** (2026-07-05) — merit_metrics on the run-5 volumes
  independently confirms the npz metrics: mass 0.985–1.090,
  R2_mol 0.979–0.994, R2_dif 0.735–0.996, MAE_mw 0.03–0.09. The
  scientific-merit review's complaints (mass ×1.63–1.85, diffuse R²
  down to 0.60) are resolved in the delivered volumes.
- **Figures DONE** (2026-07-05): run 6 (`figures/fig1..fig8`)
  from the run-5 volumes; `fig_model_comparison.png` (run 1b + run 3 via
  `--cnn-log`); `fig_feature_importance.png` (run 1b — log_fh2 dominates
  ≈0.6 local/≈0.85 aggregated, then T/nH/nHp/G0; v and B negligible →
  closes D4 with the ablation, and predicts run 8's outcome).
- Run 5 note: `predict_and_visualize.py` applies **mass-weighted**
  recalibration by default (`--recal-mode mass`); the legacy cell-mean
  correction is `--recal-mode mean`, off is `--recal-mode off`. Both fits
  are recorded in each npz regardless of mode.
- Figure note for `plot_model_comparison.py`: it reads only
  arch_comparison JSONs, so the U-Net per-fold line (now in a cnn_test
  JSON) is missing from the replacement for paper Fig 3 — either extend
  the script with a `--cnn-log` merge option or drop the U-Net line and
  cite its numbers in Table 1 only.

Notes:
- Run 6 and 7 automatically pick the **latest** prediction file per G0 and
  skip the CNN-format npz that lives in `predictions/`.
- U-Net configuration-sensitivity claims do NOT need reruns — cite the
  archived logs (`cnn_test_20260322_221728.json`, `cnn_training_*.json`,
  earlier `arch_comparison_*` runs with `--cnn`). Only the best config gets
  fresh numbers.

## Optional runs (new science, recommended)

| # | Command | Time | Purpose |
|---|---------|------|---------|
| 8 | `python -u compare_architectures.py --no-vel --no-B --log results/ablation_novelB_<date>.json > logs/run_novelB.log` | ~1 h (per runs 1–2 timing) | **ELEVATED 2026-07-05** (both external reviews request it): converts §5.3's "minimal six-feature deployable set … not separately evaluated" into a table row. Expected: little accuracy loss → new result + closes review item D4 together with the importance figure |
| 9 | z-column cumulative-sum features — **code not yet written** (add a cumsum-along-z variant to `model_helpers._compute_spatial_X`, expose as `--spatial-mode zcum` or similar), then one comparison run and one `--no-fh2` run with it | ~0.5 day | Directional (physics-informed) vs isotropic non-locality; strongest candidate headline for the methods contribution |

## Optional runs from the 2026-07-04 design audit (see DESIGN_DECISIONS.md)

| # | Command | Time | Purpose / status |
|---|---------|------|------------------|
| 12 | Density-weight α sensitivity: **small code change first** (expose `_compute_weights` α — e.g. `--weight-alpha` in compare_architectures, or a standalone XGB-only sweep script), then α ∈ {10, 100, 1000} on `xgb_standard_sp_w` | ~1 h total (XGB-only) | **ELEVATED 2026-07-05** (ChatGPT review #6 requests exactly this table). The weighting is inside the headline model with three untested hyperparameters (p99/p99.99/α=100); a flat response inoculates the reviewer question. DESIGN_DECISIONS §4.1 |
| 13 | Different-seed test cube — **blocked on data** (request one re-seeded simulation at an existing G0 from the collaborators), then `predict_and_visualize.py` inference on it | inference only, minutes | Highest-value robustness addition: separates G0-difficulty from realization noise; scopes the OOD claim beyond shared initial conditions. DESIGN_DECISIONS §3.2 |
| 14 | Weighted U-Net (density-weighted CNN loss) — **post-paper / future work** | hours | Removes the mass-budget training asymmetry vs the weighted stack; for this paper a limitation sentence suffices. DESIGN_DECISIONS §4.6 |
| 15 | Boundary-mode robustness: **small code change first** (`mode` flag on `_compute_spatial_X`), then one `xgb_standard_sp` fold-set with `mode='wrap'` | ~20 min | Only if the sims turn out periodic (see Verification below); quantifies the `reflect` approximation on ~13 % boundary cells. DESIGN_DECISIONS §5.1 |

## Runs from the 2026-07-05 dual-AI review (paper/REVIEW3_DISPOSITIONS.md)

| # | Command | Time | Purpose / status |
|---|---------|------|------------------|
| 16 | **BLOCKING (both reviewers): leakage-free U-Net checkpoints.** Code change first: `test_cnn.py` selects the checkpoint on an *inner validation cube* (one of the six training cubes) instead of the held-out cube, and additionally records final-epoch metrics. Then rerun the run-3 config (`--variants unet_baseline`), and ideally the run-4 11-input config | ~4.6 h per config | Replaces the test-set-selected U-Net rows in Tables 2–3 and §5.4. Until done, the ≤12 % disclosure stands (per-epoch `history` in the cnn_test JSONs audits the bound but cannot yield final-epoch R²/mass) |

- **Run 16 DONE** (2026-07-05, ~4.5 h) → `results/cnn_test_20260705_185002.json`
  (`checkpoint_selection: inner_val_nearest_log_g0`; log
  `logs/run16_cnn_innerval.log`). `test_cnn.py` modified: each fold trains on
  FIVE cubes and selects the checkpoint on an inner validation cube
  (`pick_inner_val`, rule `nearest`: closest to the test cube in log10 G0,
  ties → lower). History records train/inner_val/test loss (test audit-only);
  fold logs add `inner_val_g0`/`best_epoch`/`metrics_final` (final-epoch);
  summaries add `summary_final`.
  **RESULT — the U-Net story changes qualitatively:** mean R² 0.914 ± 0.117
  (was 0.970 ± 0.039 in run 3), RMSE 0.500 (was 0.310), skill vs pointwise
  XGBoost **−0.35** (was +0.48). Interior folds R² 0.9857 / RMSE 0.313 —
  statistically indistinguishable from the shortcut stack (0.9836/0.311),
  beats it on only 1 of 5 interior folds (G0=0.8). Edge folds collapse much
  harder: G0=0.1 R² 0.794 (bias −1.01 dex), G0=6.4 R² 0.677 with mass ratio
  ×1128 (clipped lower bound), R2_mol −3.15, bias_mol +2.36. Selection vs
  final epoch is now nearly a no-op (mean R² 0.914 vs 0.907) — the old
  test-based selection was where the U-Net's advantage lived. **The "best
  interpolator" narrative does not survive the leakage-free protocol.**
  ⚠ **CONFOUND / OPEN DECISION:** the `nearest` rule removes each edge
  fold's only adjacent cube from training (fold 6.4 loses 3.2, fold 0.1
  loses 0.2), turning the edge tasks into TWO-step extrapolation while the
  classical models keep the one-step design — the edge collapse conflates
  honest selection with a harder task. `--inner-val-rule central` was added
  (hold out the training cube nearest the median training log-G0: edge folds
  hold out 0.8/0.4 and KEEP their neighbours; every fold keeps a one-step
  neighbour; assignments 0.1→0.8, 0.2→0.8, 0.4→0.8, 0.8→0.4, 1.6→0.4,
  3.2→0.4, 6.4→0.4).
- **Run 16b DONE** (2026-07-06, ~4.5 h) →
  `results/cnn_test_20260706_170201.json`
  (`checkpoint_selection: inner_val_central_log_g0`; log
  `logs/run16b_cnn_innerval_central.log`). **Central rule ADOPTED for the
  paper.** With the two-step-extrapolation confound removed, leakage-free
  selection costs little vs the old test-based selection: mean R²
  0.963 ± 0.047 (run 3: 0.970 ± 0.039), RMSE 0.348 (0.310), skill vs
  pointwise XGBoost +0.36 (mean-per-fold; run 3: +0.48). Per-fold R²
  0.896 / 0.992 / 0.993 / 0.991 / 0.995 / 0.992 / 0.882. **"Best
  interpolator" survives**: interior R² 0.992 / RMSE 0.223 still beats the
  shortcut mass-cal stack (0.984/0.31) in aggregate. Edges: R² 0.889 mean,
  skill −1.37 (G0=0.1) / −0.01 (6.4). Mass ratios remain uncontrolled and
  non-monotonic: 1.7 / 2.1 / 1.1 / 0.71 / 1.8 / 19.0 / **290.7** (clipped
  lower bound); bias_mol +1.13 at 6.4. Selection vs final epoch a no-op
  (0.9627 vs 0.9625). Nearest-rule run 16 kept as rule-sensitivity
  (0.914–0.963 across rules, disclosed in §4.2.3 + §7.5).
- **Run 16c DONE** (2026-07-06, ~4.5 h, 11-input run-4 config with central
  rule) → `results/cnn_test_20260706_205017.json` (log
  `logs/run16c_cnn_11input_central.log`). R² 0.851 ± 0.121 (run 4
  test-selected: 0.893 ± 0.088), RMSE 0.811, scatter 0.72, frac_01 0.15,
  R2_mol mean 0.06 (negative at both edges: −2.3 / −1.3), mass 0.29 to
  clipped ×513. §5.3 "morphology" bound updated 90 → 85 per cent.
- **Paper UPDATED 2026-07-06/07 with runs 16b+16c** (both `paper.tex` and
  the new `paper_short.tex`): Tables 2–3 U-Net rows, §4.2.3 checkpoint
  paragraph (inner-val protocol + rule sensitivity replaces the ≤12 %
  disclosure), §5.4, §5.3 11-input paragraph, §7.2, §7.5, abstract,
  conclusions (v)+(vi), Table 2 caption; `figures/fig_model_comparison.png`
  regenerated with `--cnn-log results/cnn_test_20260706_170201.json`
  (variant list matching the committed figure: xgb_standard,
  xgb_standard_sp, mlp_wide_sp, stacked_weighted, stacked_weighted_mwcal,
  unet_baseline). Both PDFs compile clean.
| 17 | Intra-cube random-mask seed repeats: rerun `intra_cube_section.py` rand splits with ≥3 seeds (add a `--seed` flag + loop, or a `--splits rand_*` subset flag) | a few hours | ChatGPT #9: one mask per fraction is not enough for a claim about random coverage. Current (run 11) rand_1 spread across cubes is 0.79–0.94, so instability is unlikely — this quantifies it |

### Analysis items (no GPU training — scripts over saved artifacts)

| # | Task | Time | Purpose |
|---|------|------|---------|
| A1 | Phase-threshold sensitivity: recompute phase-conditional metrics of the run-5 volumes for thresholds in [−5, −3] | minutes | **DONE 2026-07-05** — `phase_threshold_sensitivity.py` → `results/a1_phase_threshold_20260705_191051.json`. Smooth + monotone across [−5,−3]: R2_mol 0.976→0.995, R2_dif 0.881→0.932, f_mol 26→23 %; no qualitative change. Stability sentence added to §3.2 |
| A2 | Cell-level bootstrap CIs (within fold) for the deployed pipeline's headline metrics, from the run-5 npz + truth | minutes | **DONE 2026-07-05** — `bootstrap_cis.py` (200 reps, cell-level + 16³-block) → `results/a2_bootstrap_cis_20260705_191624.json`. Cell 95 % CIs ≤0.0002 in R²; block CIs ≲0.004 (≪ fold-to-fold σ = 0.006); mass-ratio block CIs the widest, all within [0.93, 1.14]. CI sentence added to §5.5; point estimates independently re-confirm `deployed_row_metrics.json` |
| A3 | Replot Figs 11–12 heatmaps from the saved run-10/11 JSONs with a diverging colormap clipped to [−1, 1] (values < −1 marked); enlarge in-cell text | ~1 h code | **DONE 2026-07-06** — `--replot JSON` mode added to `single_cube_extrapolation.py` and `intra_cube_section.py`; both PNGs regenerated in place from the run-10/11 JSONs (RdBu clipped to [−1, 1], white at 0, sub-−1 cells hatched with the true value printed, in-cell text enlarged to 8 pt); clipping note added to both captions; paper recompiles clean. No retraining |
| A4 | (Optional) Compare mass-weighted-residual calibration vs exact log-mass-ratio fitting on the run-5 volumes | minutes | **DONE 2026-07-06** — `calibration_functionals.py` → `results/a4_calibration_functionals_20260706_173659.json`. On the raw deployed volumes, b_M and the exact log-mass-ratio offset differ by ≤0.018 dex per fold (mean 0.009 dex); in-sample b_M mass closure within 4.3 %. Quantitative parenthetical added to §4.5 |
| A5 | Line-level numbers audit: every number in the abstract, §5, §7.3, and the conclusions checked against its source table/log | ~1 h | **DONE 2026-07-05.** ~200 numbers checked against run-1b/2b/3/4/deployed/run-10/run-11 logs (incl. every Table 2/3 cell, the 3-of-7 / 7-of-7 / 6-of-7 fold-count claims, and recomputed skill values). Four fixes applied to paper.tex: (1) §5.3 importance fractions 0.58/0.85 → **0.57/0.83** (recomputed from run-1b `xgb_feature_importance`; genuine error); (2) §5.2 "R² > 0.95 throughout" → "R² > 0.91 in every fold" (raw-stack fold minima are 0.916/0.927); (3) §5.4 U-Net "beating the stack on every one of them" → "in aggregate" (the stack wins interior folds G0=1.6 and 3.2 on both R² and RMSE); (4) conclusion (vi) "two orders of magnitude higher training cost" → "per-model training cost" (4.6 h vs ~3.5 min/model; vs the full 17-variant hour it is only ~5×). Paper recompiles clean. |

Done 2026-07-05 without new runs (from existing artifacts): multiplicity/
per-fold-consistency sentence (§5.1, from run-1b), calibration LOO
stability sentence (§5.2, from run-5 npz), clip-never-active verification
(§5.5 + Table 3 caption), deployed nested rows in Tables 2–3
(`results/deployed_row_metrics.json`).

## Verification items (no compute — ask the simulation collaborators)

Feed the answers into the paper's §2 wording (see PAPER_UPDATE_INSTRUCTIONS
"Audit-mandated disclosures"):

1. **Boundary conditions** of the MHD cubes (periodic?) → decides whether
   `reflect`/zero-padding is a disclosed approximation or simply wrong-free,
   and whether optional run 15 is worth doing. DESIGN_DECISIONS §5.1.
2. **UV field geometry** (directional along z vs isotropic) → justifies the
   z-preserving augmentation subgroup wording. DESIGN_DECISIONS §5.2.
3. **Box metadata** (physical size identical across G0, uniform cell
   volumes) → underpins the mass_ratio = mass interpretation; same source
   as review item C3 (physical units). DESIGN_DECISIONS §7.3.
4. **How f_H2 is computed** in the solver (from H2 column density?) →
   pins the exact wording of the solver-internal-quantity disclosure.
   DESIGN_DECISIONS §2.2.

## Optional reruns (only if the paper should quote new metrics there)

| # | Command | Time | Note |
|---|---------|------|------|
| 10 | `python -u single_cube_extrapolation.py > logs/run_scx.log` | ~30 min | §6.1. Old log + heatmap remain valid for the R²-only presentation; rerun records full metric dicts |
| 11 | `python -u intra_cube_section.py > logs/run_intra.log` | ~30 min | §6.2. New timestamped heatmap → update the figure path in paper.tex |

- **Run 10 DONE** (2026-07-05, ~30 min) → `logs/single_cube_extrapolation/run_20260705_114057.json` + heatmap.
  Consistent with the March matrix (off-diag stacked mean 0.7060 → 0.7076);
  only extreme-extrapolation corners move (≤0.06). §6.1 numbers updated in
  paper.tex (steps 3/4/5/6: 0.732/0.457/0.171/−0.013; off-diag mean 0.708);
  Fig 9 path updated.
- **Run 11 DONE** (2026-07-05, ~30 min) → `logs/intra_cube_section/run_20260705_121037.json` + heatmap.
  ⚠ **The March §6.2 log (`run_20260313_142443`) is NOT reproducible by any
  committed code**: the three 2026-03-13 runs are mutually inconsistent (the
  script was edited between them) and the rerun instead reproduces the
  12:44 run's behaviour, which matches the committed code AND the masked-filter
  convention described in the paper. New picture: slabs no longer fail in
  log-space R² (mean 0.85–0.89, worst 0.42) but DO fail on the primary
  metrics (RMSE ≈0.86 dex, frac_01 0.07–0.18, R2_mol ≤0.13, mass ratios
  0.006–4.4; small boxes predict essentially zero mass); rand_1 is stable
  at 0.90 (0.79–0.94) though with RMSE 0.69/mass 0.52–2.5; random ≥10 %
  preserves everything (RMSE 0.24, mass 0.82–1.23). §6.2 (now with the
  full metric hierarchy), Fig 10 caption+path, abstract and conclusion
  (vii) rewritten accordingly; §6.1 also quotes RMSE/bias/mass by step.
  Paper recompiled clean (2026-07-05).

## Figures to produce

Static / data-independent (no rerun needed):
- `figures/nH2_histograms.png` (`plot_nH2_histograms.py`) — paper Fig 1. Data unchanged.
- `fig_method_diagram.png` — paper Fig 2. Static diagram.

From run 6 (`figures/`):
- `fig2_scatter.png` — paper Fig 4 (pred vs truth per fold)
- `fig3_error_dist.png` — paper Fig 5 (residual distributions)
- `fig4_stratified.png` — paper Fig 6 (density-stratified errors)
- `fig6_distributions.png` — paper Fig 7 (marginals; caption must cite KS/W1,
  not KL — the old caption is stale)
- `fig7_massbudget.png` — **NEW**: mass ratio + bias, raw vs recalibrated,
  + phase-conditional R² per fold
- `fig8_slices.png` — paper Fig 8 (truth | prediction | error mid-plane
  slices; regenerated properly — the old fig5_spatial projections no longer
  match the paper caption)
- `fig1_summary.png` is still produced but should be DROPPED from the paper
  (bar charts of table numbers), replaced by:

Standalone figure scripts (after run 1):
- `python plot_model_comparison.py` → `figures/fig_model_comparison.png`
  — **NEW**, replaces paper Fig 3. Use `--r2-min 0.85` if a weak variant
  compresses the axis; `--variants ...` to select rows.
- `python plot_feature_importance.py` → `figures/fig_feature_importance.png`
  — **NEW** (needs the run-1 JSON; older logs have no importance blocks).

From runs 10/11 (only if rerun):
- `logs/single_cube_extrapolation/heatmap_<ts>.png` — paper Fig 9
- `logs/intra_cube_section/run_<ts>_heatmap.png` — paper Fig 10
  (update the hardcoded timestamped paths in paper.tex!)

## Bookkeeping

- After all runs finish, record the JSON-file → table/figure mapping in
  `paper/PAPER_UPDATE_INSTRUCTIONS.md` (section "Result sources") so every
  quoted number is traceable.
- Do not delete the old logs — unchanged claims (U-Net sensitivity, §6.1/§6.2
  if not rerun) still cite them.
