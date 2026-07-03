# Run plan — regenerating all paper data and figures

Written 2026-07-03, after the metrics overhaul (full metric suite recorded by
`compute_metrics`: RMSE = bias ⊕ scatter, mass_ratio, frac_01/03/05,
phase-conditional R²/RMSE/bias, MAE_mw, CCC, W1, skill_vs_xgb) and the figure
overhaul (fig7/fig8 + two standalone figure scripts). Old JSON logs remain
*valid* but lack the new metric keys — that is why the reruns are needed.

## Before starting

- Data: `icedrive-dl-182bd/UVonly/<G0>/`; loading all 7 cubes takes several
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
| 1 | `python -u compare_architectures.py > run_main.log` | ~2.5 h | `arch_comparison_<ts>.json` | Table 1 (all metrics + skill), model-comparison figure, feature-importance figure |
| 2 | `python -u compare_architectures.py --no-fh2 --log ablation_nofh2_<date>.json > run_nofh2.log` | ~2.5 h | ablation JSON | Table 2 (f_H2 ablation) |
| 3 | `python -u test_cnn.py --variants unet_baseline > run_cnn.log` | hours | `cnn_test_<ts>.json` | U-Net row of Table 1 |
| 4 | `python -u test_cnn.py --no-fh2 --no-nH --no-nHp --no-ext --variants unet_baseline > run_cnn11.log` | hours | `cnn_test_<ts>.json` | 11-input U-Net upper-bound number (§5.2) |
| 5 | `python -u predict_and_visualize.py --all > run_pred.log` | ~2–3 h | `predictions/pred_g0_*_<ts>.npz` (raw + recalibrated volumes, `metrics_json`) | §5.4 error analysis; closes review items D2 (full 100-epoch schedule) and D3 (bias recalibration) |
| 6 | `python statistical_analysis.py --pred-dir predictions --save-dir analysis_output` | ~10 min | `analysis_output/fig1..fig8` | Figures (see below) |
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
- **Run 2b TODO**: `python -u compare_architectures.py --no-fh2 --log
  ablation_nofh2_128eval.json > run_nofh2_2b.log` (~1 h). Run 2's
  ensemble/stacked rows are 64³-pooled (it launched before `_align_preds`;
  its rows bit-match the 7/1 ablation). Run 2b gives the no-fh2
  ensemble/stacked/mwcal rows evaluated at native 128³ for a consistent
  Table 2. Pointwise rows must bit-match run 2.
- Run 5 note: `predict_and_visualize.py` now applies **mass-weighted**
  recalibration by default (`--recal-mode mass`); the legacy cell-mean
  correction is `--recal-mode mean`, off is `--recal-mode off`. Both fits
  are recorded in each npz regardless of mode.

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
| 8 | `python -u compare_architectures.py --no-vel --no-B --log ablation_novelB_<date>.json > run_novelB.log` | ~2.5 h | Minimal deployable feature set (6 local features). Expected: little accuracy loss → new result + closes review item D4 together with the importance figure |
| 9 | z-column cumulative-sum features — **code not yet written** (add a cumsum-along-z variant to `model_helpers._compute_spatial_X`, expose as `--spatial-mode zcum` or similar), then one comparison run and one `--no-fh2` run with it | ~0.5 day | Directional (physics-informed) vs isotropic non-locality; strongest candidate headline for the methods contribution |

## Optional reruns (only if the paper should quote new metrics there)

| # | Command | Time | Note |
|---|---------|------|------|
| 10 | `python -u single_cube_extrapolation.py > run_scx.log` | hours | §6.1. Old log + heatmap remain valid for the R²-only presentation; rerun records full metric dicts |
| 11 | `python -u intra_cube_section.py > run_intra.log` | hours | §6.2. Same note. New timestamped heatmap → update the figure path in paper.tex |

## Figures to produce

Static / data-independent (no rerun needed):
- `nH2_histograms.png` (`plot_nH2_histograms.py`) — paper Fig 1. Data unchanged.
- `fig_method_diagram.png` — paper Fig 2. Static diagram.

From run 6 (`analysis_output/`):
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
- `python plot_model_comparison.py` → `analysis_output/fig_model_comparison.png`
  — **NEW**, replaces paper Fig 3. Use `--r2-min 0.85` if a weak variant
  compresses the axis; `--variants ...` to select rows.
- `python plot_feature_importance.py` → `analysis_output/fig_feature_importance.png`
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
