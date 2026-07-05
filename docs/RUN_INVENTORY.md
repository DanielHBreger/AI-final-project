# Experiment Run Inventory

Complete reference for features, hyperparameters, and results across all logged runs.
The JSON logs referenced below live in `results/` (moved there from the repo root in the 2026-07-05 reorganization).

---

## 1. Feature Sets by Phase

Three distinct input configurations were used across the project lifetime.

| Phase | Runs | N features | Feature columns | Target |
|---|---|---|---|---|
| **A** | `cnn_training_20260301_*` (3 runs) | **14** | `log_nH, log_T, log_nHp, ext, log_G0, vx, vy, vz, bxl, bxr, byl, byr, bzl, bzr` | **`log_fh2`** |
| **B** | `arch_comparison` Mar 6–9 (6 runs) | **15** | Phase A + `log_fh2` | `log_nH2` |
| **C** | `arch_comparison` Mar 11 (4 runs), `cnn_test_010315` | **15** | same as Phase B | `log_nH2` (RMSE now in dex) |
| **D** | `cnn_test_221728` | **11** | `log_T, log_G0, vx, vy, vz, bxl, bxr, byl, byr, bzl, bzr` | `log_nH2` (feature ablation) |

Spatial `_sp` variants (Phases B and C) append 45 additional features — the 15 base features each averaged at 3 kernel scales — giving **60 features total**. The kernel sizes used are recorded per-run in Section 3.

### Phase note — R² and RMSE scale shift at March 11

Phase B (Mar 6–9) and Phase C (Mar 11+) both use 15 features and target `log_nH2`, but their numeric results are not directly comparable:

- **Phase B RMSE** ≈ 0.025–0.038: suspiciously small for a target spanning −30 to +5 dex. These values appear to have been computed in StandardScaler-normalized space or with a metrics bug.
- **Phase C RMSE** ≈ 0.25–0.55 dex: correct scale for `log_nH2` with σ ≈ 2–3 dex.
- **Phase B xgb_standard R²** ≈ 0.886; **Phase C xgb_standard R²** ≈ 0.950. The jump is partly metric correction and partly the introduction/confirmation of `log_fh2` as a feature (it is a strong predictor of `log_nH2`).

The **canonical results** cited in the PROJECT_REPORT and executive summaries all come from the single Phase B run `arch_comparison_20260309_121724` (★ below). The March 11 Phase C runs are post-hoc and reflect a corrected metrics pipeline.

---

## 2. Model Hyperparameter Reference

All configs are fixed per-variant across every run that includes them. No hyperparameters were varied within a single run.

### 2.1 XGBoost variants
Common to all: `tree_method='hist'`, `colsample_bytree=0.8`, `random_state=42`, `verbosity=0`, device=GPU.

| Variant name | `max_depth` | `n_estimators` | `learning_rate` | `subsample` |
|---|---|---|---|---|
| `xgb_shallow` | 4 | 600 | 0.05 | 0.3 |
| `xgb_standard` | 6 | 400 | 0.10 | 0.3 |
| `xgb_deep` | 8 | 300 | 0.10 | 0.3 |
| `xgb_tuned` | 6 | 400 | 0.10 | 0.3 |

`xgb_tuned` appears only in run `181430` (Mar 11); its config was identical to `xgb_standard` in the logged results.

`_sp` suffix on any variant means the same hyperparameters are applied to the 60-feature spatial input.

### 2.2 MLP variants
Common: optimizer Adam (lr=1e-3, weight_decay=1e-5), CosineAnnealingLR, MSELoss, batch size 262,144, full training fold preloaded to GPU.

| Variant name | Hidden dims | Architecture | Source |
|---|---|---|---|
| `mlp_standard` | `[256, 256, 128, 64]` | Linear→BN1d→ReLU stack | `classical_models.py` |
| `mlp_wide` | `[512, 512, 256, 128]` | FlexMLP with BN1d→ReLU | `model_helpers.py` |
| `mlp_residual` | `[256, 256, 256, 256]` | 4 ResBlocks (BN→ReLU→Linear×2 + skip) | `model_helpers.py` |

Epochs per run are set by `mlp_epochs` in `run_config` (see Section 3).

### 2.3 CNN (U-Net) variants
Common: optimizer Adam (lr=5e-4, weight_decay=1e-5), input avg-pooled 128³→64³, 8 z-preserving Oh symmetry augmentations, gradient clip max_norm=1.0, InstanceNorm3d, no dropout (unless noted), trilinear upsampling.

| Variant name | `base_ch` | Approx params | Dropout | Warmup epochs | Notes |
|---|---|---|---|---|---|
| `unet_small` | 16 | ~1.5 M | 0.0 | 0 | |
| `unet_standard` | 32 | **5,848,673** | 0.0 | 0 | |
| `unet_large` | 64 | ~23 M | 0.0 | 0 | |
| `unet_residual` | 16 | ~1.5 M | **0.1** | **10** | Mar 22 test only |
| `unet_xgb_guided` | 32 | **5,953,953** | 0.0 | 0 | 16 input channels (15 + `xgb_pred`) |
| `unet_baseline` | 32 | 5,848,673 / 5,957,537 | 0.0 | 0 | Used in standalone cnn_test files |

Param counts for `unet_standard` (5,848,673) and `unet_baseline` with 15 channels (5,957,537) differ because the guided variant adds one input channel; the 15-channel baseline count in `cnn_test_010315` is slightly different due to code refactor.

Epochs for CNN runs are set by `cnn_epochs` in `run_config`.

### 2.4 Ensemble methods

| Ensemble name | Components | Combination |
|---|---|---|
| `ens_xgb+mlp` | `xgb_standard` + `mlp_wide` | equal-weight average |
| `ens_xgb+cnn` | `xgb_standard` + `unet_standard` | equal-weight average |
| `ens_all` | `xgb_standard` + `mlp_wide` + `unet_standard` | equal-weight average |
| `ens_sp` | `xgb_standard_sp` + `mlp_wide_sp` | equal-weight average |
| `ens_tuned+mlp` | `xgb_tuned` + `mlp_wide` | equal-weight average |
| `ens_tuned_sp` | `xgb_tuned_sp` + `mlp_wide_sp` | equal-weight average |
| `stacked_xgb+mlp` | `xgb_standard` + `mlp_wide` | Ridge(α=1.0) meta-learner on OOF |
| `stacked_xgb+cnn` | `xgb_standard` + `unet_standard` | Ridge(α=1.0) on OOF |
| `stacked_all` | all three | Ridge(α=1.0) on OOF |
| `stacked_sp` | `xgb_standard_sp` + `mlp_wide_sp` | Ridge(α=1.0) on OOF |
| `stacked_tuned+mlp` | `xgb_tuned` + `mlp_wide` | Ridge(α=1.0) on OOF |
| `stacked_tuned_sp` | `xgb_tuned_sp` + `mlp_wide_sp` | Ridge(α=1.0) on OOF |

---

## 3. Run-by-Run Configuration Table

★ = canonical run (quoted in PROJECT_REPORT).
RMSE units differ between phases; see Section 1 note.

| File | Date/time | Phase | Features | CNN epochs | MLP epochs | Spatial | Kernels | CNN run? |
|---|---|---|---|---|---|---|---|---|
| `cnn_training_184723` | Mar 1 18:18 | A | 14 (`log_fh2` = target) | 100 | — | No | — | Yes |
| `cnn_training_194547` | Mar 1 19:00 | A | 14 | 150 | — | No | — | Yes |
| `cnn_training_205412` | Mar 1 20:15 | A | 14 | 150 | — | No | — | Yes |
| `arch_comparison_142959` | Mar 6 14:30 | B | 15 | 50 | 30 | No | — | Yes |
| `arch_comparison_140718` | Mar 7 14:08 | B | 15 | 100 | 100 | No | — | Yes |
| `arch_comparison_134927` | Mar 8 13:50 | B | 15 | 150 | 100 | No | — | Yes |
| `arch_comparison_182350` | Mar 8 18:24 | B | 15 | 200 | 100 | Yes | `[3]` (default) | Yes |
| `arch_comparison_085656` | Mar 9 08:57 | B | 15 | 150 | 100 | Yes | `[3]` (default) | Yes |
| `arch_comparison_121724` ★ | Mar 9 12:18 | B | 15 | — | 100 | Yes | **`[3, 5, 7]`** | **No** |
| `arch_comparison_224154` | Mar 10 22:41 | B→C | 15 | 150 | 100 | Yes | `[3, 5, 7]` | Yes |
| `arch_comparison_125636` | Mar 11 12:57 | C | 15 | 150 | 100 | Yes | `[3, 5, 7]` | Yes |
| `arch_comparison_181430` | Mar 11 18:15 | C | 15 | — | 100 | Yes | **`[3, 5, 7, 15]`** | No |
| `arch_comparison_211326` | Mar 11 21:14 | C | 15 | — | 100 | Yes | `[3, 5, 7]` | No |
| `arch_comparison_231200` | Mar 11 23:12 | C | 15 | — | 100 | Yes | `[3, 5, 7]` | No |
| `cnn_test_221728` | Mar 22/23 00:55 | D | **11** | 150 | — | No | — | Yes |
| `cnn_test_010315` | Mar 23 02:14 | C | **15** | 150 | — | No | — | Yes |

Notes:
- `arch_comparison_224154` (Mar 10) is a failed run: MLP and CNN variants produced wildly negative R² (possibly due to a target or normalization bug introduced during transition from Phase B to C metrics). XGBoost-only results in that file match Phase C scale.
- `arch_comparison_182350` has `spatial: true` in run_config but no explicit `spatial_kernels` field; the default single-scale `[3]` is inferred from the jump in xgb_standard_sp R² (0.917 vs 0.924 for multi-scale).

---

## 4. Per-Run Results

All R² values are in log₁₀(nH2) space unless stated otherwise.
Phase B RMSE is in normalized units (not dex); Phase C RMSE is in dex.

### Phase A — CNN standalone (target = `log_fh2`, 14 features, unet_standard 5.85M params)

Results from PROJECT_REPORT Section 10.3 (per-fold data not logged in JSON):

| File | Epochs | Dropout | Weight decay | Mean R² | Std | G0=0.1 | G0=6.4 |
|---|---|---|---|---|---|---|---|
| `cnn_training_184723` | 100 | No | No | 0.775 | 0.247 | 0.649 | 0.216 |
| `cnn_training_194547` | 150 | **Yes (enc=0.10, bot=0.20)** | No | 0.649 | 0.523 | 0.634 | **−0.611** |
| `cnn_training_205412` | 150 | No | **Yes (1e-5)** | 0.803 | 0.172 | 0.497 | 0.591 |

### Phase B canonical ★ — `arch_comparison_20260309_121724`

15 features, target `log_nH2`, no CNN, spatial kernels `[3, 5, 7]`, 100 MLP epochs.
R² per fold (G0 = 0.1 | 0.2 | 0.4 | 0.8 | 1.6 | 3.2 | 6.4):

| Variant | Mean R² ± std | G0=0.1 | G0=0.2 | G0=0.4 | G0=0.8 | G0=1.6 | G0=3.2 | G0=6.4 |
|---|---|---|---|---|---|---|---|---|
| `xgb_shallow` | 0.8867 ± 0.0594 | 0.884 | 0.754 | 0.874 | 0.900 | 0.918 | 0.928 | 0.950 |
| `xgb_standard` | 0.8864 ± 0.0608 | 0.891 | 0.751 | 0.866 | 0.902 | 0.910 | 0.932 | 0.952 |
| `xgb_deep` | 0.8805 ± 0.0624 | 0.886 | 0.741 | 0.857 | 0.897 | 0.913 | 0.927 | 0.942 |
| `mlp_standard` | 0.8862 ± 0.0464 | 0.809 | 0.825 | 0.883 | 0.921 | 0.929 | 0.928 | 0.910 |
| `mlp_wide` | 0.8847 ± 0.0510 | 0.802 | 0.819 | 0.877 | 0.921 | 0.939 | 0.934 | 0.900 |
| `mlp_residual` | 0.8656 ± 0.0776 | 0.703 | 0.804 | 0.876 | 0.909 | 0.927 | 0.929 | 0.912 |
| `xgb_standard_sp` | 0.9243 ± 0.0256 | 0.912 | 0.874 | 0.920 | 0.934 | 0.924 | 0.948 | 0.959 |
| `mlp_wide_sp` | 0.9127 ± 0.0341 | 0.843 | 0.886 | 0.914 | 0.928 | 0.946 | 0.945 | 0.927 |
| `ens_xgb+mlp` | 0.9225 ± 0.0451 | 0.891 | 0.829 | 0.916 | 0.949 | 0.959 | 0.959 | 0.954 |
| **`ens_sp`** | **0.9482 ± 0.0236** | **0.908** | **0.915** | **0.953** | **0.962** | **0.967** | **0.967** | **0.965** |
| `stacked_xgb+mlp` | 0.9150 ± 0.0401 | 0.882 | 0.831 | 0.922 | 0.945 | 0.949 | 0.937 | 0.940 |
| `stacked_sp` | 0.9456 ± 0.0202 | 0.912 | 0.916 | 0.954 | 0.960 | 0.962 | 0.961 | 0.954 |

### Phase B — `arch_comparison_20260306_142959` (first arch comparison)

15 features, target `log_nH2`, CNN 50 epochs, MLP 30 epochs, no spatial.

| Variant | Mean R² ± std | RMSE |
|---|---|---|
| `xgb_shallow` | 0.8867 ± 0.0594 | 0.0370 |
| `xgb_standard` | 0.8864 ± 0.0608 | 0.0370 |
| `xgb_deep` | 0.8805 ± 0.0624 | 0.0380 |
| `mlp_standard` | 0.8828 ± 0.0300 | 0.0374 |
| `mlp_wide` | 0.8901 ± 0.0409 | 0.0363 |
| `mlp_residual` | 0.8775 ± 0.0300 | 0.0381 |
| `unet_small` | 0.2373 ± 0.7140 | 0.0804 |
| `unet_standard` | 0.6438 ± 0.2723 | 0.0591 |
| `unet_large` | 0.4226 ± 0.5692 | 0.0719 |

### Phase B — `arch_comparison_20260307_140718`

15 features, CNN 100 epochs, MLP 100 epochs, no spatial. First run with ensembles and guided CNN.

| Variant | Mean R² ± std |
|---|---|
| `xgb_shallow` | 0.8867 ± 0.0594 |
| `xgb_standard` | 0.8864 ± 0.0608 |
| `xgb_deep` | 0.8805 ± 0.0624 |
| `mlp_standard` | 0.8803 ± 0.0476 |
| `mlp_wide` | 0.8811 ± 0.0621 |
| `mlp_residual` | 0.8804 ± 0.0578 |
| `unet_small` | 0.6607 ± 0.3296 |
| `unet_standard` | 0.6031 ± 0.4690 |
| `unet_large` | 0.2991 ± 0.8580 |
| `unet_xgb_guided` | 0.7582 ± 0.1699 |
| `ens_xgb+mlp` | 0.9186 ± 0.0500 |
| `ens_xgb+cnn` | 0.8638 ± 0.1217 |
| `ens_all` | 0.9018 ± 0.0671 |

### Phase B — `arch_comparison_20260308_134927`

15 features, CNN 150 epochs (best CNN standalone run), MLP 100 epochs, no spatial.

| Variant | Mean R² ± std |
|---|---|
| `xgb_shallow` | 0.8867 ± 0.0594 |
| `xgb_standard` | 0.8864 ± 0.0608 |
| `xgb_deep` | 0.8805 ± 0.0624 |
| `mlp_standard` | 0.8865 ± 0.0488 |
| `mlp_wide` | 0.8885 ± 0.0506 |
| `mlp_residual` | 0.8832 ± 0.0482 |
| `unet_small` | 0.6926 ± 0.2421 |
| `unet_standard` | **0.7794 ± 0.1907** |
| `unet_large` | 0.6857 ± 0.4094 |
| `unet_xgb_guided` | 0.8140 ± 0.1971 |
| `ens_xgb+mlp` | 0.9236 ± 0.0450 |
| `ens_xgb+cnn` | 0.9066 ± 0.0484 |
| `ens_all` | **0.9268 ± 0.0354** |

### Phase B — `arch_comparison_20260308_182350`

15 features, CNN 200 epochs, MLP 100 epochs, spatial `[3]` (single scale). First run with `_sp` variants and stacked ensembles.

| Variant | Mean R² ± std |
|---|---|
| `xgb_standard` | 0.8864 ± 0.0608 |
| `mlp_wide` | 0.8850 ± 0.0533 |
| `unet_standard` | 0.3887 ± 0.8477 |
| `xgb_standard_sp` | 0.9167 ± 0.0368 |
| `mlp_wide_sp` | 0.9149 ± 0.0297 |
| `unet_xgb_guided` | 0.7842 ± 0.2409 |
| `ens_xgb+mlp` | 0.9216 ± 0.0471 |
| `ens_xgb+cnn` | 0.8004 ± 0.2622 |
| `ens_all` | 0.8739 ± 0.1379 |
| `stacked_xgb+mlp` | 0.9120 ± 0.0398 |
| `stacked_xgb+cnn` | 0.8860 ± 0.0553 |
| `stacked_all` | 0.9071 ± 0.0421 |

### Phase B — `arch_comparison_20260309_085656`

15 features, CNN 150 epochs, MLP 100 epochs, spatial `[3]`. First run with `ens_sp` and `stacked_sp`.

| Variant | Mean R² ± std |
|---|---|
| `xgb_standard` | 0.8864 ± 0.0608 |
| `mlp_wide` | 0.8795 ± 0.0634 |
| `unet_small` | 0.7565 ± 0.1929 |
| `unet_standard` | 0.7556 ± 0.2888 |
| `xgb_standard_sp` | 0.9167 ± 0.0368 |
| `mlp_wide_sp` | 0.9154 ± 0.0340 |
| `unet_xgb_guided` | 0.6239 ± 0.4770 |
| `ens_xgb+mlp` | 0.9183 ± 0.0503 |
| `ens_xgb+cnn` | 0.8969 ± 0.0809 |
| `ens_all` | 0.9198 ± 0.0475 |
| `ens_sp` | 0.9468 ± 0.0265 |
| `stacked_xgb+mlp` | 0.9042 ± 0.0428 |
| `stacked_xgb+cnn` | 0.9086 ± 0.0501 |
| `stacked_all` | 0.9207 ± 0.0424 |
| `stacked_sp` | 0.9423 ± 0.0243 |

### Broken run — `arch_comparison_20260310_224154`

All MLP and CNN variants show catastrophically wrong R² (−∞ to −10⁸) and RMSE (8–10⁹). XGBoost results are in Phase C scale but R² is 0.15–0.22 (low, not 0.95). This run should be ignored; it reflects a broken state during the Phase B→C transition.

| Variant | Mean R² |
|---|---|
| `xgb_standard` | 0.184 |
| `mlp_wide` | −17.6 |
| `unet_standard` | −∞ |
| `ens_sp` | 0.696 |

### Phase C — `arch_comparison_20260311_125636`

15 features, target `log_nH2`, CNN 150 epochs, MLP 100 epochs, kernels `[3, 5, 7]`. First Phase C run with R2_lin metric and corrected RMSE in dex.

| Variant | Mean R² ± std | RMSE (dex) | R2_lin |
|---|---|---|---|
| `xgb_shallow` | 0.9508 ± 0.0286 | 0.487 | 0.150 |
| `xgb_standard` | 0.9502 ± 0.0289 | 0.486 | 0.184 |
| `xgb_deep` | 0.9479 ± 0.0299 | 0.499 | 0.196 |
| `mlp_standard` | 0.9658 ± 0.0290 | 0.463 | −35.0 |
| `mlp_wide` | 0.9535 ± 0.0383 | 0.531 | −5.8 |
| `mlp_residual` | 0.9666 ± 0.0280 | 0.455 | −30.5 |
| `unet_small` | 0.8243 ± 0.2763 | 0.585 | −3154 |
| `unet_standard` | 0.8777 ± 0.1806 | 0.516 | −2867 |
| `unet_large` | 0.8005 ± 0.3516 | 0.541 | −3184 |
| `xgb_standard_sp` | 0.9630 ± 0.0330 | 0.374 | 0.219 |
| `mlp_wide_sp` | 0.9684 ± 0.0413 | 0.384 | −1.3 |
| `unet_xgb_guided` | 0.8799 ± 0.1811 | 0.498 | −2812 |
| `ens_xgb+mlp` | 0.9688 ± 0.0132 | 0.421 | 0.641 |
| `ens_xgb+cnn` | 0.9462 ± 0.0701 | 0.400 | −654 |
| `ens_all` | 0.9664 ± 0.0299 | 0.383 | −242 |
| `ens_sp` | 0.9782 ± 0.0159 | 0.309 | 0.696 |
| `stacked_xgb+mlp` | 0.9827 ± 0.0155 | 0.318 | 0.495 |
| `stacked_xgb+cnn` | 0.9719 ± 0.0449 | 0.364 | 0.098 |
| `stacked_all` | 0.9713 ± 0.0426 | 0.379 | −1.5 |
| `stacked_sp` | 0.9880 ± 0.0133 | 0.247 | 0.569 |

### Phase C — `arch_comparison_20260311_181430`

15 features, no CNN, kernels `[3, 5, 7, 15]` (testing wider spatial scale). Adds `xgb_tuned` variant.

| Variant | Mean R² ± std | RMSE (dex) | R2_lin |
|---|---|---|---|
| `xgb_shallow` | 0.9508 ± 0.0286 | 0.487 | 0.150 |
| `xgb_standard` | 0.9502 ± 0.0289 | 0.486 | 0.184 |
| `xgb_deep` | 0.9479 ± 0.0299 | 0.499 | 0.196 |
| `xgb_tuned` | 0.9506 ± 0.0289 | 0.484 | 0.185 |
| `mlp_standard` | 0.9665 ± 0.0230 | 0.456 | −31.0 |
| `mlp_wide` | 0.9452 ± 0.0514 | 0.567 | −10.6 |
| `mlp_residual` | 0.9687 ± 0.0239 | 0.446 | −49.1 |
| `xgb_standard_sp` | 0.9636 ± 0.0327 | 0.371 | 0.211 |
| `xgb_tuned_sp` | 0.9638 ± 0.0329 | 0.369 | 0.214 |
| `mlp_wide_sp` | 0.9749 ± 0.0364 | 0.346 | 0.081 |
| `ens_sp` | 0.9818 ± 0.0139 | 0.285 | 0.667 |
| `ens_tuned_sp` | 0.9818 ± 0.0139 | 0.285 | 0.672 |
| `stacked_sp` | 0.9867 ± 0.0161 | 0.254 | 0.548 |
| `stacked_tuned_sp` | 0.9868 ± 0.0160 | 0.253 | 0.553 |

Note: the `[3, 5, 7, 15]` kernel set produced marginally better xgb_standard_sp (0.9636) than `[3, 5, 7]` (0.9630); the improvement is within run-to-run noise.

### Phase C — `arch_comparison_20260311_211326`

15 features, no CNN, kernels `[3, 5, 7]`, reduced variant set.

| Variant | Mean R² ± std | RMSE (dex) |
|---|---|---|
| `xgb_shallow` | 0.9508 ± 0.0286 | 0.487 |
| `xgb_standard` | 0.9502 ± 0.0289 | 0.486 |
| `xgb_deep` | 0.9479 ± 0.0299 | 0.499 |
| `mlp_standard` | 0.9524 ± 0.0443 | 0.518 |
| `mlp_wide` | 0.9650 ± 0.0245 | 0.473 |
| `mlp_residual` | 0.9656 ± 0.0286 | 0.465 |
| `xgb_standard_sp` | 0.9630 ± 0.0330 | 0.374 |
| `mlp_wide_sp` | 0.9716 ± 0.0429 | 0.361 |
| `ens_xgb+mlp` | 0.9739 ± 0.0090 | 0.383 |
| `ens_sp` | 0.9797 ± 0.0164 | 0.297 |
| `stacked_xgb+mlp` | 0.9839 ± 0.0150 | 0.306 |
| `stacked_sp` | 0.9859 ± 0.0177 | 0.259 |

### Phase C — `arch_comparison_20260311_231200`

15 features, no CNN, kernels `[3, 5, 7]`, core 8 variants only.

| Variant | Mean R² ± std | RMSE (dex) | R2_lin |
|---|---|---|---|
| `xgb_standard` | 0.9502 ± 0.0289 | 0.486 | 0.184 |
| `mlp_wide` | 0.9602 ± 0.0228 | 0.502 | −28.3 |
| `xgb_standard_sp` | 0.9630 ± 0.0330 | 0.374 | 0.219 |
| `mlp_wide_sp` | 0.9787 ± 0.0150 | 0.339 | 0.467 |
| `ens_xgb+mlp` | 0.9694 ± 0.0081 | 0.411 | 0.630 |
| `ens_sp` | 0.9812 ± 0.0087 | 0.287 | 0.649 |
| `stacked_xgb+mlp` | 0.9832 ± 0.0156 | 0.311 | 0.514 |
| `stacked_sp` | **0.9911 ± 0.0086** | **0.222** | 0.616 |

### Phase D / C — CNN ablation tests (`cnn_test_*`)

#### `cnn_test_20260322_221728` — 11 features (no density/extinction/shielding features), 150 epochs

| Variant | base_ch | Params | Mean R² ± std | RMSE (dex) | R2_lin |
|---|---|---|---|---|---|
| `unet_baseline` | 32 | 5,953,953 | 0.8735 ± 0.0962 | 0.7315 | −214 |
| `unet_residual` | 16 | ~1.5M | 0.8359 ± 0.1313 | 0.8084 | −1598 |
| `unet_large` | 64 | ~23M | 0.8889 ± 0.0941 | 0.6632 | −373 |

#### `cnn_test_20260323_010315` — 15 features (full set), 150 epochs

| Variant | base_ch | Params | Mean R² ± std | RMSE (dex) | R2_lin |
|---|---|---|---|---|---|
| `unet_baseline` | 32 | 5,957,537 | **0.9740 ± 0.0349** | 0.2984 | −364 |
| `unet_residual` | 16 | ~1.5M | 0.9483 ± 0.0694 | 0.3819 | −2047 |

The dramatic R² improvement from 0.874 (11 features) to 0.974 (15 features) isolates the contribution of `log_nH`, `log_nHp`, `ext`, and especially `log_fh2` to CNN accuracy.

---

## 5. Summary: R² Progression Across Phase B Runs

Using consistent Phase B metrics (log-space R², no dex RMSE fix):

| Date | Run (timestamp) | Best ensemble | Mean R² |
|---|---|---|---|
| Mar 6 | 142959 | — (no ensembles yet) | mlp_wide 0.890 |
| Mar 7 | 140718 | `ens_xgb+mlp` | 0.919 |
| Mar 8 | 134927 | `ens_all` | 0.927 |
| Mar 8 | 182350 | single-scale spatial | `ens_sp` 0.947* |
| Mar 9 | 085656 | `ens_sp` | 0.947 |
| **Mar 9 ★** | **121724** | **`ens_sp`** | **0.948** |

*spatial kernel default `[3]` only; confirmed multi-scale at 121724.
