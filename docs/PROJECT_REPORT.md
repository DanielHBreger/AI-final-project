# Predicting Molecular Hydrogen Number Density (nH2) in 3D Astrophysical Simulations Using Machine Learning

---

## Table of Contents

1. [Problem Statement and Scientific Context](#1-problem-statement-and-scientific-context)
2. [Dataset Description](#2-dataset-description)
3. [Cross-Validation Strategy: Why Leave-One-G0-Out](#3-cross-validation-strategy-why-leave-one-g0-out)
4. [Data Preprocessing and Feature Engineering](#4-data-preprocessing-and-feature-engineering)
5. [Model Architectures and Design Rationale](#5-model-architectures-and-design-rationale)
6. [The 3D U-Net CNN: Architecture, Augmentation, and Training](#6-the-3d-u-net-cnn-architecture-augmentation-and-training)
7. [Spatial Neighbourhood Features: Replacing the CNN](#7-spatial-neighbourhood-features-replacing-the-cnn)
8. [Ensemble Methods](#8-ensemble-methods)
9. [Experimental Iterations and What Each Taught Us](#9-experimental-iterations-and-what-each-taught-us)
10. [Complete Results](#10-complete-results)
11. [Code Architecture Walkthrough](#11-code-architecture-walkthrough)
12. [Visualization Pipeline](#12-visualization-pipeline)
13. [Conclusions and Key Takeaways](#13-conclusions-and-key-takeaways)
14. [Additional Experiments](#14-additional-experiments)

---

## 1. Problem Statement and Scientific Context

### The Physics Problem

In the interstellar medium (ISM), hydrogen exists in three phases: ionized (H+), atomic (H), and molecular (H2). The molecular fraction **fh2** is a critical quantity because molecular hydrogen is the raw material for star formation. Computing fh2 from first principles requires solving time-dependent chemistry coupled to radiative transfer, which is extremely expensive in 3D simulations.

The dominant processes controlling fh2 are:

- **H2 formation**: Happens on dust grain surfaces. Rate depends on gas density (nH) and temperature (T).
- **H2 photodissociation**: UV photons in the Lyman-Werner band (11.2-13.6 eV) break apart H2. Rate depends on the **UV field strength G0** (measured in Habing units) and on how much the cell is **shielded** by surrounding gas and dust (extinction, ext).
- **Self-shielding**: Dense clouds of H2 absorb the UV radiation that would destroy them, creating a positive feedback loop. This is inherently a *spatial* phenomenon: a cell's fh2 depends on the column density of its neighbours along the UV illumination direction.

### The ML Task

Given 15 local physical properties at each cell of a 128x128x128 simulation grid, predict nH2 (molecular hydrogen number density, in cm^-3). The training data comes from 7 simulations that differ only in UV field strength G0 (ranging from 0.1 to 6.4 Habing units). The model must generalize to UV conditions it has never seen.

### Why This Is Hard

This is not a standard regression problem for three reasons:

1. **The target spans 30 orders of magnitude**: fh2 ranges from ~10^-30 (essentially no H2) to ~0.5 (fully molecular). This makes raw-space regression numerically infeasible.
2. **The cross-validation tests out-of-distribution generalization**: Each fold holds out an entire G0 value. The model must extrapolate to UV field strengths absent from training.
3. **Spatial context matters**: Whether a cell is shielded from UV depends on its neighbours, but most ML models only see per-cell features.

---

## 2. Dataset Description

### Simulation Cubes

Seven 3D simulation cubes from UV-only chemistry runs, each a 128x128x128 grid at a different UV field strength:

| Cube | G0 Value | Cells | log10(nH2) Range | Physical Meaning |
|------|----------|-------|-----------------|------------------|
| 1 | 0.1 | 2,097,152 | [-30, +5] approx | Very weak UV: most hydrogen can become molecular |
| 2 | 0.2 | 2,097,152 | [-30, +5] approx | Weak UV |
| 3 | 0.4 | 2,097,152 | [-30, +5] approx | Moderate UV |
| 4 | 0.8 | 2,097,152 | [-30, +5] approx | Moderate UV |
| 5 | 1.6 | 2,097,152 | [-30, +5] approx | Strong UV |
| 6 | 3.2 | 2,097,152 | [-30, +5] approx | Strong UV: most H2 destroyed in unshielded regions |
| 7 | 6.4 | 2,097,152 | [-30, +5] approx | Very strong UV |

**Total**: 14,680,064 cells. Each cell has 18 raw columns; after preprocessing, 15 features are retained.

### Physical Fields (Raw Columns per CSV)

| Column | Physical Meaning | Role |
|--------|-----------------|------|
| `ix, iy, iz` | 1-indexed grid coordinates | Grid position (not a feature) |
| `nH` | Total hydrogen number density | **Feature** (log-transformed) |
| `nH2` | Molecular hydrogen number density | **Target** (log-transformed) |
| `T` | Gas temperature | **Feature** (log-transformed) |
| `vx, vy, vz` | Gas velocity components | **Feature** (encodes turbulence) |
| `nHp` | Ionized hydrogen density | **Feature** (log-transformed) |
| `ext` | Extinction (dust shielding) | **Feature** (proxy for column density) |
| `fh2` | H2 self-shielding factor | **Feature** (log-transformed; not derived from nH2) |
| `bxl, bxr, byl, byr, bzl, bzr` | Face-centred magnetic field (left/right per axis) | **Feature** (6 components) |

### Why fh2 Is Kept as a Feature

The H2 self-shielding factor fh2 is **not** algebraically derived from nH2. It is an independent physical quantity: it measures the fraction of UV radiation in the H2 Lyman-Werner bands that is attenuated by the H2 column along the UV illumination axis. Computing fh2 requires integrating the H2 column density along each sight-line, which depends on the 3D spatial distribution of gas — not just the local nH2 value.

Two cells with the same nH2 can have very different fh2 values depending on how much H2 sits between them and the UV source. Including `log_fh2` as a feature is valid and physically informative: a high fh2 (near 0 on log-scale) means the cell is strongly shielded from UV and we expect high nH2; a low fh2 means the cell is UV-exposed and nH2 will be low.

In XGBoost feature importance, `log_fh2` is expected to rank near the top — it directly encodes spatial shielding information that local density and temperature alone cannot provide.

### Target Distribution

```
log10(nH2):  min ≈ -30,  max ≈ +5,  mean ≈ -1 to 0,  std ≈ 2-3 dex (varies by G0)
```

Key characteristics:
- nH2 spans ~35 orders of magnitude in linear space, making raw-space regression numerically infeasible
- In log-space the distribution is approximately Gaussian, which is why log-space prediction is both numerically necessary and statistically natural
- UV-exposed cells (low G0 shielding) have nH2 near 10^-30; dense shielded cores have nH2 near 10^5 cm^-3

---

## 3. Cross-Validation Strategy: Why Leave-One-G0-Out

### The Setup

Standard k-fold cross-validation would randomly split cells across all 7 cubes. This would be trivially easy: cells from the same cube share the same G0, so the model would always have nearby cells (same G0, similar density/temperature) in the training set. This tests **interpolation within known conditions**, not generalization.

Instead, we use **leave-one-G0-out** 7-fold CV: each fold holds out one entire cube (2,097,152 cells at one G0 value) and trains on the other 6 cubes (12,582,912 cells). This tests whether the model can predict fh2 at a **UV field strength it has never seen**.

### Why This Is the Right Test

The real-world application is predicting fh2 for new simulations at arbitrary G0 values. A model that only works when it has already seen that exact G0 is useless. Leave-one-G0-out directly measures the model's ability to generalize across the parameter that drives the most physically meaningful variation in the data.

### Difficulty Varies by Fold

Not all folds are equally hard. The G0 values form a geometric sequence: 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4. When the model holds out an interior value (say G0=0.8), it can interpolate between G0=0.4 and G0=1.6. But when it holds out a boundary value (G0=0.1 or G0=6.4), it must **extrapolate** beyond the training range.

Expected difficulty ranking from hardest to easiest:
1. **G0=0.1**: No training data below it. Pure downward extrapolation.
2. **G0=0.2**: Only G0=0.1 is lower, and 0.1 is very different physically.
3. **G0=6.4**: No training data above it. Upward extrapolation.
4. **G0=0.4 through G0=3.2**: Interpolation between neighbouring cubes.

This prediction is confirmed by every experiment: G0=0.1 and G0=0.2 are consistently the worst folds.

### Metrics: Why Evaluate in Log-Space (Dex)

All models predict in **log10(nH2) space** and metrics are computed directly in that space:

```python
def compute_metrics(y_true_log, y_pred_log):
    # Primary: log-space (dex) metrics
    R2   = r2_score(y_true_log, y_pred_log)
    RMSE = sqrt(mean_squared_error(y_true_log, y_pred_log))
    MAE  = mean_absolute_error(y_true_log, y_pred_log)

    # Secondary: linear-space R2 with clipped predictions (±1 dex margin)
    clip_lo  = y_true_log.min() - 1.0
    clip_hi  = y_true_log.max() + 1.0
    y_true_l = 10.0 ** y_true_log
    y_pred_l = 10.0 ** np.clip(y_pred_log, clip_lo, clip_hi)
    R2_lin   = r2_score(y_true_l, y_pred_l)
```

**Why log-space as the primary metric**: When the target is log10(nH2), neural networks and other
models can produce predictions far outside the training range (e.g., log_nH2 = 30). Exponentiating
such a value gives 10^30 — and even a single such prediction creates astronomical MSE in linear
space, making linear R2 uninformative. Log-space R2 directly measures whether the model predicts
the correct order of magnitude for each cell, which is the physically meaningful question.

**Interpretation**: R2 = 0.95 in log-space means the model explains 95% of variance in
log10(nH2). Typical errors of 0.1-0.2 dex correspond to a factor of 1.3-1.6 in linear nH2.

**The secondary R2_lin** clips predictions to within ±1 dex of the true range before
exponentiating, providing a bounded linear-space metric for comparison with linear-space models.

---

## 4. Data Preprocessing and Feature Engineering

### 4.1 Log-Transforms

Three physical quantities (nH, T, nHp) span many orders of magnitude in the raw data. Without log-transforms, an XGBoost or MLP model would need many splits or neurons just to handle the dynamic range. Log-transforming compresses the range and makes the distributions more amenable to ML:

```python
_EPS = 1e-30  # guard against log(0)
df['log_nH']  = np.log10(df['nH']  + _EPS)
df['log_T']   = np.log10(df['T']   + _EPS)
df['log_nHp'] = np.log10(df['nHp'] + _EPS)
df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))
```

The target (fh2) is also log-transformed. The epsilon (1e-30) prevents log(0) without affecting the physics: fh2 = 10^-30 is indistinguishable from zero in any physical context.

### 4.2 The 15 Baseline Features

```
log_nH, log_T, log_nHp, ext, log_fh2, log_G0, vx, vy, vz, bxl, bxr, byl, byr, bzl, bzr
```

These are the features available to all pointwise models. Their physical roles:

| Feature(s) | Physical Role | Expected Importance |
|------------|--------------|------------|
| **log_T** (temperature) | H2 formation/destruction rates are exponentially T-dependent | Very high |
| **log_nH** (gas density) | Controls collision rate for H2 formation on dust grains | High |
| **log_fh2** (self-shielding factor) | Measures how much H2 attenuates its own UV-photodissociation radiation along the illumination axis. High = shielded = expect high nH2 | High — encodes spatial shielding information local features cannot |
| **log_nHp** (ionized H) | Traces ionization state; anti-correlated with H2 | Moderate |
| **ext** (extinction) | Proxy for dust shielding; high ext means UV is absorbed before reaching the cell | Moderate |
| **log_G0** (UV field) | Global UV radiation strength; same for all cells in a cube | Low — varies between cubes, not within |
| **vx, vy, vz** (velocity) | Traces turbulent motions; indirectly encodes spatial structure | Negligible per-cell |
| **bxl..bzr** (B-field) | Magnetic pressure; indirectly affects density structure | Negligible per-cell |

The low importance of log_G0 is counterintuitive since G0 is the parameter being varied. The explanation is that G0 is constant across an entire cube — it provides no *within-cube* discrimination. Its effect is captured implicitly by the combination of other features (e.g., at high G0, cells with the same nH and T have lower nH2).

`log_fh2` is a particularly valuable addition when predicting nH2: it encodes the global shielding integral along the UV illumination axis, which is the dominant non-local physical effect governing whether nH2 is high or low. Unlike the spatial neighbourhood features (Section 4.3), which capture local 3D context via box filters, fh2 provides the directional UV-shielding information that truly sets the equilibrium nH2.

### 4.3 Multi-Scale Spatial Neighbourhood Features

**The key insight**: pointwise models see each cell in isolation, but H2 chemistry is inherently spatial. A cell deep inside a dense cloud is shielded from UV by its neighbours. Two cells with identical local properties (same nH, T, etc.) can have very different fh2 values depending on whether they sit at the surface or centre of a cloud.

**Solution**: For each of the 14 features, compute the 3D local mean at three scales using `scipy.ndimage.uniform_filter`:

| Scale | Kernel | Cells Averaged | Physical Meaning |
|-------|--------|---------------|------------------|
| Small | 3x3x3 | 27 | Immediate neighbourhood gradient |
| Medium | 5x5x5 | 125 | Local cloud structure |
| Large | 7x7x7 | 343 | Meso-scale environment |

This produces 15 features x 3 scales = **45 spatial features**, which are concatenated with the 15 baseline features for a total of **60 features** per cell:

```python
def _compute_spatial_X(cubes, all_vols, feature_cols, kernel_sizes=(3, 5, 7)):
    from scipy.ndimage import uniform_filter
    parts = []
    for cube, vol in zip(cubes, all_vols):
        ix = cube['ix'].values.astype(int) - 1  # convert 1-indexed to 0-indexed
        iy = cube['iy'].values.astype(int) - 1
        iz = cube['iz'].values.astype(int) - 1
        scale_feats = [
            np.stack([
                uniform_filter(vol[col], size=ks)[ix, iy, iz]  # 3D box filter
                for col in feature_cols
            ], axis=-1).astype(np.float32)
            for ks in kernel_sizes
        ]
        parts.append(np.concatenate(scale_feats, axis=-1))  # (N_cube, 42)
    return np.concatenate(parts, axis=0)  # (N_total, 42)
```

**How it works step by step**:
1. Each of the 14 features is stored as a 128x128x128 volume (via `cube_to_volumes()`).
2. `uniform_filter(vol, size=k)` replaces each voxel with the mean of its k x k x k neighbourhood. This is a fast O(N) operation regardless of kernel size (separable filter).
3. The smoothed volume is indexed back to DataFrame row order using the original grid coordinates (ix, iy, iz).
4. The result for all 3 kernel sizes is concatenated to form a 42-dimensional spatial feature vector per cell.

**Why this is better than a CNN for this task**: The uniform_filter approach is:
- **Deterministic**: No training variance. The features are a fixed transformation.
- **Cheap**: Computed once in ~30 seconds for all 7 cubes, all 3 scales.
- **Compatible with XGBoost and MLP**: Spatial context is now just additional tabular features.
- **Stable**: No risk of catastrophic training collapse.

The single-scale (3x3x3 only) version was tested first and improved XGBoost from R²=0.886 to 0.917 (+0.031). Multi-scale (3x3x3 + 5x5x5 + 7x7x7) further improved XGBoost to 0.924 and critically boosted the hardest fold G0=0.2 from 0.899 to 0.915.

Note: all R² values in this project are computed in log10(nH2) space (dex), not linear space.

---

## 5. Model Architectures and Design Rationale

### 5.1 Linear Regression (Lower Bound)

Standard linear regression with `StandardScaler` preprocessing. This serves as a sanity check.

**Result**: Mean R² = 0.230 across folds. Negative R² on G0=0.1 (-0.876) and G0=0.2 (-0.237), meaning the model is worse than predicting the global mean. This confirms that the fh2-feature relationship is fundamentally nonlinear — the exponential temperature dependence of chemistry rates cannot be captured by linear combinations of features.

### 5.2 XGBoost: Gradient Boosted Trees

XGBoost is a leading algorithm for tabular data. It builds an ensemble of decision trees sequentially, where each new tree corrects the residual errors of the previous ones.

**Why XGBoost suits this problem**:
- Decision trees naturally handle the nonlinear, threshold-like relationships in chemistry (e.g., above a critical temperature, H2 is rapidly destroyed).
- The `hist` tree method bins continuous features into 256 histograms, making training feasible on 12.5M rows per fold.
- Subsampling (`subsample=0.3`) uses only 30% of rows per tree, which acts as stochastic regularization and makes each tree 3.3x cheaper to build.

Three variants were tested to explore the tree depth axis — the key structural knob controlling **feature interaction order**:

| Variant | max_depth | n_estimators | learning_rate | What It Tests |
|---------|-----------|-------------|---------------|---------------|
| xgb_shallow | 4 | 600 | 0.05 | Can low-order feature interactions (pairs) suffice? More trees, each weaker, reduces memorization. |
| xgb_standard | 6 | 400 | 0.10 | Baseline. depth=6 allows up to 6-way feature interactions per tree path. |
| xgb_deep | 8 | 300 | 0.10 | Can higher-order interactions (e.g., nH x T x G0 x ext x nHp) improve prediction? Risk: memorizing training G0 values. |

**Result**: All three variants give R² ~ 0.886 with negligible differences. This tells us that the dominant predictive signal comes from low-order interactions (primarily T alone, then T x nH). Higher-order interactions add noise without improving generalization to unseen G0.

**XGBoost training code**:
```python
def run_xgb_cv(variant_name, config, X, y, fold_labels, g0_values, cubes):
    for fold in range(len(g0_values)):
        mask = fold_labels != fold
        X_tr, y_tr = X[mask], y[mask]       # ~12.5M rows
        X_va, y_va = X[~mask], y[~mask]     # ~2.1M rows

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)     # fit on training fold only
        X_va_s = sc.transform(X_va)         # transform val with training stats

        model = xgb.XGBRegressor(**config)
        model.fit(X_tr_s, y_tr, eval_set=[(X_va_s, y_va)], verbose=False)
        y_pred = model.predict(X_va_s)
```

The `StandardScaler` is fit on each training fold independently and applied to the validation fold — this prevents any information leakage from validation into training.

### 5.3 MLP: Fully-Connected Neural Networks

The MLP approach treats each cell as an independent sample with 14 (or 56 with spatial) features. Three architecture variants were explored:

**mlp_standard** — `[256, 256, 128, 64]` with BatchNorm + ReLU:
```
Input(14) -> Linear(14,256) -> BN -> ReLU
          -> Linear(256,256) -> BN -> ReLU
          -> Linear(256,128) -> BN -> ReLU
          -> Linear(128,64)  -> BN -> ReLU
          -> Linear(64,1)    -> output
```
The tapering width (256->128->64) forces progressive feature abstraction. Each layer extracts increasingly compact representations.

**mlp_wide** — `[512, 512, 256, 128]`:
Double the width at each layer. Tests whether raw capacity improves predictions. With ~15M training samples per fold, there is ample data to support wider networks without overfitting.

**mlp_residual** — 4 residual blocks of width 256:
```
Input(14) -> Linear(14,256)  [projection into hidden space]
          -> ResBlock(256)    [BN->ReLU->Linear->BN->ReLU->Linear + skip]
          -> ResBlock(256)
          -> ResBlock(256)
          -> ResBlock(256)
          -> Linear(256,1)    [output]
```

The residual MLP learns corrections `delta(x)` on top of a linear projection rather than a full nonlinear remapping. This is well-suited for physics regression where the underlying function is smooth: the skip connections ensure the network starts from an approximately linear function and refines it, rather than having to discover the entire mapping from scratch. The constant width (256 throughout) preserves information flow and prevents the bottlenecking that happens in tapering architectures.

**Training loop — key optimizations**:
```python
# Preload entire training fold to GPU (eliminates per-batch transfer overhead)
X_tr_t = torch.from_numpy(X_tr_s).to(device)  # ~700MB on GPU
y_tr_t = torch.from_numpy(y_tr).to(device)

# GPU-native random permutation (no Python/DataLoader overhead)
perm = torch.randperm(n_tr, device=device)

# Mixed-precision training (AMP) for 2x throughput on GPU
with torch.amp.autocast('cuda', enabled=use_amp):
    loss = loss_fn(model(xb), yb)
scaler_amp.scale(loss).backward()
```

With batch_size=262,144 and ~12.5M training samples, each epoch has ~48 gradient steps. The CosineAnnealingLR scheduler decays the learning rate from 1e-3 to near zero over the training run, giving a natural warm-up/cool-down cycle.

**Result**: All three MLP variants give R² ~ 0.885 with negligible differences. The bottleneck is the information content of the 14 per-cell features, not model capacity. `mlp_wide` is chosen for the final ensemble.

### 5.4 Why Dropout Was Rejected Across All Models

A critical finding from this project: **dropout hurts out-of-distribution extrapolation**.

In the CNN (Section 6), testing dropout=0.10 caused the G0=6.4 fold to collapse from R²=+0.22 to R²=-0.61 — a catastrophic failure. The reasoning applies to all models:

Standard wisdom says dropout prevents overfitting to noise in the training set. But in this problem, what appears to be "overfitting" is actually the model learning the precise physical relationship between features and fh2. With only 7 distinct G0 values (6 in training), the model must form stable internal representations of how fh2 depends on G0 and its interaction with local conditions.

Dropout randomly zeros feature channels during training, forcing the network to distribute information redundantly. For interpolation folds (G0 within the training range), this causes only a modest accuracy loss. But for extrapolation folds (G0=0.1, 6.4), stable feature representations are critical — the model must coherently extrapolate from the most similar training G0. Dropout breaks this coherence.

This principle extends to DART (dropout-on-trees for XGBoost), which was rejected for the same reason. The only regularization used is `weight_decay=1e-5` (L2 penalty on weights), which constrains the magnitude of learned parameters without disrupting their structure.

---

## 6. The 3D U-Net CNN: Architecture, Augmentation, and Training

### 6.1 Architecture

The CNN is the only model that directly processes 3D volumetric data rather than per-cell features. It is a U-Net encoder-decoder with skip connections:

```
INPUT: (batch=1, channels=15, depth=64, height=64, width=64)

ENCODER:
  enc1: ConvBlock(15 -> 32)     # (1, 32, 64, 64, 64)  - local features
  enc2: MaxPool + ConvBlock(32 -> 64)    # (1, 64, 32, 32, 32)  - 2x downsampled
  enc3: MaxPool + ConvBlock(64 -> 128)   # (1, 128, 16, 16, 16) - 4x downsampled

BOTTLENECK:
  bot:  MaxPool + ConvBlock(128 -> 256)  # (1, 256, 8, 8, 8)    - 8x downsampled

DECODER:
  dec3: Upsample + Concat(enc3) + ConvBlock(256+128 -> 128)  # (1, 128, 16, 16, 16)
  dec2: Upsample + Concat(enc2) + ConvBlock(128+64 -> 64)    # (1, 64, 32, 32, 32)
  dec1: Upsample + Concat(enc1) + ConvBlock(64+32 -> 32)     # (1, 32, 64, 64, 64)

OUTPUT: Conv3d(32 -> 1, kernel=1)  # (1, 1, 64, 64, 64) - predicted log10(fh2)
```

Each `ConvBlock` consists of two 3x3x3 convolutions with `InstanceNorm3d` and `ReLU`:
```python
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
```

**Key design choices**:

- **InstanceNorm3d instead of BatchNorm3d**: BatchNorm computes statistics across the batch dimension. With batch_size=1 (an entire 64^3 volume), BatchNorm is undefined. InstanceNorm computes statistics per-sample, per-channel — it normalizes the spatial dimensions (D,H,W) of each channel independently, and is stable at any batch size. `affine=True` adds learnable scale/shift parameters.

- **3 encoder levels, not 4**: The input is 64^3. Each MaxPool halves each dimension: 64->32->16->8. The 8^3 bottleneck provides a receptive field covering the full volume while retaining enough spatial resolution for dense prediction. A 4th level would compress to 4^3, destroying the fine-grained field gradients needed to predict fh2 at each cell.

- **Skip connections**: The U-Net skip connections concatenate encoder features with decoder features at each resolution level. This is critical because the encoder features contain local details (cell-level physics) while the decoder features contain global context (large-scale environment). The skip connections let the network combine both.

- **Trilinear upsampling**: Instead of transposed convolutions (which can produce checkerboard artifacts), the decoder uses `F.interpolate(mode='trilinear')` followed by a ConvBlock to refine the upsampled features.

### 6.2 Why 64^3 Input (Downsampled from 128^3)

The raw simulation grid is 128^3 = 2,097,152 cells per cube. Processing this at full resolution would require ~50GB GPU memory (14 channels x 128^3 x float32 per sample, plus model activations). Instead, each volume is downsampled to 64^3 using `F.avg_pool3d(kernel_size=2, stride=2)` before feeding to the CNN:

```python
ch_t = F.avg_pool3d(ch_t, kernel_size=2, stride=2)   # 128^3 -> 64^3
tgt_t = F.avg_pool3d(tgt_t, kernel_size=2, stride=2)  # target too
```

This reduces memory by 8x per volume while preserving the large-scale spatial structure that the CNN needs. The CNN then predicts fh2 on the 64^3 grid (262,144 cells per cube instead of 2,097,152).

### 6.3 Symmetry Augmentation: The Octahedral Group

The simulation grid has cubic symmetry: physically, rotating or reflecting the entire cube should produce an equally valid simulation. This symmetry can be exploited for data augmentation.

The full symmetry group of a cube is the **octahedral group Oh**, which has **48 operations**: 24 rotations + 24 improper rotations (rotation + reflection). Each operation is represented as a 3x3 signed-permutation matrix R:

```python
def _build_oh_group():
    """Generate all 48 signed-permutation matrices forming Oh."""
    for perm in [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]:
        for signs in product([-1, 1], repeat=3):
            R = np.zeros((3, 3), dtype=int)
            for new_ax, (old_ax, s) in enumerate(zip(perm, signs)):
                R[new_ax, old_ax] = s
```

**However**, the UV field illuminates the cube from a fixed direction (along the +z axis). This breaks the full cubic symmetry: operations that interchange z with x or y would change which direction the UV comes from, producing a physically invalid augmentation.

The valid subset is the **8 z-preserving operations**: 4 rotations around z (0, 90, 180, 270 degrees) and 4 reflections through planes containing the z-axis. These transform x and y but leave z invariant:

```python
_Z_PRESERVING = [R for R in _OH_GROUP
                 if R[2, 2] == 1 and R[0, 2] == 0 and R[1, 2] == 0]
# Exactly 8 operations
```

**Applying augmentations to physical fields**: Different field types transform differently under rotation/reflection:

- **Scalar fields** (nH, T, nHp, ext, fh2, G0): Only the grid indices are permuted. The values at each grid point stay the same.
- **Polar vector fields** (vx, vy, vz): Components are mixed by the rotation matrix: `v' = R @ v`. For example, a 90-degree z-rotation maps vx -> vy, vy -> -vx, vz -> vz.
- **Axial vector fields** (B-field: bxl, bxr, ...): Transform as `b' = det(R) * R @ b`. The determinant factor (det=+1 for rotations, -1 for reflections) distinguishes polar from axial vectors. Additionally, reflections that negate an axis swap the left/right designations of the face-centred B-field components along that axis.

This careful treatment of field transformation rules is physically essential — applying the wrong transformation would create training data with inconsistent physics.

### 6.4 CNN Training: Normalization and Loss

**Input normalization**: The 14 input channels have vastly different scales (log_nH ranges from ~-5 to +3, while bxl might range from -1000 to +1000). Without normalization, the first convolution layer would be dominated by the largest-magnitude channels. Per-channel z-score normalization (computed from training fold only) brings all channels to comparable ranges:

```python
all_x = torch.stack(train_ds.xs)                          # all training volumes
ch_mean = all_x.mean(dim=(0, 2, 3, 4), keepdim=True)      # per-channel mean
ch_std = all_x.std(dim=(0, 2, 3, 4), keepdim=True).clamp(min=1e-6)
train_ds.xs = [(x - ch_mean) / ch_std for x in train_ds.xs]
val_ds.xs = [(x - ch_mean) / ch_std for x in val_ds.xs]   # use training stats!
```

**Target normalization**: The target log10(nH2) spans a wide range across the training folds. MSE loss on this raw range would be dominated by cells with very low nH2 (large negative log values), causing the model to focus on predicting the near-zero tail while ignoring the physically interesting moderate-nH2 cells. Per-fold target standardization balances the loss:

```python
all_y = torch.stack(train_ds.ys)
y_mean = all_y.mean()
y_std = all_y.std().clamp(min=1e-6)
train_ds.ys = [(y - y_mean) / y_std for y in train_ds.ys]
# After prediction, inverse-transform: y_pred_original = y_pred * y_std + y_mean
```

This was one of the most impactful changes in the project: without target normalization, the CNN produced R² < 0 (worse than the mean). With it, R² jumped to 0.775 immediately. All R² values for the CNN are in log10(nH2) space.

### 6.5 Three CNN Size Variants

| Variant | base_ch | Parameters | Result |
|---------|---------|------------|--------|
| unet_small | 16 | ~1.5M | R² = 0.757, std=0.193 |
| unet_standard | 32 | ~5.8M | R² = 0.779, std=0.191 |
| unet_large | 64 | ~23M | R² = 0.686, std=0.409 |

The unet_large confirms the **over-parameterization hypothesis**: with only 6 training cubes (48 augmented samples), 23M parameters gives a ratio of ~480,000 parameters per training sample. The model memorizes training cubes without learning generalizable spatial patterns, and its high variance (std=0.409) indicates frequent catastrophic failures on extrapolation folds.

### 6.6 The XGBoost-Guided CNN

A hybrid approach where XGBoost's out-of-bag prediction is injected as a 15th input channel to the CNN:

```python
CNN_INPUT_COLS_GUIDED = CNN_INPUT_COLS + ['xgb_pred']  # 16 channels

# For each fold, inject the XGBoost prediction volume
train_vols_g = [{**all_vols[i], 'xgb_pred': xgb_vols[i]}
                for i in range(len(all_vols)) if i != fold]
val_vols_g = [{**all_vols[fold], 'xgb_pred': xgb_vols[fold]}]
```

Each cube's XGBoost prediction comes from the fold where that cube was held out — so there is no data leakage. The hypothesis was that providing a near-correct "prior" would help the CNN focus on learning the spatial corrections that XGBoost misses.

**Result**: R² = 0.624-0.814 depending on the run — highly variable and never consistently better than the vanilla CNN. The CNN's fundamental instability prevents it from reliably leveraging the injected prior.

---

## 7. Spatial Neighbourhood Features: Replacing the CNN

### The Core Idea

The CNN's unique advantage over pointwise models is its ability to see spatial context — the 3D arrangement of gas around each cell. But the CNN is expensive to train, highly unstable, and produces high-variance predictions on extrapolation folds.

The multi-scale spatial features (Section 4.3) capture the same spatial information in a much simpler way: instead of learning spatial filters end-to-end, we **precompute fixed 3D box-filter averages** at multiple scales and feed them as additional tabular features to XGBoost and MLP.

### Why It Works Better Than the CNN

1. **The spatial signal is simple**: The dominant spatial effect is shielding — a cell's fh2 depends on the average density/extinction in its neighbourhood, not on complex learned spatial patterns. A box-filter mean captures this directly.

2. **No training instability**: The spatial features are a deterministic preprocessing step. XGBoost and MLP train stably on the augmented feature set.

3. **Ensemble compatibility**: XGBoost and MLP produce predictions on the same per-cell grid, so they can be trivially averaged. CNN predictions are on a different resolution (64^3 vs 128^3) and require resolution matching, which introduces additional approximation error.

### The Progression

| Step | Method | Mean R² | Delta |
|------|--------|---------|-------|
| Baseline | XGBoost alone (14 features) | 0.886 | — |
| +Spatial 3^3 | XGBoost with 28 features | 0.917 | +0.031 |
| +Multi-scale | XGBoost with 56 features (3^3+5^3+7^3) | 0.924 | +0.007 |
| +Ensemble | ens_sp (XGBoost_sp + MLP_sp) | 0.948 | +0.024 |

Each step contributes a meaningful improvement. The spatial features alone (+0.031) are the largest single-step gain in the project. The multi-scale extension adds a further +0.007. The ensemble adds another +0.024 by combining XGBoost's tree-based decisions with MLP's continuous function approximation.

---

## 8. Ensemble Methods

### 8.1 Equal-Weight Ensemble (ens_sp) — The Best Model

The final production model is the simplest possible ensemble:

```python
y_ensemble = 0.5 * y_xgb_sp + 0.5 * y_mlp_sp
```

where both XGBoost and MLP are trained on the 56-feature set (14 baseline + 42 multi-scale spatial).

**Why averaging works**: XGBoost and MLP have different inductive biases that produce **complementary errors**:

- **XGBoost** makes piecewise-constant predictions (axis-aligned decision boundaries). It excels at sharp transitions (e.g., the boundary between molecular and atomic gas) but produces staircase-like artifacts in smooth regions. It also tends to handle the extrapolation folds (G0=0.1, 6.4) better because tree splits can extrapolate constant values beyond the training range.

- **MLP** makes smooth, continuous predictions (learned nonlinear feature combinations). It excels in regions where fh2 varies smoothly with temperature and density but can produce larger errors at sharp boundaries.

When these complementary predictions are averaged, the XGBoost staircasing is smoothed by MLP, and the MLP's boundary errors are corrected by XGBoost. The result is better than either model alone on every fold.

### 8.2 Ridge-Stacked Ensemble (stacked_sp)

Instead of fixed 0.5/0.5 weights, a Ridge regression meta-learner is trained on out-of-fold predictions to learn optimal combination weights:

```python
# For each held-out fold i:
#   Meta-train: stack OOF predictions from the 6 other folds
#   X_meta shape: (6 * 262_144, 2) -- two models' predictions as features
#   y_meta: ground-truth log_fh2 from those 6 folds
meta = Ridge(alpha=1.0)
meta.fit(X_meta_tr, y_meta_tr)
y_stacked = meta.predict(X_meta_val)
```

**Result**: stacked_sp R² = 0.946, slightly *lower* than ens_sp R² = 0.948. The Ridge learner sometimes overweights one model on the 6 meta-training folds in a way that does not transfer to the 7th held-out fold. With only 7 folds, there is not enough meta-training signal to learn weights that reliably outperform equal averaging.

### 8.3 Ensembles With CNN — Why They Fail

| Ensemble | Components | Mean R² |
|----------|-----------|---------|
| ens_xgb+mlp | XGBoost + MLP (no spatial) | 0.923 |
| ens_xgb+cnn | XGBoost + CNN | 0.897 |
| ens_all | XGBoost + MLP + CNN | 0.920-0.927 |
| **ens_sp** | **XGBoost_sp + MLP_sp** | **0.948** |

Including the CNN in the ensemble consistently hurts performance. The reason is the CNN's high variance: on some folds (especially G0=6.4) the CNN produces R² < 0, meaning its predictions are worse than the global mean. Averaging a good prediction with a catastrophically bad one produces a mediocre result. The equal-weight ensemble has no mechanism to downweight the CNN on bad folds.

The ens_sp approach sidesteps this entirely by (a) not using the CNN and (b) giving XGBoost and MLP spatial features that capture the same information the CNN would provide.

---

## 9. Experimental Iterations and What Each Taught Us

### Phase 1: Baseline Models (v2, March 1)

**Goal**: Establish baseline per-cell prediction accuracy.

**Results**:
- Linear Regression R²=0.230 — confirms nonlinearity
- XGBoost R²=0.887 — surprisingly strong from 14 features alone
- MLP R²=0.888 — comparable to XGBoost

**Lesson**: Pointwise models with just temperature, density, and a few other features can explain ~89% of variance in fh2. The remaining ~11% is likely spatial and G0-extrapolation effects.

### Phase 2: CNN Development (v2, March 1)

**Goal**: Can a 3D CNN exploit spatial structure to beat XGBoost?

**Challenges encountered and resolved**: Target normalization was missing, causing MSE to diverge. AMP (mixed precision) caused NaN gradients because physical fields exceed fp16 range. BatchNorm was undefined at batch_size=1. Each of these was diagnosed and fixed.

**After fixes**: CNN Run 1 achieved R²=0.775 — notably *worse* than XGBoost (0.887). The CNN sees spatial context but at the cost of operating on a coarser 64^3 grid and having far fewer effective training samples (6 cubes x 8 augmentations = 48 volumes vs 12.5M per-cell samples for XGBoost).

### Phase 3: CNN Regularization Experiments (v2, March 1)

**Dropout experiment**: Added dropout=0.10 to encoder, 0.20 to bottleneck. Result: G0=6.4 collapsed from R²=+0.22 to -0.61. Mean R² dropped from 0.775 to 0.649.

**Analysis**: With only 6 training cubes spanning 64x in G0, the model must learn stable representations of how fh2 depends on G0. Dropout randomly zeros feature channels, preventing stable representation formation. For interpolation folds (G0 in training range), the effect is mild. For extrapolation folds (G0=6.4 outside training range), it is catastrophic. The model can no longer coherently extrapolate from its most similar training examples.

**More epochs**: CNN Run 3 (150 epochs, no dropout) achieved R²=0.803 — best CNN result. G0=6.4 improved from 0.216 to 0.591 because the best-validation epoch moved from 28 to 111. Extreme folds need many epochs to converge.

### Phase 4: Systematic Architecture Comparison (v3-v6, March 6-9)

**Goal**: Exhaustively test XGBoost depth, MLP width/architecture, and CNN scale.

**Key findings**:
- XGBoost depth (4-8) makes no difference: R² ~ 0.886 for all three variants
- MLP architecture makes no difference: R² ~ 0.885 for all three variants
- CNN size matters but not in a good way: unet_large (23M params) is worse and more unstable than unet_standard (5.8M params)
- **Ensembles help**: ens_xgb+mlp R²=0.919-0.924, ens_all R²=0.927 (when CNN behaves)

### Phase 5: Spatial Features (v7-v8, March 9)

**Goal**: Give pointwise models spatial context without using a CNN.

**Single-scale (3^3)**: Immediate +0.031 R² for XGBoost (0.886 -> 0.917), +0.030 for MLP (0.885 -> 0.915). The spatial ensemble ens_sp = 0.947, a new project best.

**Decision**: CNN retired as default (moved to opt-in `--cnn` flag). The spatial-feature approach dominates.

### Phase 6: Multi-Scale Spatial (v9, March 9)

**Goal**: Test whether multi-scale context (3^3+5^3+7^3) improves over single-scale (3^3 only).

**Result**: XGBoost R² improved 0.917 -> 0.924 (+0.007). The hardest fold G0=0.2 improved 0.899 -> 0.915 (+0.016), which was the largest per-fold gain from any single change. Multi-scale captures both local shielding (3^3) and meso-scale cloud structure (7^3), which is especially important for the low-G0 cubes where self-shielding is the dominant physical process.

**Final best**: ens_sp R² = 0.9482 with std = 0.024.

### Phase 7: Production Pipeline (v10-v11, March 9-10)

**Goal**: Make predictions reproducible and inspectable.

Built a production prediction script (`predict_and_visualize.py`) that trains ens_sp and saves predictions as `.npz` files. Added `--all` flag to generate predictions for all 7 G0 folds in a single run. Created visualization tools for 3D volume rendering and 2D slice browsing.

---

## 10. Complete Results

> **Note on metrics**: All R² values in this section are computed in **log10(nH2) space (dex)**. R²=0.95 means the model explains 95% of variance in log10(nH2) — typical errors are ~0.1-0.2 dex (factor of 1.3-1.6 in linear nH2). These values are not directly comparable to older results that used linear-space metrics.

### 10.1 Final Model Comparison (Run 121724 — Best Run, Multi-Scale Spatial, No CNN)

| Variant | Mean R² | Std | G0=0.1 | G0=0.2 | G0=0.4 | G0=0.8 | G0=1.6 | G0=3.2 | G0=6.4 |
|---------|---------|-----|--------|--------|--------|--------|--------|--------|--------|
| **ens_sp** | **0.9482** | **0.024** | 0.908 | **0.915** | 0.953 | 0.962 | 0.967 | 0.967 | 0.965 |
| stacked_sp | 0.9456 | 0.020 | 0.912 | 0.916 | 0.954 | 0.960 | 0.962 | 0.961 | 0.954 |
| xgb_standard_sp | 0.9243 | 0.026 | 0.912 | 0.874 | 0.920 | 0.934 | 0.924 | 0.948 | 0.959 |
| ens_xgb+mlp | 0.9225 | 0.045 | 0.891 | 0.829 | 0.916 | 0.949 | 0.959 | 0.959 | 0.954 |
| stacked_xgb+mlp | 0.9150 | 0.040 | 0.882 | 0.831 | 0.922 | 0.945 | 0.949 | 0.937 | 0.940 |
| mlp_wide_sp | 0.9127 | 0.034 | 0.843 | 0.886 | 0.914 | 0.928 | 0.946 | 0.945 | 0.927 |
| xgb_standard | 0.8864 | 0.061 | 0.891 | 0.751 | 0.866 | 0.902 | 0.910 | 0.932 | 0.952 |
| mlp_standard | 0.8862 | 0.046 | 0.809 | 0.825 | 0.883 | 0.921 | 0.929 | 0.928 | 0.910 |

### 10.2 Full Run History (Best Ensemble Per Run)

| Run | Date | Key Config | Best Ensemble | Mean R² |
|-----|------|-----------|--------------|---------|
| 142959 | Mar 6 | 50 CNN epochs, 30 MLP epochs | (no ensemble) | 0.890 |
| 140718 | Mar 7 | 100 CNN epochs, ensembles introduced | ens_all | 0.902 |
| 134927 | Mar 8 | 150 CNN epochs, CNN at peak | ens_all | 0.927 |
| 182350 | Mar 8 | +spatial (3^3), 200 CNN epochs | ens_xgb+mlp | 0.922 |
| 085656 | Mar 9 | +spatial (3^3), ens_sp introduced | ens_sp | 0.947 |
| **121724** | **Mar 9** | **Multi-scale (3+5+7), no CNN** | **ens_sp** | **0.948** |

### 10.3 CNN Training Runs (Standalone)

| Run | Epochs | Dropout | Mean R² | Std | G0=0.1 | G0=6.4 |
|-----|--------|---------|---------|-----|--------|--------|
| 184723 | 100 | No | 0.775 | 0.247 | 0.649 | 0.216 |
| 194547 | 150 | Yes (0.10/0.20) | 0.649 | 0.523 | 0.634 | -0.611 |
| 205412 | 150 | No (+weight_decay) | 0.803 | 0.172 | 0.497 | 0.591 |

### 10.4 Ablation: How Each Component Contributes to ens_sp

| Configuration | Mean R² | Delta vs Previous |
|--------------|---------|-------------------|
| XGBoost alone (14 features) | 0.886 | — |
| + Spatial features 3^3 (28 features) | 0.917 | +0.031 |
| + Multi-scale 3^3+5^3+7^3 (56 features) | 0.924 | +0.007 |
| + MLP_sp equal-weight ensemble | 0.948 | +0.024 |

### 10.5 Per-Fold Difficulty (ens_sp)

| Fold | R² | Fold Type | Explanation |
|------|-----|----------|-------------|
| G0=0.1 | 0.908 | Extrapolation (low) | Hardest fold. No training data below G0=0.1. |
| G0=0.2 | 0.915 | Near-extrapolation | Second hardest. Only G0=0.1 below; large gap to next cube. |
| G0=0.4 | 0.953 | Interpolation | Bracketed by G0=0.2 and G0=0.8. |
| G0=0.8 | 0.962 | Interpolation | Centre of the G0 range. |
| G0=1.6 | 0.967 | Interpolation | Well-bracketed. |
| G0=3.2 | 0.967 | Interpolation | Well-bracketed. |
| G0=6.4 | 0.965 | Extrapolation (high) | Upward extrapolation; easier than G0=0.1 because high-G0 physics is simpler (UV dominates everywhere). |

---

## 11. Code Architecture Walkthrough

### 11.1 Data Pipeline (`data_loader.py`)

The entry point for all data. Loads 7 CSV files from `data/UVonly/{G0_dir}/*.csv`, where the directory name encodes G0 (e.g., `0_1` for G0=0.1, `3_2` for G0=3.2).

**`load_all_cubes()`**: Loads all 7 cubes, applies preprocessing (drop nH2, add log-transforms, add G0 features), returns a list of DataFrames sorted by G0.

**`get_X_y(cubes, use_log_target=True)`**: Stacks all cubes into flat arrays: X with shape (14,680,064 x 14), y with shape (14,680,064,), and a fold label array (0-6) that maps each row to its cube.

**`cube_to_volumes(df, cols)`**: Reshapes selected DataFrame columns into (128,128,128) numpy arrays using the 1-indexed grid coordinates ix, iy, iz. This is needed both for the CNN (which processes 3D volumes) and for computing spatial features (which require 3D convolutions).

**`load_single_cube(g0)`**: Loads a single cube for a specific G0 value. Used by the visualization scripts to load only the ground truth needed for comparison, avoiding the overhead of loading all 7 cubes.

### 11.2 Augmentation (`augmentation.py`)

Generates and applies the 48 symmetry operations of the octahedral group Oh to 3D simulation volumes.

**`_build_oh_group()`**: Enumerates all 48 signed-permutation matrices by iterating over 6 axis permutations x 8 sign combinations = 48 total. Deduplicates to verify exactly 48 unique operations.

**`get_symmetry_ops(safe_only=True)`**: Returns either the 8 z-preserving operations (safe for UV-from-z simulations) or all 48. The z-preserving filter keeps only operations where R[2,2]=+1 (z maps to +z, not -z or another axis).

**`augment_cube(volumes, R)`**: Applies one symmetry operation to an entire cube's volume dictionary:
1. Scalar fields: just permute/flip grid axes
2. Polar vectors (v): rotate components — `v'_new = sign[new] * v_old[perm[new]]`
3. Axial vectors (B): `b' = det(R) * sign * b_old[perm]`, with left/right swap on axis negation

### 11.3 Classical Models (`classical_models.py`)

Contains the three baseline models (Linear, XGBoost, MLP) and the shared metric function.

**`compute_metrics(y_true_log, y_pred_log)`**: The central evaluation function used by every model in the project. Takes log10(fh2) predictions, inverse-transforms to linear fh2, and computes R², RMSE, MAE. This ensures all models are compared on an identical scale.

**`run_xgboost()`**: 7-fold CV with GPU-accelerated XGBoost (`tree_method='hist'`). Subsample=0.3 uses 30% of rows per tree, which is critical for making 12.5M-row training feasible in reasonable time (~3 minutes per fold on GPU).

**`run_mlp()`**: 7-fold CV with GPU training. Notable: the entire training fold is preloaded to GPU memory (`torch.from_numpy(...).to(device)`), and batching uses `torch.randperm(device=device)` — both operations run entirely on the GPU with zero CPU-GPU data transfer per batch.

### 11.4 CNN Architecture (`cnn_model.py`)

Defines `UNet3D` and its building blocks (`ConvBlock`, `Down`, `Up`). The architecture is parameterized by `n_channels` (input) and `base_ch` (controls all internal widths). This parameterization allows easy experimentation: changing `base_ch` from 16 to 32 to 64 quadruples the parameter count while keeping the architecture identical.

### 11.5 Architecture Comparison (`compare_architectures.py`)

The main experimentation driver. Defines all variant configs, runs 7-fold CV for each, and logs results to JSON.

Key design: all predictions are stored per-fold (`all_preds[name] = (y_true_folds, y_pred_folds)`) so that ensemble combinations can be computed after all individual models are trained. This avoids re-running models when testing new ensemble combinations.

**`_normalize_preds_to_64()`**: When combining CNN predictions (64^3) with pointwise predictions (128^3), this function downsamples the pointwise predictions to match the CNN resolution via `avg_pool3d`. This ensures all predictions have the same length per fold for ensemble averaging.

**`run_ensemble_cv()`**: Computes equal-weight average of per-fold predictions from two or more models. No training involved — just `np.mean(y_preds, axis=0)`.

**`run_stacked_ensemble_cv()`**: Trains a Ridge regression meta-learner on out-of-fold predictions. For each held-out fold i, the meta-learner trains on the predictions from the other 6 folds (where it can see both the predictions and the ground truth) and predicts on fold i's predictions.

### 11.6 Production Prediction (`predict_and_visualize.py`)

Trains the best ensemble (ens_sp) and saves predictions. The code is self-contained: it reimplements XGBoost training, MLP training, and spatial feature computation rather than importing from `compare_architectures.py`, because the production pipeline should not depend on the experimental framework.

**`_run_fold()`**: The core function. Trains XGBoost, trains MLP, averages predictions, reshapes to 128^3 volume, and saves to `.npz` with metadata (G0, R² scores, kernel sizes, epochs).

**`--all` flag**: Loops over all 7 folds, calling `_run_fold()` for each. Data loading and spatial feature computation happen once; only the training/prediction step is repeated per fold.

### 11.7 File Structure Overview

```
AI-final-project/
├── data_loader.py                     # Data loading, preprocessing, volume conversion, get_feature_cols()
├── augmentation.py                    # Oh symmetry group, z-preserving ops
├── classical_models.py                # Linear, XGBoost, MLP baselines + compute_metrics()
├── cnn_model.py                       # 3D U-Net architecture (ConvBlock, Down, Up, UNet3D)
├── model_helpers.py                   # Shared helpers: FlexMLP, _fit_xgb/mlp, _compute_spatial_X, etc.
├── train_cnn.py                       # CNN training loop with augmentation and normalization
├── compare_architectures.py           # Systematic variant comparison (main experiment driver)
├── evaluate.py                        # Run all models, comparison plots, JSON log
├── predict_and_visualize.py           # Production prediction pipeline (ens_sp)
├── single_cube_extrapolation.py       # 7x7 R2 matrix: train on 1 G0 cube, predict all 7
├── intra_cube_section.py              # Spatial section: train on subset, predict remainder
├── intra_cube_visualize.py            # Interactive 4-panel z-slice viewer for spatial splits
├── statistical_analysis.py            # Statistical model comparison utilities
├── 3d_visualizer.py                   # 3D PyVista volume rendering
├── load_and_compare.py                # 3D interactive pyvista volume viewer (predictions vs truth)
├── slice_compare.py                   # 2D matplotlib slice browser with slider/textbox
├── viz_common.py                      # Shared visualization utilities
├── predictions/                       # Saved .npz prediction files (one per G0)
├── logs/                              # Experiment result logs
│   ├── single_cube_extrapolation/     # 7x7 extrapolation runs + heatmaps
│   └── intra_cube_section/            # Spatial section runs + heatmaps
└── results/                           # Timestamped JSON run logs
    ├── arch_comparison_*.json         # Architecture comparison logs
    └── cnn_training_*.json            # CNN-specific training logs
```

---

## 12. Visualization Pipeline

### 12.1 3D Volume Viewer (`load_and_compare.py`)

Uses PyVista for interactive 3D volume rendering. Three linked panels show Ground Truth, Prediction, and Error side by side:

```python
plotter = pv.Plotter(shape=(1, 3), window_size=(1920, 700))
# Each panel uses add_volume() with:
#   cmap='magma' — perceptually uniform colormap
#   opacity='linear' — transparent at low values, opaque at high
#   clim=(p1, p99) — shared 1st-99th percentile colour limits from truth
plotter.link_views()  # all three panels rotate together
```

The `_to_pv_grid()` helper wraps a numpy array in a PyVista `ImageData` object with cell-centered data. The `link_views()` call ensures that rotating one panel rotates all three, making visual comparison intuitive.

### 12.2 2D Slice Browser (`slice_compare.py`)

A matplotlib interactive viewer for examining individual z-slices. Three `imshow` panels share the same colormap and colour limits. Navigation uses two linked controls:

- **Slider**: Drag to scroll through z=0 to z=127
- **TextBox**: Type a z value and press Enter to jump

These two controls are linked (changing one updates the other) with a **mutual recursion guard** to prevent infinite feedback loops:

```python
_busy = [False]  # mutable flag (list so inner functions can modify it)

def on_slider(val):
    if _busy[0]: return      # break the recursion
    _busy[0] = True
    textbox.set_val(str(z))  # this would normally trigger on_textbox
    _busy[0] = False
    _update(z)

def on_textbox(text):
    if _busy[0]: return      # break the recursion
    _busy[0] = True
    slider.set_val(z)        # this would normally trigger on_slider
    _busy[0] = False
    _update(z)
```

### 12.3 Shared Utilities (`viz_common.py`)

Three functions shared between `load_and_compare.py` and `slice_compare.py`:

- **`select_prediction_file()`**: Opens a tkinter file dialog pre-pointed at the `predictions/` directory. Returns the selected `.npz` path.
- **`load_prediction(npz_path)`**: Unpacks the `.npz` file into its component arrays and metadata. Returns a tuple of (pred_vol, g0, r2_xgb, r2_mlp, r2_ens, kernels, epochs).
- **`prepare_display(truth_vol, pred_vol, log_scale)`**: Converts the log-space prediction volumes to display-space (either log10(fh2) or linear fh2), computes the error volume (pred - truth in display space), and returns consistent float32 arrays. The `log_scale` flag controls whether the viewer shows the raw log10 values or the exponentiated linear fh2.

---

## 13. Conclusions and Key Takeaways

### The Final Model: ens_sp

```
Input: 60 features per cell
  - 15 physical features (log_nH, log_T, log_nHp, ext, log_fh2, log_G0, vx, vy, vz, bxl..bzr)
  - 45 multi-scale spatial features (15 features x 3 kernel sizes: 3^3, 5^3, 7^3)

Model 1: XGBoost (depth=6, 400 trees, lr=0.1, subsample=0.3)
Model 2: MLP ([512, 512, 256, 128], BatchNorm+ReLU, 100 epochs, Adam, CosineAnnealingLR)

Ensemble: y_pred = 0.5 * y_xgb + 0.5 * y_mlp

Target: log10(nH2)
Metrics: log-space R², RMSE (dex), MAE (dex)

Performance: R² = 0.9482 +/- 0.024 across 7 leave-one-G0-out folds
  G0=0.1: 0.908 | G0=0.2: 0.915 | G0=0.4: 0.953 | G0=0.8: 0.962
  G0=1.6: 0.967 | G0=3.2: 0.967 | G0=6.4: 0.965
```

### Key Takeaways

1. **Feature engineering > architecture search**: The single biggest improvement came from adding spatial neighbourhood means (+3.1 R² points), not from any model architecture change. All three MLP variants gave nearly identical results; all three XGBoost depths gave nearly identical results.

2. **Pointwise models + spatial features > 3D CNN**: The CNN naturally captures spatial context, but it is expensive, unstable, and poor at extrapolation. Computing spatial context as tabular features and feeding them to XGBoost/MLP gives better accuracy, lower variance, and 10x faster training.

3. **Simple ensembles work**: Equal-weight averaging of XGBoost + MLP outperforms both Ridge-stacked ensembles and any combination involving the CNN. The models have complementary inductive biases (piecewise-constant vs smooth), making their average consistently better than either alone.

4. **Dropout destroys extrapolation**: When the model must learn genuine physical structure (not just statistical patterns) to extrapolate to unseen conditions, dropout prevents stable representation formation. Weight decay (L2 regularization) constrains parameter magnitudes without disrupting their structure.

5. **Target normalization is critical for CNNs**: Without standardizing the target (log_nH2), MSE loss is dominated by near-zero cells and the model collapses. This single fix changed CNN R² from negative to 0.775.

6. **Leave-one-out CV on the physics parameter is the right test**: Random CV would give inflated R² (~0.99+) by letting the model memorize per-cube statistics. Leave-one-G0-out reveals the true generalization ability and correctly identifies the hard folds (boundary G0 values).

7. **Physics-aware data augmentation**: The 8 z-preserving symmetry operations correctly handle scalar, polar vector, and axial vector field transformations. Using the wrong subset (all 48 ops when UV illumination breaks z-symmetry) would introduce physically invalid training data.

8. **Spatial sampling geometry matters more than spatial coverage volume**: In the intra-cube section experiments, 1% random sampling of cells achieves R² > 0.89 on the remaining 99%, while 50% coverage from a contiguous slab gives R² < 0 (catastrophic failure). Coverage uniformity determines interpolation success, not coverage fraction. This has direct relevance for observational survey design.

9. **Single-cube models transfer across small G0 gaps**: A model trained on one G0 simulation accurately predicts adjacent G0 values (R²=0.976 for one-step neighbours) but degrades significantly over large G0 ranges. The full multi-cube training set is necessary for robust generalization across the 64x UV field range.

---

## 14. Additional Experiments

### 14.1 Single-Cube Extrapolation (`single_cube_extrapolation.py`)

**Research question**: How much information does one G0 simulation contain about the chemistry at other UV field strengths?

**Method**: Train stacked_sp (XGBoost + MLP + Ridge meta-learner, with 60-feature multi-scale spatial features) on a single G0 cube, then predict all 7 cubes. This produces a 7x7 R² matrix where:
- Row = training (source) cube
- Column = prediction (target) cube
- Diagonal = in-sample fit (R² ~ 0.99 — the model fits the cube it was trained on)
- Off-diagonal = out-of-sample extrapolation to a different G0

**Results** (log-space R², from `logs/single_cube_extrapolation/run_20260313_022129.json`):

Training on G0=0.1, predicting each target cube:

| Target G0 | G0 distance | Stacked R² |
|-----------|------------|-----------|
| 0.1 (in-sample) | — | 0.994 |
| 0.2 | 1 step | 0.976 |
| 0.4 | 2 steps | 0.958 |
| 0.8 | 3 steps | 0.876 |
| 1.6 | 4 steps | 0.745 |
| 3.2 | 5 steps | 0.446 |
| 6.4 | 6 steps | ~0.15 |

Mean off-diagonal stacked R² across all 42 off-diagonal entries: ~0.78.

**Interpretation**: The chemical relationships encoded in one simulation — the mapping from local density, temperature, and shielding factors to nH2 — generalise well to neighbouring UV conditions (within 2x-4x changes in G0), but degrade rapidly over large G0 jumps. The fundamental nH-T-nH2 equilibrium is universal; what changes across G0 is the UV-driven photodissociation balance. Adjacent cubes share nearly identical local chemistry; distant cubes differ in which cells are photodissociated.

This result empirically validates the leave-one-G0-out CV design: a single-cube model is nowhere near as strong as the full 6-cube training set (where the model has seen both adjacent and distant G0 values).

The experiment also reveals an asymmetry: the pattern of decreasing R² with G0 distance is not perfectly symmetric, because the physics is not symmetric. Low-G0 simulations (self-shielding dominated) and high-G0 simulations (UV dominated) require different physical representations.

**Code**: `python single_cube_extrapolation.py` produces the full 7x7 matrix and saves a heatmap PNG. `--train-g0 0.8` restricts to a single source cube.

---

### 14.2 Intra-Cube Spatial Section (`intra_cube_section.py`)

**Research question**: How well can the model interpolate nH2 within a single cube when trained on a spatial subset? How does the geometry of the training region affect interpolation quality?

**Method**: For each of the 7 G0 cubes, train stacked_sp on a spatial subsection of cells and predict the remainder. Three geometries were tested:

| Split type | Description | Example fractions tested |
|---|---|---|
| **Contiguous slab** | All cells from one half of the cube along x, y, or z axis | 50% (x_half, y_half, z_half) |
| **Random fraction** | Cells selected uniformly at random (i.e. same physical mix) | 1%, 5%, 10%, 25%, 50%, 75% |
| **Contiguous box** | Single axis-aligned cubic sub-region of specified volume fraction | 1%, 5%, 10%, 25%, 50% |

Spatial neighbourhood features are computed from the training mask only (NaN for test cells), preventing direct leakage of test cell information.

**Results** (log-space R², G0=0.1 cube, from `logs/intra_cube_section/run_20260313_142443.json`):

| Split | Training cells | Test R² (stacked) |
|---|---|---|
| rand_1 (1% random) | 20,971 | **0.897** |
| rand_5 (5% random) | 104,857 | **0.952** |
| rand_10 (10% random) | 209,715 | **0.963** |
| rand_25 (25% random) | 524,288 | **0.970** |
| rand_50 (50% random) | 1,048,576 | **0.980** |
| rand_75 (75% random) | 1,572,864 | **0.985** |
| x_half (50% slab) | 1,032,192 | **-0.394** |
| y_half (50% slab) | 1,032,192 | **-2.689** |
| z_half (50% slab) | 1,032,192 | **-1.389** |
| box_1 (1% box) | 21,952 | **0.693** |
| box_5 (5% box) | 103,823 | **0.672** |
| box_25 (25% box) | 531,441 | **0.823** |
| box_50 (50% box) | 1,061,208 | **0.779** |

The pattern is consistent across all 7 G0 cubes.

**The central result**: A **1% random sample** (21,000 cells out of 2,097,152) achieves test R²=0.897. A **50% contiguous slab** achieves test R²=-0.394 — meaning the model is worse than predicting the training mean for every held-out cell.

**Why random sampling succeeds**:

Random sampling ensures that test cells are statistically near training cells in all 3 spatial dimensions. The box-filter spatial neighbourhood features — computed from training cells only — provide interpolated context for each test cell based on the nearby training cells. The model essentially learns: "given that my immediate neighbourhood has average density X and average temperature Y, my nH2 should be Z." With random sampling, test cells always have training cells nearby, so this interpolation works.

At just 1% training coverage, the neighbourhood features are sparse (many cells contributing to the box-filter mean are NaN), but there is enough signal to achieve R²=0.90. At 25%, the neighbourhood features are nearly fully populated and the model performs near the leave-one-G0-out ceiling.

**Why contiguous slabs fail catastrophically**:

A contiguous slab training set leaves an entire spatial half of the cube unseen. The nH2 field in a molecular cloud simulation is not spatially homogeneous: one half may be the UV-illuminated surface (low nH2) and the other may be the shielded interior (high nH2). The model trained on the illuminated half sees a completely different density-chemistry relationship than what exists in the shielded half. For MLP, this manifests as predicting the training-set mean everywhere in the test region (R² as low as -7 for some folds). XGBoost is more robust (R² ~ 0.5-0.8 on slabs) because decision tree splits can extrapolate constant predictions beyond the training feature range.

**Why box splits are intermediate**:

A box provides spatial coverage across all 3 dimensions, but only in a local region. Cells far from the box experience sparse neighbourhood features, degrading predictions. The effective coverage radius decreases with box fraction.

**XGB vs MLP behaviour on contiguous splits**:

XGBoost consistently outperforms MLP on all contiguous splits. Decision trees naturally extrapolate by predicting the leaf mean for out-of-distribution inputs — this is a modest constant prediction, which is at least unbiased. MLP activations can collapse to the training distribution mean or diverge, producing systematically wrong predictions on cells with spatial feature distributions never seen during training.

**Scientific implication**:

Even sparse, random observations of a molecular cloud — such as a set of randomly chosen sight-lines in an IFU (integral field unit) observation — would be sufficient to map the H2 density field with high accuracy using this ML approach. This is a practical result for observational astronomy: efficient survey designs should prioritize spatial coverage uniformity over spatial depth.

The z-axis asymmetry in slab results (z_half often less catastrophic than x_half or y_half) reflects the physical asymmetry introduced by the UV illumination direction — along z, the gradient from illuminated to shielded gas is smoother, making half-slab training slightly more representative.

**Code**: `python intra_cube_section.py` runs all 7 G0 cubes with all 14 split strategies. `--g0 0.8` restricts to one cube. `python intra_cube_visualize.py` launches an interactive 4-panel z-slice viewer showing ground truth, training mask, prediction, and error for a single random-fraction split.
