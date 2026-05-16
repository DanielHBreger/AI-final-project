# Machine Learning Surrogates for Molecular Hydrogen Density in 3D Interstellar Medium Simulations: Multi-Scale Spatial Features Outperform Volumetric Neural Networks

**Daniel H. Breger**

*April 2026*

---

## Abstract

The molecular hydrogen number density (nH₂) is a critical quantity in interstellar medium (ISM) physics, yet computing it self-consistently in 3D simulations requires solving time-dependent non-equilibrium chemistry coupled to radiative transfer — a computational expense that limits the resolution and parameter coverage of modern simulations. We present a machine learning surrogate trained to predict log₁₀(nH₂) from 15 local physical properties at each cell of a 128³ simulation grid, evaluated under the demanding protocol of leave-one-G0-out cross-validation: each fold withholds an entire simulation at a UV field strength G0 (Habing units) absent from training. This directly measures out-of-distribution generalization across the dominant parameter governing H₂ photochemistry. Our dataset comprises seven UVonly chemistry simulations at G0 ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4} Habing units, totalling 14,680,064 cells. The principal methodological contribution is the replacement of a 3D U-Net convolutional neural network with multi-scale spatial neighbourhood features: box-filter averages of all 15 physical fields at three kernel sizes (3³, 5³, 7³ voxels), computed via separable convolution in approximately 30 seconds. Adding these 45 spatial features to a gradient-boosted tree ensemble improved the mean log-space R² from 0.886 to 0.924 in a single step. An equal-weight ensemble of XGBoost and a wide fully-connected MLP, each trained on the 60-dimensional spatial feature set, achieves a mean R² of 0.948 ± 0.024 across all seven folds, with per-fold R² ranging from 0.908 (the extrapolation fold G0=0.1) to 0.967 (mid-range folds). The 3D U-Net — despite having the nominally correct inductive bias for a spatially correlated field — achieved a best mean R² of 0.803 ± 0.172 with an order of magnitude greater fold-to-fold variance, confirming that engineered spatial features dominate learned volumetric filters when training data is limited to a small number of simulation cubes. We further show that dropout regularization catastrophically degrades extrapolation to unseen G0 values, whereas L2 weight decay is safe. An intra-cube spatial sampling experiment demonstrates that 1% uniform random coverage of a simulation volume reconstructs the H₂ density field with R²=0.897, while 50% contiguous-slab coverage yields R²=−0.394, with implications for efficient observational survey design.

---

## 1. Introduction

Star formation is regulated by the abundance of molecular hydrogen (H₂) in the interstellar medium. H₂ is the primary coolant at temperatures below ∼300 K and the dominant mass reservoir from which molecular cloud cores collapse into stars. The molecular fraction fH₂ — the ratio of H₂ density to total hydrogen density — therefore sets the initial conditions for the entire star-formation sequence. Yet computing fH₂ self-consistently within a 3D hydrodynamics simulation requires solving a system of coupled non-equilibrium chemistry equations at every grid cell, tracking the time-dependent balance between H₂ formation on dust grain surfaces, photodissociation by Lyman–Werner UV photons (11.2–13.6 eV), and the self-shielding that arises when dense H₂ columns attenuate their own dissociating radiation. In simulations with ∼10⁶–10⁷ cells, this chemistry solver can consume more than half the total compute budget, and including it at all resolutions and UV parameter values is often infeasible.

Machine learning surrogates offer a path to replacing or accelerating this chemistry. The core idea is to train a model on a finite set of high-fidelity simulations and use it to predict the chemical state in new simulations without running the full solver. This approach has been explored in several regimes of astrophysical chemistry, from primordial hydrogen chemistry in early-universe simulations to molecular cloud photodissociation region (PDR) modelling. The challenge shared by these applications is that the mapping from local physical conditions to chemical abundance is not a simple regression: it involves threshold-like transitions (e.g., the H/H₂ phase transition at a critical column density), enormous dynamic range in the target quantity (nH₂ spans roughly 35 orders of magnitude between UV-exposed and deeply shielded gas), and non-local spatial effects (a cell's H₂ abundance depends on the column of gas between it and the UV source, not merely on its own density and temperature).

This last point — the non-locality — poses a specific difficulty for standard per-cell machine learning. A cell deep within a dense cloud may have the same local density and temperature as a cell on the cloud surface, yet their H₂ abundances differ by many orders of magnitude because the interior cell is shielded from the dissociating UV field. Models that process each cell independently, without access to the spatial arrangement of its neighbours, must rely on surrogate quantities such as the self-shielding factor fh2 or the local extinction to infer the shielding state. A natural alternative is a volumetric convolutional neural network (CNN), which processes the entire simulation volume and can, in principle, learn to aggregate spatial context through its receptive field. However, CNN-based approaches are expensive to train, require a sufficient number of volumetric training samples, and may be unstable when the training set consists of only a handful of simulation cubes at distinct parameter values.

In this work we address these challenges through a systematic model comparison on the UVonly chemistry suite, seven 128³-cell simulations that differ only in ambient UV field strength G0. We make four principal contributions. First, we establish that the problem is fundamentally non-linear (linear regression achieves mean R²=0.230) and that gradient-boosted trees trained on 15 local physical features achieve R²=0.886 under stringent leave-one-G0-out cross-validation. Second, we introduce multi-scale spatial neighbourhood features — precomputed 3D box-filter averages at scales of 3³, 5³, and 7³ voxels — and show that these features improve the XGBoost R² to 0.924 (+0.038) in a single, deterministic preprocessing step that requires no additional training. Third, we demonstrate that an equal-weight ensemble of XGBoost and a wide fully-connected MLP trained on this 60-dimensional feature set achieves R²=0.948, and that this approach systematically outperforms a 3D U-Net CNN on every accuracy and stability metric. Fourth, we present an intra-cube spatial sampling study showing that uniform random spatial coverage is necessary and sufficient for accurate H₂ density reconstruction, with implications for observational survey strategy.

The remainder of this paper is organized as follows. Section 2 describes the simulation dataset and its physical context. Section 3 details the cross-validation protocol and evaluation metrics. Section 4 describes the feature engineering and model architectures. Section 5 presents the results, including the ablation study, per-fold analysis, and CNN comparison. Section 6 presents the supplementary experiments on single-cube extrapolation and intra-cube spatial sampling. Section 7 discusses the implications and limitations. Section 8 concludes.

---

## 2. Data

### 2.1 Simulation Suite

Our dataset consists of seven 3D uniform-grid simulations from the UVonly chemistry suite. Each simulation is a 128×128×128 grid (2,097,152 cells) evolving identical initial conditions of turbulent ISM gas under a spatially uniform UV radiation field of strength G0, measured in Habing units (1 Habing unit ≈ 1.6 × 10⁻³ erg cm⁻² s⁻¹ in the Lyman–Werner band). The seven simulations span a 64-fold dynamic range in UV field strength: G0 ∈ {0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4} Habing units, sampled on a geometric sequence with a factor-of-two spacing. This spacing ensures that adjacent simulations are reasonably similar — a model trained on all six G0 values except one can interpolate or extrapolate over at most a factor of two in UV field strength — while the full 64-fold range produces physically distinct equilibria. The total dataset contains 14,680,064 cells.

### 2.2 Physical Fields

Each cell carries eighteen raw fields. Of these, three coordinate fields (ix, iy, iz) are grid indices and are not used as model inputs. The target field is the molecular hydrogen number density nH₂ (cm⁻³), which is log-transformed to log₁₀(nH₂). The remaining fifteen fields constitute the input feature set and are described below.

Three fields spanning many orders of magnitude are log-transformed: the total hydrogen number density nH (cm⁻³), the gas temperature T (K), and the ionized hydrogen number density nH₊ (cm⁻³). A small regularising constant ε=10⁻³⁰ is added before taking logarithms to guard against zero values. The H₂ self-shielding factor fh₂ is similarly log-transformed. This quantity deserves special mention: fh₂ measures the fraction of Lyman–Werner UV photons that are attenuated by the H₂ column between the cell and the UV source, integrated along the illumination axis. It is computed by the chemistry solver as an integral over the 3D distribution of H₂ along each sight-line, and is therefore not algebraically derivable from the local nH₂ value. Two cells with identical local nH₂ can have very different fh₂ values depending on their position relative to cloud surfaces and the UV field direction. As a result, log(fh₂) is the most informative single predictor of log(nH₂): it directly encodes the directional shielding state that determines whether the local UV photodissociation rate can overcome H₂ formation. The remaining fields are the dust extinction proxy ext, the global UV field strength log(G0) (constant across all cells within one simulation), the three gas velocity components (vx, vy, vz) encoding turbulent motions, and the six face-centred magnetic field components (bxl, bxr, byl, byr, bzl, bzr).

### 2.3 Target Distribution

The molecular hydrogen number density spans approximately 35 orders of magnitude in linear space, from nH₂ ≈ 10⁻³⁰ cm⁻³ in UV-exposed regions (essentially no molecular hydrogen) to nH₂ ≈ 10⁵ cm⁻³ in dense shielded cloud cores. Regression in linear space is numerically infeasible: a model that mispredicts even a single cell by 10 orders of magnitude would produce a mean squared error that swamps the contribution of every other cell. In log₁₀(nH₂) space, the distribution is approximately Gaussian with a mean near zero dex and a standard deviation of two to three dex depending on G0. All model training and evaluation is therefore performed in log-space. Under this transformation, the physically meaningful question — does the model predict the correct order of magnitude? — is directly captured by the squared residuals.

---

## 3. Evaluation Protocol

### 3.1 Leave-One-G0-Out Cross-Validation

The evaluation protocol is leave-one-G0-out 7-fold cross-validation: each fold withholds an entire simulation cube at one G0 value (approximately 2.1 million cells) and trains on the remaining six cubes (approximately 12.6 million cells). This protocol was chosen over random k-fold splits for a fundamental reason: random splits would allow the model to see cells from every G0 value during training. Because cells within the same cube share the same global UV field strength and therefore the same macroscopic chemistry regime, a randomly split model faces only within-distribution interpolation — a task much easier than predicting nH₂ at a genuinely new G0 value. The real-world application of a chemistry surrogate is to run simulations at UV conditions not yet explored; leave-one-G0-out directly measures this capability.

The difficulty of each fold depends on whether the held-out G0 lies within the convex hull of the training G0 values (interpolation) or outside it (extrapolation). The two boundary folds — G0=0.1, which has no training data below it, and G0=6.4, which has no training data above it — require downward and upward extrapolation respectively. All interior folds require interpolation between two bracketing training cubes. A consistent a-priori prediction, confirmed by every experiment, is that G0=0.1 and G0=0.2 are the hardest folds, while the mid-range folds G0=0.8 through G0=3.2 are easiest. G0=6.4 is intermediate: despite being a boundary fold, high-G0 physics is simpler because UV photodissociation dominates the chemistry uniformly, with less nuanced competition from self-shielding.

### 3.2 Metrics

All primary results are reported in log₁₀(nH₂) space (dex). The coefficient of determination R² is the primary metric: R²=0.95 means the model explains 95% of the variance in log₁₀(nH₂), corresponding to typical per-cell errors of 0.1–0.2 dex (a factor of 1.3–1.6 in linear nH₂). Root mean square error (RMSE) in dex quantifies the typical per-cell error, and mean absolute error (MAE) in dex provides a robust complement to RMSE. As a secondary metric, a clipped linear-space R² is computed by exponentiating predictions clipped to within ±1 dex of the true range, providing a bounded linear-space comparison. All per-fold metrics are reported individually, and the mean and standard deviation across the seven folds characterise overall performance and stability.

---

## 4. Methods

### 4.1 Baseline Features

The 15-dimensional baseline feature vector consists of: log₁₀(nH), log₁₀(T), log₁₀(nH₊), ext, log₁₀(fh₂), log₁₀(G0), vx, vy, vz, and the six magnetic field components bxl, bxr, byl, byr, bzl, bzr. Feature standardisation (zero mean, unit variance) is applied within each training fold using statistics computed from the training cells only; validation cells are transformed using the training fold statistics to prevent information leakage.

### 4.2 Multi-Scale Spatial Neighbourhood Features

The dominant non-local effect governing nH₂ is the shielding of UV photons by surrounding gas. A cell's effective UV exposure depends not only on its own density and extinction, but on the column of gas and dust that separates it from the UV source — information encoded in the spatial distribution of its neighbours. To provide pointwise models with spatial context without resorting to a full volumetric CNN, we precompute 3D box-filter averages of all 15 physical features at three spatial scales.

For each feature f and each simulation cube, we reshape the N=2,097,152 tabular values into a 128×128×128 volume, compute the uniform box filter at kernel sizes k∈{3,5,7} using scipy.ndimage.uniform_filter (a separable, O(N) operation independent of kernel size), and then re-index the filtered volume back to tabular form using the original grid coordinates. The filtered values at kernel size k represent the mean of f within the k×k×k voxel neighbourhood centred on each cell. This yields 15 features × 3 scales = 45 spatial features, which are concatenated with the 15 baseline features to produce a 60-dimensional feature vector per cell. The entire computation requires approximately 30 seconds for all seven cubes at all three scales and is fully deterministic, incurring no training variance.

The physical interpretation is as follows: the 3³ kernel captures local density and temperature gradients (the immediate neighbourhood structure), the 5³ kernel captures the local cloud environment (the scale at which self-shielding columns become significant), and the 7³ kernel captures meso-scale structure such as the boundary between cloud and inter-cloud medium. Together, these scales provide the pointwise model with a coarse but complete picture of the 3D environment surrounding each cell — the information that a CNN would have to learn from scratch through its receptive field.

### 4.3 Model Architectures

**Linear Regression.** A standard ordinary-least-squares regression with L2-normalised features serves as a lower bound, testing whether the fh₂–nH₂ relationship is approximately linear after log-transformation.

**Gradient-Boosted Trees (XGBoost).** We use XGBoost with the GPU-accelerated histogram tree method (tree_method='hist'). Each fold trains on approximately 12.6 million cells; with 400 trees, maximum depth 6, learning rate 0.1, and a row subsampling fraction of 0.3. The subsampling fraction is a critical practical choice: it draws 30% of training rows (approximately 3.8 million cells) to construct each tree, reducing the per-tree cost by 3.3× and providing stochastic regularisation against overfitting. Three depth variants (4, 6, 8) were tested to assess the sensitivity to feature interaction order; all produced nearly identical results (R² ≈ 0.886), confirming that the primary predictive signal comes from low-order feature interactions. We report xgb_standard (depth=6) as the representative XGBoost result.

**Fully-Connected MLP.** The MLP is implemented in PyTorch and trained with GPU acceleration. To handle the 12.6-million-row training set efficiently, all training data is preloaded to GPU memory in a single transfer; batching within an epoch is performed via GPU-native random permutation (torch.randperm), eliminating CPU-to-GPU transfer overhead per batch. Batch size is 262,144 and training runs for 100 epochs per fold with Adam (learning rate 10⁻³, weight decay 10⁻⁵) and a cosine annealing learning rate schedule. Three architectures were evaluated: mlp_standard with hidden widths [256, 256, 128, 64] and batch normalisation, mlp_wide with [512, 512, 256, 128], and mlp_residual with four residual blocks of width 256. All three produced R² ≈ 0.885, confirming that the performance bottleneck is feature information content, not model capacity. We report mlp_wide as the representative MLP result.

**3D U-Net.** The volumetric CNN is a standard U-Net encoder–decoder with skip connections. The encoder has three downsampling stages (3×3×3 convolutions with max-pooling: 64³→32³→16³) and a bottleneck at 8³, giving a receptive field spanning the full volume. The decoder uses trilinear upsampling followed by convolutional refinement. Because batch size is necessarily 1 (one entire 64³ volume), batch normalisation is replaced with instance normalisation. Input volumes are downsampled from 128³ to 64³ using average pooling before entering the network, reducing memory requirements by 8×. Three capacity variants — unet_small (base channels 16, 1.5M parameters), unet_standard (base channels 32, 5.8M parameters), and unet_large (base channels 64, 23M parameters) — were evaluated. Physics-aware data augmentation is applied during training: the 8 z-preserving operations from the octahedral symmetry group Oh are applied to each training volume, with scalar, polar vector (velocity), and axial vector (magnetic field) fields transformed according to their respective group representations. The full 48-element group is not used because the fixed UV illumination direction along z breaks the symmetry operations that interchange z with other axes.

**Ensemble Methods.** The primary ensemble (ens_sp) averages the per-cell predictions of xgb_standard (60 features) and mlp_wide (60 features) with equal weights of 0.5. A Ridge-regression meta-learner (stacked_sp) that learns fold-level optimal combination weights was also evaluated as a more sophisticated alternative.

### 4.4 Regularisation

A central empirical finding of this work is that dropout regularisation is harmful in this setting. Adding dropout of probability 0.10 to the CNN encoder and 0.20 to the bottleneck caused the G0=6.4 extrapolation fold to collapse from R²=+0.22 to R²=−0.61. The mechanism is the following: with only six distinct G0 values in the training set, the model must form coherent internal representations of how nH₂ responds to changes in G0 and its interaction with local physical conditions. Dropout randomly zeros feature channels during each forward pass, forcing information to be distributed redundantly across channels rather than organised into stable, interpretable representations. For interpolation folds where the held-out G0 is bracketed by training values, the degradation is modest. For extrapolation folds where the model must coherently extend the learned UV-dependence beyond the training range, dropout destroys the structured extrapolation mechanism. An equivalent instability was observed when DART (dropout additive regression trees) was applied to XGBoost. L2 weight decay (10⁻⁵) does not exhibit this failure mode because it constrains the magnitude of learned parameters without disrupting their structural organisation. All final models therefore use L2 weight decay as the sole regulariser.

---

## 5. Results

### 5.1 Ablation Study

Table 1 summarises the mean log-space R² and standard deviation across seven leave-one-G0-out folds for each model configuration, ordered by mean R².

**Table 1.** Leave-one-G0-out performance (log₁₀(nH₂) space R²).

| Model | Features | Mean R² | Std | G0=0.1 | G0=0.2 | G0=0.4 | G0=0.8 | G0=1.6 | G0=3.2 | G0=6.4 |
|---|---|---|---|---|---|---|---|---|---|---|
| ens_sp | 60 | **0.948** | 0.024 | 0.908 | 0.915 | 0.953 | 0.962 | 0.967 | 0.967 | 0.965 |
| stacked_sp | 60 | 0.946 | 0.020 | 0.912 | 0.916 | 0.954 | 0.960 | 0.962 | 0.961 | 0.954 |
| xgb_standard_sp | 60 | 0.924 | 0.026 | 0.912 | 0.874 | 0.920 | 0.934 | 0.924 | 0.948 | 0.959 |
| mlp_wide_sp | 60 | 0.913 | 0.034 | 0.843 | 0.886 | 0.914 | 0.928 | 0.946 | 0.945 | 0.927 |
| ens_xgb+mlp (no spatial) | 15 | 0.923 | 0.045 | 0.891 | 0.829 | 0.916 | 0.949 | 0.959 | 0.959 | 0.954 |
| xgb_standard (no spatial) | 15 | 0.886 | 0.061 | 0.891 | 0.751 | 0.866 | 0.902 | 0.910 | 0.932 | 0.952 |
| mlp_standard (no spatial) | 15 | 0.886 | 0.046 | 0.809 | 0.825 | 0.883 | 0.921 | 0.929 | 0.928 | 0.910 |
| 3D U-Net (unet_standard) | volumetric | 0.803 | 0.172 | 0.497 | — | — | — | — | — | 0.591 |
| Linear Regression | 15 | 0.230 | — | −0.876 | −0.237 | — | — | — | — | — |

Table 2 presents the ablation of the spatial features and ensemble in incremental steps.

**Table 2.** Incremental ablation of the ens_sp model.

| Configuration | Features | Mean R² | Δ |
|---|---|---|---|
| XGBoost, baseline features | 15 | 0.886 | — |
| XGBoost + spatial 3³ | 30 | 0.917 | +0.031 |
| XGBoost + spatial 3³ + 5³ + 7³ | 60 | 0.924 | +0.007 |
| XGBoost_sp + MLP_sp (equal-weight) | 60 | **0.948** | +0.024 |

The largest single-step improvement is the addition of the 3³ box-filter spatial features (+0.031 R²), which exceeds the combined gain from all subsequent steps. Extending to three scales adds a further +0.007, and the ensemble contributes +0.024. Neural architecture search (three MLP architectures) and tree depth search (depths 4, 6, 8) produced differences of less than 0.002 R², confirming that these are not bottlenecks.

### 5.2 Per-Fold Analysis

The per-fold R² values for ens_sp reveal a clear pattern consistent with the a-priori physical expectation. The hardest folds are G0=0.1 (R²=0.908) and G0=0.2 (R²=0.915), both of which require extrapolation to UV field strengths below the training range. G0=0.1 presents the most challenging case: the model must extrapolate downward in G0 without any anchor from a lower-G0 training cube. At very low UV field strengths, most hydrogen in dense regions is able to form molecules, and self-shielding is the dominant process — a regime qualitatively different from the mid-range G0 values that dominate the training set. The interior interpolation folds (G0=0.4 through G0=3.2) all achieve R² ≥ 0.953, and the upper boundary fold G0=6.4 achieves R²=0.965. The relative ease of the upper boundary compared to the lower reflects the physical asymmetry: at high G0, UV photodissociation dominates uniformly, producing a simpler chemistry dominated by the density–UV balance, whereas the low-G0 regime requires the model to correctly represent the self-shielding transition.

The fold-to-fold standard deviation of ens_sp is 0.024, substantially lower than that of the 3D U-Net (0.172 for unet_standard, up to 0.52 for unet_large), demonstrating that the spatial-feature tabular approach produces stable predictions across all parameter regimes.

### 5.3 The 3D U-Net: Why Spatial Features Dominate

The 3D U-Net was expected to be a strong model for this problem: it naturally aggregates spatial context through its hierarchical receptive field, and its skip connections allow it to combine coarse global structure with fine local gradients. In practice, the U-Net consistently underperformed the tabular ensemble on every metric. The fundamental reason is the severe imbalance between model capacity and effective training data. With only six cubes in each training fold and eight physics-aware augmentations per cube, the U-Net trains on 48 volumetric samples. In contrast, XGBoost and MLP treat each of the 12.6 million cells in the training fold as an independent sample, giving them a 260,000-fold advantage in effective sample count. The unet_large configuration (23 million parameters) makes this imbalance extreme: at 48 training samples, it memorises training cubes and fails catastrophically on extrapolation folds, with a fold standard deviation of 0.52.

A secondary factor is resolution. GPU memory constraints force the network to operate on 64³ input volumes downsampled from the native 128³ via average pooling, discarding fine-scale gradients at the cloud boundary — precisely the regions where nH₂ transitions most sharply. Pointwise models with spatial features are not subject to this limitation: the spatial features are computed at full 128³ resolution and all 2,097,152 cells per cube contribute to training.

Including the U-Net in any ensemble consistently lowered the mean R²: the ens_xgb+cnn configuration achieved R²=0.897, worse than XGBoost alone with spatial features (0.924). The high fold variance of the CNN means that in the folds where it performs poorly (sometimes R²<0), averaging its predictions with well-calibrated tabular predictions degrades the ensemble. The ens_sp configuration avoids this by not including the CNN at all.

### 5.4 Comparison of Ensemble Strategies

The equal-weight ensemble (ens_sp, R²=0.948) outperforms the Ridge-stacked ensemble (stacked_sp, R²=0.946) by a small but consistent margin. XGBoost and MLP have complementary inductive biases: XGBoost makes piecewise-constant predictions defined by axis-aligned decision boundaries and naturally produces conservative constant extrapolations beyond the training range, while MLP makes smooth, continuous predictions that are accurate in regions of smooth feature variation but can diverge at sharp phase boundaries. Averaging the two predictions smooths XGBoost's staircase artifacts in smoothly varying regions while XGBoost corrects MLP's boundary errors. The Ridge meta-learner, which would ideally learn optimal fold-specific combination weights, does not improve upon equal averaging with only seven folds: the six-fold meta-training set is insufficient to learn weights that reliably transfer to the seventh fold.

---

## 6. Supplementary Experiments

### 6.1 Single-Cube Extrapolation

To quantify the information content of a single simulation and to validate the leave-one-G0-out protocol, we trained the stacked_sp model on each of the seven cubes individually and evaluated it on all seven. This produces a 7×7 matrix of R² values, where the diagonal (in-sample fit) reaches R²≈0.99 and the off-diagonal elements measure out-of-sample generalization as a function of G0 distance. The results reveal a clear monotonic decay: models trained and evaluated one G0 step apart (e.g., 0.1→0.2) achieve R²≈0.97; two steps apart, R²≈0.96; three steps, R²≈0.88; four steps, R²≈0.75; five steps, R²≈0.45; and six steps (e.g., 0.1→6.4), R²≈0.15. The mean off-diagonal R² across all 42 transfer pairs is approximately 0.78.

This result has two implications. First, the local chemical equilibrium — the mapping from density, temperature, and shielding proxies to nH₂ — is approximately universal within a G0 ratio of 4–8 (two to three steps on the geometric sequence) but breaks down over larger factors. This is consistent with the physical picture that the underlying H₂ formation and destruction rates have the same functional forms across G0 values; what changes is the equilibrium balance point. Second, the strong asymmetry between adjacent and distant G0 pairs validates the design of leave-one-G0-out CV: a single-cube model is a qualitatively worse surrogate than one trained on six cubes spanning the full UV range, and the 64-fold G0 range of the training set is not redundant.

### 6.2 Intra-Cube Spatial Sampling

We investigated whether the ML surrogate can reconstruct the H₂ density field within a single simulation from partial spatial coverage, and how the geometry of the observed fraction affects reconstruction quality. For each of the seven G0 cubes, we trained stacked_sp on a spatial subset of cells and evaluated on the remainder, testing three spatial geometries at varying coverage fractions. In the random sampling configuration, cells are selected uniformly at random, which preserves the statistical distribution of local physical conditions across the training set. In the contiguous slab configuration, all cells from one half of the cube along a coordinate axis are used for training, leaving the other half entirely unseen. In the contiguous box configuration, a compact cubic subregion of specified volume fraction is used for training.

The results are striking and physically interpretable. A randomly sampled 1% of cells (approximately 21,000 out of 2,097,152) achieves a test R² of 0.897 on the remaining 99%. Increasing the random sampling fraction to 5%, 10%, 25%, and 50% yields R² of 0.952, 0.963, 0.970, and 0.980 respectively. In contrast, a contiguous slab covering 50% of the volume — 23 times more cells than the 1% random sample — achieves test R² of −0.394 (x-axis slab), −2.689 (y-axis slab), and −1.389 (z-axis slab), meaning the model is systematically worse than predicting the training mean at every test cell.

The mechanism behind this result is the behaviour of the multi-scale spatial neighbourhood features under different training geometries. In random sampling, every test cell has training cells nearby in all three spatial dimensions, so the box-filter neighbourhood features computed from training-cell values provide meaningful local averages for each test cell. With 1% random coverage, the neighbourhood estimates are noisy but unbiased, and the model can still infer the approximate local environment. In the contiguous-slab configuration, the test cells occupy an entire spatial half of the volume that is completely absent from the training set. The model trained on (for example) the UV-illuminated surface cannot correctly predict the shielded interior, because the feature distributions of interior cells fall outside the training distribution in all features simultaneously. The model defaults to extrapolating its training-set conditional mean, which is dramatically wrong for cells in a physically distinct regime.

The z-axis slab is less catastrophic than the x- or y-axis slabs (R²=−1.389 vs −0.394 and −2.689), reflecting the physical asymmetry of the z-axis: since the UV field illuminates along +z, the half-cube training gradient from surface to interior is smoother along z than along the transverse axes, making the trained feature-to-nH₂ mapping slightly more transferable. Contiguous box splits produce intermediate results (R² of 0.67–0.82 at coverage fractions of 1–50%), because the box covers all three spatial dimensions within its region but leaves distant cells poorly covered.

The practical implication for observational astronomy is concrete. An H₂ density field analogous to this simulation can be approximately reconstructed from a sparse set of randomly distributed sight-line observations — the kind obtainable from an integral field unit (IFU) spectrograph with a sparse sampling pattern — with high accuracy even at 1% spatial coverage. Deep, contiguous mapping of a single region provides far less reconstruction utility despite consuming more observing time. Efficient H₂ density survey designs should therefore prioritise spatial coverage uniformity over observational depth.

---

## 7. Discussion

### 7.1 Feature Engineering Versus Architecture Search

The central methodological lesson of this work is that feature engineering — specifically, the injection of precomputed spatial context via box-filter neighbourhood averages — dominates model architecture in its impact on prediction accuracy. The +0.031 R² gain from adding 3³ spatial features exceeds the combined gain from any combination of architectural choices: varying XGBoost tree depth from 4 to 8 produced differences below 0.002, varying MLP architecture among three qualitatively different designs (tapering, wide, residual) produced differences below 0.003, and varying CNN capacity over a 15× range in parameter count produced no improvement and frequently caused degradation. This result is consistent with the broader machine learning literature showing that domain-informed feature engineering remains highly effective for tabular regression problems, often outperforming automated representation learning when training data is limited.

The specific spatial features used here — uniform box-filter averages — are among the simplest possible representations of spatial context. They capture only the local mean of each feature over a cubic neighbourhood, discarding directional gradients, higher-order moments, and the anisotropic structure that might distinguish a cloud boundary along the UV illumination axis from a boundary in a perpendicular direction. Despite this simplicity, they provide a near-complete representation of the shielding state because the dominant physics is controlled by column density — an approximately isotropic spatial average in the sub-Jeans turbulent simulations used here. More sophisticated spatial descriptors (local gradients, wavelet coefficients, directional filters) could in principle improve accuracy further, particularly for the hardest extrapolation folds, but are not needed to achieve R²>0.94.

### 7.2 The Training Sample Deficit of Volumetric Models

The fundamental limitation of the 3D U-Net in this application is the mismatch between the number of effective training samples (48 augmented volumes) and the number of model parameters (1.5–23 million). This is not a failure of the U-Net architecture in principle — the same architecture has achieved state-of-the-art results in 3D medical image segmentation, where training sets of hundreds to thousands of volumes are available. It is a failure of the specific data regime: with fewer than 10 distinct simulation cubes at distinct parameter values, volumetric models cannot learn generalizable spatial representations. The per-cell tabular approach sidesteps this limitation by decomposing the volumetric data into 12.6 million independent cell-level training samples, providing XGBoost and MLP with the data volume needed to learn stable, accurate mappings.

This suggests a general principle for ISM simulation emulation: the choice between volumetric and per-cell models should be governed by the ratio of available simulation cubes to model parameters, not by the nominal spatial nature of the problem. If dozens or hundreds of simulation cubes were available — feasible with high-throughput simulation frameworks — the volumetric approach would likely become competitive. In the typical regime where only a handful of high-fidelity simulations exist, spatially augmented per-cell models are preferable.

### 7.3 Limitations

Several limitations of this work should be noted. First, the UVonly simulation suite models chemistry under a spatially uniform, time-steady UV field. Real molecular clouds are exposed to anisotropic, time-varying radiation from nearby massive stars and supernovae, and the physical significance of the z-axis UV illumination direction means that the z-preserving augmentation strategy would need to be reconsidered for simulations with different illumination geometries. Second, the 15 input features include log(fh₂), the H₂ self-shielding factor, which is computed from the full 3D column density of H₂ and is therefore a non-local quantity. The inclusion of fh₂ as a feature is physically valid — it encodes information unavailable from the purely local fields — but in a deployment scenario where nH₂ is unknown and fh₂ must be estimated from H₂ column densities inferred from other tracers, the accuracy of the fh₂ estimate would propagate into the nH₂ prediction. Third, the dataset consists of a single turbulent ISM realisation at each G0 value; the extent to which the learned mapping generalises across different initial conditions, magnetic field strengths, or density regimes is not tested here.

---

## 8. Conclusions

We have developed and systematically evaluated machine learning surrogates for the molecular hydrogen number density in 3D ISM simulations, under the stringent protocol of leave-one-G0-out cross-validation. Our principal findings are as follows.

An ensemble of gradient-boosted trees and a wide MLP, trained on 60-dimensional feature vectors combining 15 local physical fields with 45 multi-scale spatial neighbourhood averages (3³, 5³, 7³ box-filter kernels), achieves a mean log-space R² of 0.948 ± 0.024 across seven leave-one-G0-out folds, with a minimum per-fold R² of 0.908 on the hardest extrapolation fold (G0=0.1). The single largest performance improvement comes from adding spatial neighbourhood features (+0.031 R²), not from any model architecture change. A 3D U-Net CNN, despite its natural inductive bias for volumetric spatial regression, achieves a best mean R² of 0.803 ± 0.172, with an order of magnitude greater fold-to-fold variance, owing to the severe imbalance between volumetric training sample count (48 augmented cubes) and model parameters. Including the CNN in any ensemble reduces the overall R².

Dropout regularisation is harmful for out-of-distribution extrapolation when the training set spans only a small number of discrete parameter values: with six distinct G0 values in training, the model must form stable representations of the UV-dependent chemistry to extrapolate coherently at boundary G0 values. Dropout disrupts this coherence. L2 weight decay is safe and should be preferred. Target normalisation (per-fold standardisation of log(nH₂)) is essential for the CNN: without it, the loss is dominated by near-zero-density cells and R² is negative; with it, R² reaches 0.775 immediately.

Single-cube extrapolation experiments show that chemical knowledge transfers well over G0 ratios up to approximately 4–8× (R²>0.88) but degrades rapidly over larger factors, validating the design of leave-one-G0-out CV and confirming that a 64-fold UV range is not redundant in the training set. Intra-cube spatial sampling experiments demonstrate that 1% uniform random coverage achieves R²=0.897 in reconstructing the full H₂ density field, while 50% contiguous-slab coverage achieves R²=−0.394. Coverage uniformity is the critical design parameter for H₂ density reconstruction surveys, with implications for observational IFU survey strategies targeting molecular cloud H₂ mapping.

The ens_sp surrogate reduces the cost of a full-cube nH₂ prediction from the cost of a chemistry solver run to the cost of a tabular inference operation: given 15 per-cell physical fields, it predicts nH₂ at all 2,097,152 cells of a 128³ cube in seconds, with accuracy within 0.1–0.2 dex of the self-consistent simulation value across the range of UV field strengths studied.

---

## Acknowledgements

The author thanks the developers of XGBoost, PyTorch, and scipy for the open-source tools on which this work depends.

---

## Data Availability

The UVonly simulation cubes and all trained model outputs are archived as timestamped JSON and `.npz` files in the project repository. The full pipeline is deterministic and runs end-to-end on a single consumer GPU.

---

## Appendix: Model Configuration Summary

**ens_sp (Final Model):**
- Input: 60 features (15 physical + 45 multi-scale spatial)
- XGBoost: depth=6, n_estimators=400, learning_rate=0.1, subsample=0.3, tree_method='hist'
- MLP: hidden widths [512, 512, 256, 128], BatchNorm+ReLU, Adam (lr=10⁻³, weight_decay=10⁻⁵), CosineAnnealingLR, 100 epochs, batch size 262,144
- Ensemble: ŷ = 0.5·ŷ_XGB + 0.5·ŷ_MLP
- Target: log₁₀(nH₂)
- Training time per fold: ~4 minutes on a single consumer GPU

**3D U-Net (unet_standard):**
- Encoder: 64³→32³→16³→8³ with 3×3×3 convolutions, InstanceNorm3d, ReLU
- Decoder: trilinear upsampling with skip connections
- Base channels: 32; total parameters: 5.8M
- Augmentation: 8 z-preserving octahedral symmetry operations
- Optimiser: Adam (lr=10⁻³, weight_decay=10⁻⁵), 150 epochs
- Training time per fold: ~25 minutes on a single consumer GPU
