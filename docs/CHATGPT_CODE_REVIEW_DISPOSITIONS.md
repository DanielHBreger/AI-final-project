# ChatGPT code-review dispositions (2026-07-07)

A static code review requested by the user, covering the training
pipeline, CNN code, stacking, preprocessing, metrics, prediction
generation, supplementary experiments, and the audit docs themselves.
Every specific claim below was checked directly against the code before
disposition — most were confirmed exactly; a few carry an important
nuance the review missed (mainly: several "bugs" are documented design
choices in `DESIGN_DECISIONS.md`/the paper, not oversights). One factual
error: the review implies the README doesn't explain the leave-one-G0-out
rationale, but `README.md` does — that part of its praise was accurate,
it just also (correctly) caught the README's headline number being stale.

Severity labels below are the review's own; disposition is mine.

## Fixed same session (2026-07-07)

1. **Critical — README headline U-Net number stale (0.974±0.035).**
   CONFIRMED exactly (`README.md` line 18 pre-fix). Now reads
   0.963±0.047 (central-rule, leakage-free, matching the paper), and the
   "Quick start" CNN commands carry a warning pointing at `test_cnn.py`
   as the only leakage-free path. FIXED.

2. **Critical — `test_cnn.py` CLI defaults to `--inner-val-rule nearest`,
   not the adopted `central`.** CONFIRMED exactly. Default flipped to
   `central`; module docstring rewritten to state which rule is adopted
   and why (the `nearest` rule turns boundary folds into two-step
   extrapolation — the run-16-vs-16b confound). FIXED.

3. **High — `fh2` code comment contradicts the design audit.**
   CONFIRMED: `data_loader.py` said "not data leakage." Replaced with a
   comment matching `DESIGN_DECISIONS.md`'s actual position (solver-
   internal, target-adjacent, quantified by ablation, no-fh2 config is
   the deployable one). FIXED.

4. **Medium — `single_cube_extrapolation.py` / `intra_cube_section.py`
   docstrings call the model "stacked_sp" (implying unweighted) when
   `_fit_xgb`/`_fit_mlp` always density-weight.** CONFIRMED exactly
   (`model_helpers.py` docstrings state the helpers always weight).
   Both scripts' docstrings corrected to name the weighted variant
   explicitly. FIXED.

5. **High — `mass_weighted_bias` docstring claims exact mass closure.**
   CONFIRMED: said "centres... on the true mass" (no qualifier). This is
   only a first-order approximation — `calibration_functionals.py`
   already quantifies the gap (≤0.018 dex on the deployed volumes, paper
   §4.5) but the helper's own docstring hadn't caught up. Docstring
   rewritten to state the approximation explicitly and point at the
   exact functional. FIXED.

6. **Critical — leaking checkpoint selection remains live in
   `train_cnn.py` and `compare_architectures.py --cnn`.** CONFIRMED
   exactly: both still pick the epoch with lowest loss on the outer
   held-out cube (`train_cnn.py` `best_state`/`val_ds`;
   `compare_architectures.py` `_train_cnn_fold`'s `val_ds` is the test
   cube). NOT deleted (see "Declined / deferred" below) — instead both
   entry points now carry an explicit, loud docstring warning that they
   are not leakage-free and that every reported U-Net number comes from
   `test_cnn.py`. FIXED (warning), not unified (see below).

7. **Medium — aggregate feature-importance error bars in
   `plot_feature_importance.py` assume independent features
   (`sqrt(sum(std_i^2))`).** CONFIRMED exactly. Rewritten to sum
   per-fold, per-field importances first (7 aggregate values), then take
   the mean/std of that vector — respecting the fold-to-fold correlation
   between a field's local and multi-scale spatial importances. Means
   are unchanged (verified: log_fh2 0.8305 identical before/after); only
   the error bars move, non-uniformly (log_fh2's honest bar is *wider*,
   0.028 vs 0.021; log_T's is *narrower*, 0.004 vs 0.008) — consistent
   with genuine cross-feature correlation, not a directional bug. The
   paper only ever cites the means (0.57/0.83 for log_fh2), never these
   error bars, so no paper number changes.
   `figures/fig_feature_importance.png` regenerated.

## Confirmed, real, deliberately deferred (not done tonight)

8. **High — no run manifest; `predictions/pred_g0_*.npz` selection picks
   the lexically-latest file per G0 independently, so a partial rerun
   can silently mix model configurations across folds.** CONFIRMED
   exactly in both `statistical_analysis.py`'s `_find_npz` and
   `bootstrap_cis.py`'s `latest_per_g0` (duplicated logic, not shared).
   This is the single highest-value structural fix the review
   identified. NOT done: it requires a coordinated change to the writer
   (`predict_and_visualize.py`) and every reader, plus deciding what a
   manifest schema looks like — a deliberate refactor, not a drive-by
   edit, and touching it blind risks breaking the exact prediction files
   the current paper figures were generated from. Added to RUN_PLAN as a
   concrete backlog item (below).

9. **High — intra-cube masked spatial mean is not renormalised by
   observed-neighbour count (`intra_cube_section.py`).** CONFIRMED
   mechanism exactly: held-out cells are zeroed, then `uniform_filter`
   is applied with no coverage denominator, so a window with 1-of-27
   observed cells reports value/27, not the observed cell's own value.
   **Important nuance the review missed:** this is a *documented,
   deliberate* choice, not an oversight — the paper's Eq. 8 states
   explicitly "no renormalisation... deliberately conservative," and the
   docstring now added to `intra_cube_section.py` says the same. The
   review's "correct calculation" (`filter(XM)/filter(M)`) answers a
   different question — best-possible estimate from observed neighbours
   with an oracle coverage mask — which is a legitimate thing to *also*
   report, but changing the existing calculation would silently change
   what runs 10/11 and the sparse-coverage numbers in §6.2 mean. Left
   as-is pending an explicit decision to add the normalised variant as a
   second, clearly-labelled experiment rather than replace the current
   one.

10. **High — `mass_ratio` clips predictions using the held-out truth's
    own range (`clip_lo = yt.min()-1`, `clip_hi = yt.max()+1` in
    `compute_metrics`).** CONFIRMED exactly. Already extensively
    disclosed (paper §3.2, Table 3 dagger footnote, DESIGN_DECISIONS:
    "clip is a lower bound on the violation"). The review's proposed fix
    (report `mass_ratio_raw` as the primary metric) is reasonable but
    would touch the semantics of the primary metric in every table in
    the paper — not a change to make unilaterally after the paper is
    otherwise finalized. Deferred; flagged for a future revision if a
    reviewer asks for it explicitly.

11. **High — main comparison-table stacking is the CV-OOF shortcut, not
    fully nested.** CONFIRMED and already the single most-discussed item
    in `paper/REVIEW3_DISPOSITIONS.md` (#5) and DESIGN_DECISIONS —
    the deployed pipeline (`predict_and_visualize.py`) already uses fully
    nested stacking and its numbers score *higher* than the shortcut
    table rows, which is disclosed as making the table conservative.
    Making nested stacking canonical everywhere would mean re-running
    the entire 17-variant `compare_architectures.py` comparison nested
    (hours, and every base model refit per outer fold) — a real cost,
    not the "pay compute once" the review implies, since today only the
    single deployed configuration is nested. Deferred as a candidate for
    a future full rerun, not a same-session fix.

## Confirmed but lower priority / declined as stated

12. **Medium — U-Net trains on 5 cubes (train + inner-val) while
    tabular models train on 6.** Real asymmetry. My first-pass dismissal
    ("reintroduces epoch-count guessing") was WRONG and retracted in
    round 2 below — the proposed procedure (fix the epoch count via
    inner-val, discard that model, retrain fresh on all 6, evaluate
    once) is a standard model-selection/final-refit split, not epoch
    guessing. Reclassified as a genuine, not-yet-implemented fairness
    improvement; see round 2 disposition.

13. **Medium — InstanceNorm plausibly erodes the constant `log_G0`
    conditioning channel; FiLM-style conditioning suggested.** Plausible
    and interesting, but speculative architecture research with no
    quantitative claim behind it, already implicitly future work.
    Not a code-review action item; noted for future exploration.

14. **Medium — boundary treatment (`reflect` filter vs. zero-padded
    CNN) doesn't match unconfirmed simulation periodicity.** Already
    tracked as RUN_PLAN optional run 15, blocked on confirming the sims
    are periodic (a collaborator question, not a code fix). No action.

15. **Medium — XGBoost fits a `StandardScaler` it doesn't need (tree
    splits are scale-invariant).** True in principle. Not changed:
    removing it now would make every archived comparison log
    non-bit-reproducible from current code for zero accuracy gain, this
    late in the process. Left as a known no-op.

16. **Medium/reproducibility — `compute_data_checksum` hashes only
    filename+size+first 8KB, not full file content.** CONFIRMED exactly.
    Genuine gap (a same-size later-byte edit would go undetected), and a
    real SHA-256 pass is cheap given training dominates runtime. Not
    done tonight (would invalidate the checksum recorded in every
    archived log without re-running anything, purely a bookkeeping
    change with no scientific consequence) — backlog item.

17. **Reproducibility gaps — no `requirements.txt`/`pyproject.toml`, no
    dataset-shape assertions in `cube_to_volumes`, minimal
    `.gitignore`, thin test coverage outside `smoke_test_metrics.py`.**
    All confirmed and all reasonable software-engineering asks. None
    affects any current scientific claim in the paper. Backlog items,
    not scientific-correctness fixes.

## Backlog (added to RUN_PLAN.md as informal follow-ups, not numbered runs)

- Run/prediction manifest (item 8) — needs `predict_and_visualize.py` +
  every reader touched together.
- Coverage-normalised intra-cube spatial mean as a *second*, explicitly
  labelled variant (item 9) — do not replace the existing one.
- `mass_ratio_raw` (uncalibrated by the held-out clip) as an explicit
  companion field in `compute_metrics`'s output dict (item 10).
- Full SHA-256 data fingerprint (item 16).
- `requirements.txt`/environment pin, dataset validation assertions,
  broader `.gitignore` policy, protocol-level tests (CV isolation,
  spatial-indexing invariance, augmentation transform correctness,
  nested-stacking isolation) (item 17).

## Round 2 (ChatGPT's response to the above, 2026-07-07)

ChatGPT re-checked the round-1 fixes against the actual repo (not just
the disposition text) and found one real miss, agreed with five fixes as
complete, revised its own original position on one item after reading my
counter-argument, and pushed back on three of my dispositions. Each
re-verified directly before acting.

**Real miss, fixed immediately:** `test_cnn.py`'s `--inner-val-rule`
parser default was still `default='nearest'` — I had rewritten the
docstring above it to say "defaults to central" without touching the
actual `argparse` line, so code and documentation contradicted each
other (worse than the original bug). Confirmed by direct read, fixed:
`default='central'`.

**Confirmed complete, no changes:** README correction, `fh2` comment,
`mass_weighted_bias` docstring, feature-importance error bars,
weighted-model labelling in the two supplementary scripts.

**I revise my own original review:** the intra-cube zero-masking item —
ChatGPT's round-1 "mathematically wrong" claim is withdrawn once it saw
the paper explicitly defines the no-renormalisation behaviour (Eq. 8) as
deliberate. Agreed on both sides; no code change, matches item 9 above.

**Accepted and acted on:**
- The two leaking CNN paths (`train_cnn.py`, `compare_architectures.py
  --cnn`) needed "mechanically difficult to misuse," not just a
  docstring warning someone can scroll past. Added a `--allow-leaky-cnn`
  flag to both; without it, both scripts print the leakage warning to
  stderr and exit(1) before loading any data. Smoke-tested: the gate
  fires immediately, `--help` still parses cleanly, and passing the flag
  runs the script exactly as before.
- `compare_architectures.py`'s architecture-selection comment still
  quoted a stale CNN result (`R2=0.803`) predating every subsequent CNN
  rerun, same class of error as the README miss. Removed the number;
  points readers at `test_cnn.py`/RUN_PLAN instead.
- README's "every experiment writes... package versions and a SHA-256
  fingerprint" overclaimed: only `compare_architectures.py`/`train_cnn.py`
  record that metadata; the supplementary scripts (verified by grep, no
  `compute_data_checksum` call outside those two plus `data_loader.py`
  itself) do not. Reworded to state this precisely.
- The "nested scores higher, therefore the shortcut table is
  conservative" claim, which appears in both papers (not just my
  disposition doc), is a real overclaim: empirically nested > shortcut,
  but other details also differ between those two runs (training
  schedule, data splits within the outer fold), so nesting is not
  isolated as the sole cause and the conclusion doesn't follow.
  Reworded in both `paper.tex` and `paper_short.tex` to state the
  empirical fact (nested scores higher) without the unsupported causal
  claim (shortcut is therefore conservative) — now says the stricter
  protocol does not *contradict* the comparison-table numbers, which is
  all the evidence actually supports.
- Checksum backward-compatibility: I had deferred a full SHA-256 pass
  partly because it would "invalidate" existing checksums. ChatGPT
  correctly pointed out this doesn't require invalidating anything —
  add `sha256_full` alongside the existing `sha256_partial`, old logs
  stay interpretable under the old key. Implemented in
  `compute_data_checksum` (both hashes now computed in one pass over
  each file; smoke-tested, ~1.7 s for all 7 cubes, negligible next to
  the minutes-long CSV parse). Reason for deferring was wrong; the
  underlying priority call (low, but now free to do) stands.

**Reclassified after reconsideration:** the U-Net 5-vs-6-training-cube
asymmety (item 12). ChatGPT's proposed procedure — use inner-val to pick
the epoch count, discard that model, retrain fresh on all 6 outer-training
cubes for exactly that many epochs, evaluate once — is a standard
model-selection/final-refit split (train/val to pick a hyperparameter,
then refit on train+val). My original objection ("reintroduces epoch
guessing") conflated this with reusing the held-out cube itself, which it
does not do. This is a legitimate, not-yet-implemented fairness
improvement to the U-Net protocol, not something to decline. NOT
implemented this session (it changes the CNN training entry point and
would need a fresh ~4.6 h run to produce numbers under the new
procedure) — added to RUN_PLAN backlog explicitly as item 23, separate
from the already-existing items 18–22.

**Declined, with reasons stated:** removing XGBoost's `StandardScaler`
(ChatGPT now frames this as a "should eventually" item, not urgent;
agreed, unchanged), rerunning the full comparison table fully nested
(ChatGPT now says this is "future research/cleanup," matching item 11's
disposition; unchanged), replacing the intra-cube experiment or
redesigning the U-Net architecture before finishing the project (ChatGPT
explicitly agrees these should not happen now).
