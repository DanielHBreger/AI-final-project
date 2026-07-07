# Dispositions: dual-AI review of 2026-07-05 ("Claude" + "ChatGPT" reports)

Both reviews were written **before** runs 10–11 (2026-07-05) and today's §6
rewrite; several §6 comments were already resolved by those. Every point
below is dispositioned: **FIXED** (in paper.tex today), **RUN** (queued in
RUN_PLAN), **DOCUMENTED** (tracked, needs user/collaborator action),
**DEFENDED** (no change, rationale given), or **RESOLVED-PRIOR**.

All numbers quoted in fixes were computed from existing artifacts
(run-1b JSON, run-5 npz volumes) — see `results/deployed_row_metrics.json`
and `logs/deployed_row_metrics.log`.

## Internal consistency (Claude)

1. **7 vs 9 per cent mass discrepancy** — FIXED. Real: §7.3's 7 % is the
   Table 3 shortcut row (0.93–1.07); the abstract/§5.5/conclusions 9 % is
   the nested pipeline (0.985–1.090). §7.3 now labels both instances
   explicitly. A full line-level numbers audit (Claude's offer) remains a
   recommended pre-submission pass — added to RUN_PLAN follow-ups.
2. **W₁ contradiction (§7.3 vs §5.5)** — FIXED. Both W₁ values are computed
   identically (same `compute_metrics`), so the deployed pipeline
   (mean W₁ = 0.08 dex, recomputed today = 0.082) genuinely beats the
   weighted MLP (0.109). §5.1 and §7.3 now say the nested pipeline has the
   best marginals overall; the MLP is best *among single models*, and the
   narrowing claim is scoped to the shortcut stacks.
3. **0.990 appears in no table** — FIXED. A "Deployed stack (+S+W), nested,
   mass-cal." row was added to Tables 2 and 3, set off by a rule and
   excluded from per-column bolding (see Defended #1 for why the shortcut
   rows stay in the main text). Every abstract number now has a table home.
   Deployed Table-3 row: RMSE 0.250, b +0.02, s 0.24, M 0.99–1.09,
   R²_mol 0.990, R²_dif 0.90, f₀.₁ 0.73, W₁ 0.08, skill +0.74.

## Protocol issues (both reviewers)

4. **U-Net checkpoint selection on the held-out cube** — FIXED
   2026-07-06/07 (runs 16b + 16c, central inner-val rule; RUN_PLAN run 16
   entry has full numbers). All U-Net numbers in `paper.tex` and
   `paper_short.tex` now come from leakage-free selection: 15-input
   0.963 ± 0.047 (RMSE 0.348, skill +0.36), 11-input 0.851 ± 0.121. The
   ≤12 % disclosure is replaced by the protocol description in §4.2.3
   plus a rule-sensitivity disclosure (nearest-rule 0.914 vs central
   0.963; §7.5). Selection vs final epoch is a no-op; the
   interior-champion result survives (interior R² 0.992 vs stack 0.984),
   the edge/mass failures worsen slightly.
5. **Shortcut vs nested stacking in the main tables** — PARTIALLY DEFENDED.
   Accepted: deployed nested row added to Tables 2–3 and made the source of
   all abstract/conclusion headline numbers (it already was). Defended:
   the 17-variant comparison stays under the uniform CV-OOF shortcut
   protocol and is NOT moved to an appendix — running all 17 variants fully
   nested would multiply training cost ~6× for no inferential gain, and a
   uniform protocol is what makes the variant comparison internally fair.
   §4.4 discloses the difference; the nested pipeline scoring *higher*
   makes the shortcut rows conservative.
6. **Mass-calibration mathematics (ChatGPT #3)** — FIXED. §4.5 now gives
   the exact fitted functional (mass-weighted mean residual, formula
   inline), states that exact per-cube closure would fit the log mass
   ratio instead, notes first-order equivalence, and cites the
   out-of-sample validation (fold mass 0.985–1.090). DEFENDED: we keep the
   mass-weighted-residual functional (not switching to log-mass-ratio) —
   it is what the released pipeline implements, the difference is
   empirically negligible here, and switching would invalidate all saved
   volumes for a cosmetic gain. An optional analysis item (RUN_PLAN A4)
   quantifies the difference if a referee asks.
7. **Calibration stability with 6 points (both)** — FIXED, no run needed.
   The npz volumes store `per_cube_bias_mw` + fit coefficients;
   leave-one-training-cube-out refits move the applied offset ≤ 0.04 dex
   (≤ 10 % in mass), worst at the two extrapolating boundary folds. One
   sentence added to §5.2. Full per-fold audit numbers are in
   `logs/deployed_row_metrics.log` if a table is ever wanted.
8. **Clip activity for the final model (ChatGPT #8)** — FIXED, verified.
   Zero cells clipped in all 7 deployed folds (checked against truth
   today). Sentence added to §5.5; Table 3 caption states it. The U-Net's
   clipped-lower-bound caveat now carries a † on the 0.81–172 cell.

## Statistics with n = 7 (Claude)

9. **17-variant multiplicity / RMSE photo-finish** — FIXED. New §5.1
   passage states the noise threshold (≲0.01 R², ≲0.02 dex RMSE) and
   reports the per-fold consistency the selection actually rests on
   (computed from run-1b): mass-cal beats mean-cal on RMSE in only 3/7
   folds (photo-finish acknowledged) but on |log mass| in 7/7 folds
   (0.93–1.07 vs 0.71–0.89), and beats the best single model on RMSE in
   6/7 folds. This also satisfies §3.2's promised per-fold-consistency
   check.
10. **Cell-level bootstrap CIs** — RUN (cheap analysis item A2, minutes on
    saved volumes). Not yet in the text; low risk either way.

## Provenance / release (both, pre-existing blockers C1–C3)

11. **Code release (Claude)** — DOCUMENTED, user decision. GitHub +
    Zenodo DOI for `results/` + `predictions/` is an afternoon and closes
    review blocker C2. Nothing in the pipeline is proprietary. Strongly
    recommended before submission.
12. **Simulation provenance table (both, with ChatGPT's field list)** —
    DOCUMENTED. Remains blocker C1, waiting on collaborators; ChatGPT's
    two table templates (simulation metadata; per-feature provenance
    column incl. n_H+/T/ext origins and the `ext` formula) copied into
    PAPER_UPDATE_INSTRUCTIONS as the concrete deliverable. The "deployable"
    labels in Table 1 are contingent on the same answers.

## Runs requested by reviewers

13. **Six-feature minimal set (`--no-vel --no-B`)** — RUN. Pre-existing
    run 8, elevated from optional to recommended (converts §5.3's
    speculation into a table row; feature-importance already predicts the
    outcome).
14. **Density-weighting α sensitivity (ChatGPT #6)** — RUN. Pre-existing
    run 12 (α ∈ {10, 100, 1000} on the weighted XGB), elevated: the
    weighted stack is now the headline, so the untested hyperparameters
    are a fair target.
15. **Phase-threshold sensitivity [−5, −3] (both)** — RUN (analysis item
    A1, minutes on saved volumes, no training).
16. **Intra-cube random masks: one seed per fraction (ChatGPT #9)** — RUN
    (new run 17, seeds ×3 on the rand splits). Note the §6.2 numbers were
    fully regenerated today (run 11) — the reviewers saw the old
    (irreproducible) March numbers; the instability they might worry about
    is much smaller in the current results (rand_1 range 0.79–0.94).

## Text/figure items

17. **Sparse-label caveat for §6.2 claims (ChatGPT #9)** — FIXED in
    abstract and conclusion (viii): "cell-local inputs available
    everywhere, labels and neighbourhood context restricted to the
    sample."
18. **11-input U-Net prominence (Claude)** — FIXED: split out of
    conclusion (iv) into its own bullet (v), framed as quantifying
    chemistry-vs-memorised-morphology; conclusions renumbered (now
    i–viii).
19. **Solver-cost honesty (Claude)** — FIXED: closing paragraph now states
    the solver's per-cube cost is unavailable and the speed-up over this
    suite's solver is not quantified.
20. **Wording table (ChatGPT)** — FIXED: "textbook answer"→"natural
    modelling choice"; "numerically useless"→"severely ill-conditioned";
    "morphology yes, chemistry no"→formal phrasing (§5.3; the conclusions
    version was already formal); "directly observable ones"→
    "solver-independent ones" (Fig 2 caption); "worst extrapolator"→
    "least reliable extrapolator under the physical-field metrics" (§5.4
    and conclusion vi); "fails entirely" was already gone (runs 10–11
    rewrite); H₂ "coolant"→"dominant molecular reservoir".
21. **Table 3 U-Net mass footnote (Claude)** — FIXED († marker).
22. **Figure 3 axis clipping (Claude)** — DEFENDED, verified: inspected
    the PNG; panel (a)'s floor (0.88) shows the U-Net's 0.905 minima and
    panel (c) reaches −1.8, fully containing the U-Net's worst skill
    (−1.15 at G₀ = 0.1). Nothing is clipped; no caption change needed.
23. **Figure 5 naming (ChatGPT)** — ALREADY CORRECT: caption and text
    already say "XGBoost gain importance"; the importance figure is
    explicitly paired with the ablation as the reliability check.
    Permutation importance noted as possible future work, not required.
24. **Figures 11–12 heatmap colour scaling (ChatGPT)** — FIXED 2026-07-06
    (analysis item A3): both heatmaps replotted in place from the saved
    run-10/11 JSONs via a new `--replot` mode (RdBu diverging map clipped
    to [−1, 1], white at 0, cells below −1 hatched with the true value
    printed, in-cell text enlarged to 8 pt); clipping note added to both
    captions. No retraining.
25. **KS significance (ChatGPT, Fig 9)** — ALREADY CORRECT: the caption
    reports KS D and W₁ as effect sizes; no p-value significance claim is
    made.
26. **References: Palud et al. 2023 (A&A, Meudon PDR emulation); Janssen,
    Branca & Buck 2026 (A&A, surrogate benchmarking)** — FIXED 2026-07-06:
    both verified against arXiv/journal records and inserted. Palud et al.
    2023, A&A 678, A198 (doi 10.1051/0004-6361/202347074) cited in the §1
    emulator list; Janssen, Branca & Buck 2026, A&A 708, A227
    (doi 10.1051/0004-6361/202658937) cited in the same paragraph for
    systematic surrogate benchmarking. The "zero-dimensional" sentence
    gained a one-dimensional parenthetical for the PDR emulator. Same
    pass re-verified KMT09 (ApJ 693, 216–235), Sternberg2014 (ApJ 790,
    10) and Bialy2016 (ApJ 822, 83) — all already correct in the bib.
27. **Abstract ablation attribution (Claude #15)** — FIXED: §1(iv) now
    says "for the adopted configuration in the comparison protocol of
    Table 4"; the abstract keeps the bare numbers (they now have a table
    home).

## Defended without change

- **Metric-level clip design** — unchanged (DESIGN_DECISIONS §1): rarely
  binds for classical models (now *proven* never active for the deployed
  volumes), conservative for the U-Net claims; disclosed.
- **Leaving the shortcut comparison in the main text** — see #5.
- **Keeping the mass-weighted-residual functional** — see #6.
- **PHASE_SPLIT = −4** — retained pending analysis item A1; the Fig 1
  bimodality argument stands, and A1 will add the [−5, −3] stability
  sentence rather than changing the threshold.

## Claude's overall-quality remarks and ChatGPT's "what improved" sections

No action required; noted that both reviewers independently confirm the
metric-hierarchy thesis as the paper's contribution — consistent with
today's §6.2 rewrite, which now uses that same hierarchy internally.
