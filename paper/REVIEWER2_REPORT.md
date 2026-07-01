# Reviewer 2 Report and Publication Action Plan

**Manuscript:** *Machine-learning surrogates for molecular hydrogen density in
three-dimensional interstellar medium simulations using multi-scale spatial features*
**Review date:** 2026-07-01
**Target:** a leading refereed journal (MNRAS-class)

---

## Overall assessment

The science is sound, the evaluation protocol (leave-one-G0-out) is genuinely
stringent and well motivated, the honesty about limitations is above the norm,
and the numbers check out: I independently verified every value in Table 1
against the archived run logs (`arch_comparison_20260311_125636.json` for the
tabular rows, `cnn_test_20260323_010315.json` for the U-Net row, per-fold values
included) and the f_H2 ablation against `ablation_xgb_withfh2/nofh2.json`. No
transcription errors were found. The writing is clear and largely
journal-ready.

The paper is **not submittable today** for three reasons, in decreasing order
of severity:

1. **The simulation suite is essentially undescribed** (Section 2.1 explicitly
   declines to give code, box size, resolution, metallicity, dust model). No
   astrophysics referee will accept this; see item C1.
2. **Table 2 contained "(runs in progress)" placeholders.** ~~Resolved during
   this review~~ — the missing runs were executed and the table is complete
   (item D1).
3. **The error analysis of Section 5.4 is performed on a different (weaker)
   model than the headline model** (30-epoch inner MLP vs 100). A referee will
   ask why the "final model" analysis is not of the final model (item D2).

Everything else is polish, additional robustness, or strategy.

---

## Part A — Changes already applied in this revision

These were applied directly to `paper.tex` / `references.bib`; verify with
`git diff`.

1. **Removed internal-history framing throughout.** All references to a
   "corrected pipeline", "development runs", and the former Section 6.3
   ("Regularisation and target normalisation: development notes") are gone.
   The paper now reads as one cohesive study. The two useful facts from the
   removed section were relocated into the 3D U-Net methods subsection
   (Section 4.2.3), stated only with evidence from the final task: (a) per-fold
   target standardisation is required because the target mean shifts by
   several dex between folds; (b) L2 weight decay is the sole regulariser, with
   the dropout-variant comparison (0.948 vs 0.974, losses concentrated at the
   extrapolation folds) reported with its architecture confound acknowledged.
2. **Fixed a factual contradiction in Section 3.1.** The text claimed "the
   boundary folds are the hardest for every model family"; Table 1 shows the
   MLP and both ensembles are weakest at the *interior* fold G0=0.8 (MLP+sp
   0.871; stacked 0.965) while the stacked ensemble's *best* fold is the upward
   boundary (0.999). Rewritten to describe the actual family-dependent pattern.
3. **Fixed the related claim in Section 5.1** that the trees' G0=6.4 weakness
   is "mitigated but not removed by ... ensembling" — the stacked ensemble
   reaches 0.999 there. Now stated correctly.
4. **Softened an unsupported superlative.** "f_H2 is individually the most
   informative single feature" (Section 5.2 and Conclusion ii) claimed a
   ranking never established — only f_H2 was ablated. Now: "carries substantial
   information not recoverable from the remaining local features."
5. **Corrected the MLP variability claim** (~0.01 → ~0.02): archived
   repetitions show mlp_wide mean R² ranging 0.945–0.965 across runs.
6. **Corrected the spatial-feature gains** in Section 7.1 (+0.013/+0.015, the
   MLP gain was understated).
7. **Added the missing literature context** (referees would demand both):
   - Analytic/fitted molecular-fraction prescriptions as the classical
     alternative to ML surrogates: Krumholz, McKee & Tumlinson (2009);
     Gnedin & Kravtsov (2011); Sternberg et al. (2014); Bialy & Sternberg
     (2016). Cited in the Introduction, plus a new Discussion passage on why a
     calibrated analytic baseline is natural future work.
   - Chemistry-emulator lineage beyond Branca & Pallottini: de Mijolla et al.
     (2019), Holdship et al. (2021, Chemulator), Grassi et al. (2022). The odd
     citation of Glover papers as "chemistry emulators" was repaired (they are
     now cited for the shielding physics, where they belong).
   - **Action for you:** I verified titles/volumes/pages online for
     Holdship 2021 (A&A 653, A76), de Mijolla 2019 (A&A 630, A117), Grassi 2022
     (A&A 668, A139), Gnedin & Kravtsov 2011 (ApJ 728, 88). The KMT09,
     Sternberg 2014 and Bialy & Sternberg 2016 entries are from memory —
     confirm page numbers via ADS before submission (expected: ApJ 693, 216;
     ApJ 790, 10; ApJ 822, 83).
8. **Trimmed the abstract** to ≈250 words (MNRAS limit; it was ~300).
9. **Renamed `slide6_diagram.png` → `paper/fig_method_diagram.png`** and
   updated the `\includegraphics`. Slide-deck artefact names look bad in
   source hand-offs and submission systems.
10. **Reproducibility promise made accurate.** Section 3.3 claimed every
    result carries a run timestamp in its caption — the tables carried none.
    The claim now says results are traceable to archived logs (true).

---

## Part B — Blocking issues only you can resolve

### C1. Simulation provenance (the single biggest rejection risk)

Section 2.1 currently states that the code, chemical network, box size,
resolution, metallicity, dust model and driving are "beyond the scope of this
methods-focused paper". At MNRAS/A&A/ApJ this will draw an immediate major
revision or rejection: referees cannot judge whether "R² = 0.988" means
anything without knowing what physics produced the data, and "controlled
private suite" reads as "unverifiable".

Minimum to add (a short table + one paragraph):

| Property | Needed |
|---|---|
| Simulation code + citation | name, version, reference |
| Box size / cell size | pc (or code units + conversion) |
| Metallicity, dust-to-gas ratio | values |
| Chemical network | species count, reactions, reference |
| Self-shielding prescription | which formula (presumably Wolcott-Green+11 — confirm) |
| Turbulence driving | driven/decaying, Mach number |
| "Quasi-equilibrium" criterion | how convergence was decided |
| Snapshot time | in dynamical/chemical times |

If the suite belongs to collaborators, the clean solutions are: (a) add the
simulator as co-author and a proper Section 2; or (b) cite the paper that
introduced the suite. If neither is possible, target a methods venue (Part F)
— but even RASTI/MLST referees will want the table above.

### C2. Data and code availability

"Available from the author on request" is below the current bar at leading
journals and reviewers increasingly test it. Strongly recommended before
submission:

- Push the analysis code (it is already clean and self-contained: 
  `data_loader.py`, `compare_architectures.py`, `train_cnn.py`, etc.) to a
  public GitHub repo and mint a Zenodo DOI.
- Archive the experiment logs (the JSON files are small) with it.
- If the cubes cannot be public, say precisely why and who controls them;
  consider publishing one cube as a demonstrator.

### C3. Physical units

Table 1 (features) lists velocities and magnetic-field components in "code"
units. Convert to physical units or give the conversion factors. A referee
will also ask for the box size in pc to interpret kernel widths of 3–7 voxels
physically — right now "a 7-voxel neighbourhood" has no physical scale, which
weakens the paper's central physical interpretation (neighbourhood averages ≈
crude shielding columns).

### C4. Affiliation/acknowledgements

- Acknowledge the simulation providers (currently only "the open-source
  Python ecosystem" — if someone gave you 7 simulation cubes, they are missing).
- Funding statement if applicable; ORCID on submission.

---

## Part C — Additional runs (prioritised)

### D1. Complete Table 2 (f_H2 ablation with spatial features) — **DONE**

`python compare_architectures.py --no-fh2 --log ablation_nofh2_spatial.json`
was run during this review (log: `ablation_nofh2_spatial.json`). Results
(mean log-space R² ± fold std, leave-one-G0-out):

| Model | with f_H2 | without f_H2 | penalty |
|---|---|---|---|
| XGBoost (local) | 0.950 ± 0.029 | 0.911 ± 0.033 | −0.039 |
| Wide MLP (local) | 0.954 ± 0.038 | 0.909 ± 0.057 | −0.045 |
| XGBoost + spatial | 0.963 ± 0.033 | 0.941 ± 0.029 | −0.022 |
| MLP + spatial | 0.968 ± 0.041 | 0.953 ± 0.027 | −0.016 |
| Stacked ensemble | 0.988 ± 0.013 | 0.977 ± 0.017 | −0.011 |

The spatial features recover roughly half of the f_H2 gap — exactly the
hypothesis the old text ventured — and the fully solver-independent stacked
ensemble retains R² = 0.977. Table 2 is now complete (with an added
MLP+spatial column), the "(runs in progress)" footnote is gone, and the
abstract, Section 5.2, Section 7.3 (deployability) and Conclusion (ii) were
updated with these numbers. The run also regenerates the MLP-without-f_H2
number (0.909 ± 0.057, matching the previously unprovenanced value) with a
verifiable log. The 14-feature XGBoost baseline reproduced
`ablation_xgb_nofh2.json` exactly (0.9111 ± 0.0334).

### D2. Re-run the error-analysis prediction volumes with the full schedule

Section 5.4 analyses a run whose inner MLP used 30 epochs (mean 0.968) while
the headline model uses 100 (0.988). Referees dislike "the analysed model is
not the reported model", and the +0.2 to +0.5 dex bias you carefully report
may partly be an artefact of the shortened schedule. Rerun the
prediction-volume script with the full configuration and regenerate
`analysis_output/fig1–fig6`. If the bias persists, the paper's honesty stands
on firmer ground; if it shrinks, even better.

### D3. Bias recalibration (cheap, removes a caveat)

The Discussion says "ensemble recalibration would largely remove" the interior-
fold bias — a claim, not a result. Fit a per-fold affine correction on
*training-side* out-of-fold predictions (no test leakage) and report the
post-correction bias. One paragraph + one number per fold turns a limitation
into a demonstrated fix.

### D4. Feature importance (cheap, supports the f_H2 narrative)

XGBoost gain-based importance or permutation importance on one fold's model,
reported as a small table or bar panel. This substantiates "f_H2 is highly
informative", shows what replaces it in the no-f_H2 model (presumably ext and
density averages — which would beautifully support the shielding-column
interpretation of the spatial features), and referees routinely ask for it.

### D5. Seed repeats for the MLP (moderate)

Claims like "differences between MLP architectures are comparable to
seed-to-seed variation" are currently supported by ~4 archived repetitions.
Five seeds × 7 folds for mlp_wide(+sp) would let you quote a proper seed std.

### D6. Clean dropout ablation (optional)

The dropout claim is now stated with its confound (residual variant, reduced
capacity). A single run of `unet_standard` (base_ch=32) with dropout=0.1 and
otherwise identical settings would unconfound it. Skip if time-limited — the
current phrasing is defensible.

### D7. Equal-resolution U-Net or a quantified excuse (optional)

The U-Net trains at 64³ while tabular models see 128³. Either run one 128³
configuration (even 16 base channels, one or two folds, reporting memory) or
state the memory arithmetic explicitly (a 128³×15-channel float32 volume with
activations at base_ch=32 needs ≈X GB > 24 GB available). Currently "for
memory reasons" is asserted, not shown.

### D8. Trivial baseline row (optional, strengthens Table 1)

A Ridge/linear regression on the same 60 features would anchor the table from
below and show how much of the 0.988 is nonlinearity vs features. Minutes of
compute; one extra row.

---

## Part D — Statistical and presentation notes

1. **n = 7 folds.** The stacked-vs-equal-weight gap (0.988 vs 0.978) is
   supported by consistency across repetitions, which the text already states
   — good. Resist any temptation to add p-values; with 7 folds a sign-test
   style statement ("better in 7/7 folds and in every repetition") is the
   honest form. Consider adding per-fold *minima* to the abstract claim if a
   referee pushes ("mean 0.988, worst fold 0.965").
2. **σ in Table 1** is the population std over 7 folds — fine, but say
   "std over folds" in the caption once.
3. **Figures.**
   - `nH2_histograms.png`: drop the in-figure suptitle (captions do that job);
     x-label should be log10(nH2 / cm⁻³) — units.
   - `fig2_scatter.png`: per-panel annotation text is small at print size;
     enlarge one notch. Consider PDF (vector) export for all line/histogram
     figures; keep PNG only for the 2D-histogram and slice panels.
   - Heatmap figures pulled from `logs/.../run_*_heatmap.png`: fine for now,
     but rename for submission (journals require fig1.pdf-style uploads) and
     check the colour map is colour-blind safe (viridis family: yes).
   - The method diagram (now `fig_method_diagram.png`) is a slide graphic;
     consider redrawing in vector form (TikZ/Inkscape) at publication quality.
4. **References:** add DOIs/ADS bibcodes to all entries (MNRAS style tolerates
   their absence but editors ask); Ioffe/Ulyanov are arXiv-only — fine for ML
   citations in MNRAS, but double-check the final bibliography renders
   "arXiv:1502.03167" correctly with the mnras.bst you use.
5. **Keywords** (5, from the approved MNRAS list) are fine. If you want the ML
   audience: MNRAS added "software: machine learning" to the list — consider
   swapping in.
6. **Terminology:** "dex" is used before being defined; define at first use
   ("0.25 dex, i.e. a factor of 10^0.25≈1.8").

---

## Part E — Journal targeting

Given the content (methods-forward, single physical axis, private data):

| Venue | Fit | Notes |
|---|---|---|
| **MNRAS** | Good, *if* C1 is fixed | Methods papers are welcome; 250-word abstract now met; expects the simulation table and a real data-availability statement. |
| **RASTI** (RAS Techniques & Instruments) | Very good | Purpose-built for methods; referees still want C1's table but the bar on "new astrophysics" is lower. Same submission system/style as MNRAS — zero reformatting cost. |
| **A&A** (Numerical methods and codes section) | Good | Comparable requirements; European simulation community overlap (KROME, Branca & Pallottini) may find it naturally. |
| **Machine Learning: Science & Technology** (IOP) | Good fallback | ML-methodology framing (OOD generalisation, tabular-vs-CNN) is a first-class contribution there; astrophysical provenance pressure is lower but not zero. |

My recommendation: fix C1–C3, complete D1–D4, then submit to **MNRAS**; if the
simulation description cannot be obtained at the needed depth, go **RASTI**
with the same manuscript.

---

## Status of this session's concrete work

- [x] Table 1, Table 2 (existing cells), U-Net rows verified against archived logs
- [x] All Part A text changes applied to `paper.tex` / `references.bib`
- [x] D1 ablation run complete; Table 2 and all dependent prose updated
- [x] PDF rebuilt cleanly after all edits
- [ ] Items C1–C4, D2–D8: yours

