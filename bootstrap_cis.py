"""
bootstrap_cis.py
Within-fold bootstrap confidence intervals for the deployed pipeline's
headline metrics (RUN_PLAN analysis item A2), from the saved prediction
volumes. Two resampling units per fold:

  cell  — resample the 2,097,152 cells i.i.d.; the literal cell-level CI,
          optimistic because neighbouring cells are spatially correlated;
  block — resample 512 non-overlapping 16^3 blocks; coarser but closer to
          honest under spatial correlation.

Read-only analysis; writes results/a2_bootstrap_cis_<ts>.json.
"""

import datetime
import glob
import json
import re

import numpy as np

from data_loader import load_single_cube

N_BOOT = 200
SEED = 67
BLOCK = 16   # 128/16 = 8 -> 512 blocks


def latest_per_g0(pattern: str = 'predictions/pred_g0_*.npz') -> list[str]:
    """Same selector as merit_metrics.py (not imported — that script runs
    its analysis at module level)."""
    by_g0: dict[str, str] = {}
    for path in sorted(glob.glob(pattern)):
        with np.load(path) as d:
            if 'pred_vol' not in d.files:
                continue
        m = re.search(r'pred_g0_([\d.]+)_', path)
        if m:
            by_g0[m.group(1)] = path
    return [by_g0[k] for k in sorted(by_g0, key=float)]


def _metrics(t, r, t_lin, p_lin):
    """Headline metrics on one (resampled) set of cells.
    mass_ratio is unclipped — the clip is never active for the deployed
    volumes (verified in the paper, Section 5.5)."""
    ss = np.sum((t - t.mean()) ** 2)
    mol = t > -4.0
    ss_m = np.sum((t[mol] - t[mol].mean()) ** 2)
    return {
        'R2':         1.0 - np.sum(r ** 2) / ss,
        'RMSE':       np.sqrt(np.mean(r ** 2)),
        'bias':       np.mean(r),
        'mass_ratio': np.sum(p_lin) / np.sum(t_lin),
        'R2_mol':     1.0 - np.sum(r[mol] ** 2) / ss_m,
        'frac_01':    np.mean(np.abs(r) < 0.1),
    }


def _ci(samples: list[dict]) -> dict:
    keys = samples[0].keys()
    return {k: [float(np.percentile([s[k] for s in samples], 2.5)),
                float(np.percentile([s[k] for s in samples], 97.5))]
            for k in keys}


def main() -> None:
    rng = np.random.default_rng(SEED)
    results: list[dict] = []

    for path in latest_per_g0():
        d = np.load(path)
        g0 = float(d['g0'])
        pred_vol = d['pred_vol'].astype(np.float64)

        df = load_single_cube(g0)
        ix = df['ix'].values.astype(int) - 1
        iy = df['iy'].values.astype(int) - 1
        iz = df['iz'].values.astype(int) - 1
        true_vol = np.zeros((128, 128, 128), dtype=np.float64)
        true_vol[ix, iy, iz] = df['log_nH2'].values

        t = true_vol.ravel()
        p = pred_vol.ravel()
        r = p - t
        t_lin = 10.0 ** t
        p_lin = 10.0 ** p
        n = t.size
        point = _metrics(t, r, t_lin, p_lin)

        # cell-level bootstrap
        cell_samples = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, n, n)
            cell_samples.append(_metrics(t[idx], r[idx], t_lin[idx], p_lin[idx]))

        # block bootstrap: (8,8,8) grid of 16^3 blocks -> rows of a 2-D view
        nb = 128 // BLOCK
        def blocks(vol_flat):
            return (vol_flat.reshape(nb, BLOCK, nb, BLOCK, nb, BLOCK)
                    .transpose(0, 2, 4, 1, 3, 5).reshape(nb ** 3, BLOCK ** 3))
        tb, rb = blocks(t), blocks(r)
        tlb, plb = blocks(t_lin), blocks(p_lin)
        block_samples = []
        for _ in range(N_BOOT):
            bidx = rng.integers(0, nb ** 3, nb ** 3)
            block_samples.append(_metrics(tb[bidx].ravel(), rb[bidx].ravel(),
                                          tlb[bidx].ravel(), plb[bidx].ravel()))

        results.append({'g0': g0, 'pred_file': path,
                        'point': {k: float(v) for k, v in point.items()},
                        'ci_cell': _ci(cell_samples),
                        'ci_block': _ci(block_samples)})
        print(f'G0={g0:.1f} done')

    print(f"\n{'G0':>4} {'metric':>10} {'point':>8}   {'cell 95% CI':>17}   "
          f"{'block 95% CI':>17}")
    for res in results:
        for k in ('R2', 'RMSE', 'mass_ratio'):
            c, b = res['ci_cell'][k], res['ci_block'][k]
            print(f"{res['g0']:>4.1f} {k:>10} {res['point'][k]:>8.4f}   "
                  f"[{c[0]:>7.4f},{c[1]:>7.4f}]   [{b[0]:>7.4f},{b[1]:>7.4f}]")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'results/a2_bootstrap_cis_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'n_boot': N_BOOT, 'seed': SEED,
                   'block': BLOCK, 'folds': results}, f, indent=2)
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
