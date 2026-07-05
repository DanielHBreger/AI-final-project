"""
phase_threshold_sensitivity.py
Sensitivity of the phase-conditional metrics to the diffuse/molecular split
threshold (RUN_PLAN analysis item A1): recompute R2/RMSE/bias per phase for
log10(nH2) thresholds in [-5, -3] on the saved deployed prediction volumes.
Read-only analysis; writes results/a1_phase_threshold_<ts>.json.
"""

import datetime
import glob
import json
import re

import numpy as np

from data_loader import load_single_cube

THRESHOLDS = [-5.0, -4.5, -4.0, -3.5, -3.0]


def latest_per_g0(pattern: str = 'predictions/pred_g0_*.npz') -> list[str]:
    """Most recent stacked-ensemble prediction file per G0 (lexical max =
    latest timestamp); CNN-format files without 'pred_vol' are skipped.
    Same selector as merit_metrics.py (not imported — that script runs its
    analysis at module level)."""
    by_g0: dict[str, str] = {}
    for path in sorted(glob.glob(pattern)):
        with np.load(path) as d:
            if 'pred_vol' not in d.files:
                continue
        m = re.search(r'pred_g0_([\d.]+)_', path)
        if m:
            by_g0[m.group(1)] = path
    return [by_g0[k] for k in sorted(by_g0, key=float)]


def _phase(t: np.ndarray, resid: np.ndarray, mask: np.ndarray):
    if mask.sum() < 10:
        return float('nan'), float('nan'), float('nan')
    tm, rm = t[mask], resid[mask]
    ss = float(np.sum((tm - tm.mean()) ** 2))
    r2 = 1.0 - float(np.sum(rm ** 2)) / ss if ss > 0 else float('nan')
    return r2, float(np.sqrt(np.mean(rm ** 2))), float(np.mean(rm))


def main() -> None:
    rows: list[dict] = []
    for path in latest_per_g0():
        d = np.load(path)
        g0 = float(d['g0'])
        pred = d['pred_vol'].astype(np.float64).ravel()

        df = load_single_cube(g0)
        ix = df['ix'].values.astype(int) - 1
        iy = df['iy'].values.astype(int) - 1
        iz = df['iz'].values.astype(int) - 1
        true_vol = np.zeros((128, 128, 128), dtype=np.float64)
        true_vol[ix, iy, iz] = df['log_nH2'].values
        true = true_vol.ravel()
        resid = pred - true

        for thr in THRESHOLDS:
            mol = true > thr
            r2m, rmsem, bm = _phase(true, resid, mol)
            r2d, rmsed, bd = _phase(true, resid, ~mol)
            rows.append({'g0': g0, 'threshold': thr,
                         'R2_mol': r2m, 'RMSE_mol': rmsem, 'bias_mol': bm,
                         'R2_dif': r2d, 'RMSE_dif': rmsed, 'bias_dif': bd,
                         'f_mol': float(mol.mean())})
        print(f'G0={g0:.1f} done ({path})')

    print(f"\n{'thr':>5} {'R2_mol':>7} {'R2_mol_min':>10} {'RMSE_mol':>8} "
          f"{'bias_mol':>8} {'R2_dif':>7} {'R2_dif_min':>10} {'f_mol%':>7}")
    for thr in THRESHOLDS:
        rs = [r for r in rows if r['threshold'] == thr]
        print(f"{thr:>5.1f} {np.nanmean([r['R2_mol'] for r in rs]):>7.4f} "
              f"{np.nanmin([r['R2_mol'] for r in rs]):>10.4f} "
              f"{np.nanmean([r['RMSE_mol'] for r in rs]):>8.3f} "
              f"{np.nanmean([r['bias_mol'] for r in rs]):>+8.3f} "
              f"{np.nanmean([r['R2_dif'] for r in rs]):>7.4f} "
              f"{np.nanmin([r['R2_dif'] for r in rs]):>10.4f} "
              f"{100 * np.mean([r['f_mol'] for r in rs]):>7.2f}")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'results/a1_phase_threshold_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'thresholds': THRESHOLDS,
                   'per_fold': rows}, f, indent=2)
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
