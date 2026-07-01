"""
merit_metrics.py
Astrophysically weighted evaluation of the saved stacked-ensemble prediction
volumes: total-H2-mass errors, phase-conditional accuracy, mass-weighted MAE.
Read-only analysis for the scientific-merit review.
"""

import glob
import numpy as np
from data_loader import load_single_cube, cube_to_volumes

PHASE_SPLIT = -4.0   # log10(nH2) threshold separating UV-exposed / molecular

print(f"{'G0':>4} {'massRatio':>9} {'R2_mol':>7} {'RMSE_mol':>8} {'bias_mol':>8} "
      f"{'R2_diff':>8} {'RMSE_diff':>9} {'MAE_mw':>7} {'f_mol%':>6}")

for path in sorted(glob.glob('predictions/pred_g0_*_20260312_*.npz')):
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

    # total H2 mass ratio (uniform grid -> mass proportional to sum of nH2)
    mass_ratio = 10.0 ** pred @ np.ones_like(pred) / np.sum(10.0 ** true)
    mass_ratio = np.sum(10.0 ** pred) / np.sum(10.0 ** true)

    def r2(t, p):
        ss = np.sum((t - np.mean(t)) ** 2)
        return 1.0 - np.sum((t - p) ** 2) / ss

    mol = true > PHASE_SPLIT
    dif = ~mol
    r2_mol = r2(true[mol], pred[mol]) if mol.sum() > 10 else np.nan
    rmse_mol = np.sqrt(np.mean((pred[mol] - true[mol]) ** 2))
    bias_mol = np.mean(pred[mol] - true[mol])
    r2_dif = r2(true[dif], pred[dif])
    rmse_dif = np.sqrt(np.mean((pred[dif] - true[dif]) ** 2))

    # H2-mass-weighted MAE (dex), weights = true nH2
    w = 10.0 ** true
    mae_mw = np.sum(w * np.abs(pred - true)) / np.sum(w)

    print(f"{g0:>4.1f} {mass_ratio:>9.3f} {r2_mol:>7.3f} {rmse_mol:>8.3f} "
          f"{bias_mol:>+8.3f} {r2_dif:>8.3f} {rmse_dif:>9.3f} {mae_mw:>7.3f} "
          f"{100 * mol.mean():>6.2f}")
