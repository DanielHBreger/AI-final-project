"""
calibration_functionals.py
Mass-weighted-residual calibration vs exact log-mass-ratio fitting
(RUN_PLAN analysis item A4), from the saved run-5 prediction volumes.

Section 4.5 of the paper states that the mass-calibration functional
  b_M = sum_i 10^{y_i} (yhat_i - y_i) / sum_i 10^{y_i}
and the exact per-cube mass-closure offset
  delta = log10( sum_i 10^{yhat_i} / sum_i 10^{y_i} )
coincide to first order in the residuals.  This script quantifies that
claim on the deployed volumes: both functionals are evaluated on the RAW
(uncalibrated) residuals of each held-out cube.  Subtracting a constant
c shifts the log mass ratio by -c, so (delta - b_M) is exactly the log
mass-closure error the b_M functional would leave if it were fitted on
the cube it is applied to; 10^(delta - b_M) is the corresponding mass
ratio.  The offset the pipeline actually applied (fitted on the six
training cubes) is reported alongside for context.

Read-only analysis; writes results/a4_calibration_functionals_<ts>.json.
"""

import datetime
import json

import numpy as np

from bootstrap_cis import latest_per_g0
from data_loader import load_single_cube


def main() -> None:
    results: list[dict] = []

    for path in latest_per_g0():
        d = np.load(path)
        g0 = float(d['g0'])
        p_raw = d['pred_vol_raw'].astype(np.float64).ravel()
        applied = float(d['bias_offset'])

        df = load_single_cube(g0)
        ix = df['ix'].values.astype(int) - 1
        iy = df['iy'].values.astype(int) - 1
        iz = df['iz'].values.astype(int) - 1
        true_vol = np.zeros((128, 128, 128), dtype=np.float64)
        true_vol[ix, iy, iz] = df['log_nH2'].values
        t = true_vol.ravel()

        r = p_raw - t
        t_lin = 10.0 ** t
        b_m = float(np.sum(t_lin * r) / np.sum(t_lin))
        delta = float(np.log10(np.sum(10.0 ** p_raw) / np.sum(t_lin)))

        results.append({
            'g0': g0,
            'pred_file': path,
            'b_M': b_m,                                # mass-weighted mean residual (dex)
            'delta_exact': delta,                      # exact log mass ratio (dex)
            'gap_dex': b_m - delta,                    # first-order-equivalence gap
            'mass_ratio_if_bM_insample': 10.0 ** (delta - b_m),
            'applied_offset': applied,                 # fitted on the 6 training cubes
            'mass_ratio_achieved': float(d['mass_ratio']),
        })
        print(f'G0={g0:.1f} done')

    print(f"\n{'G0':>4} {'b_M':>8} {'delta':>8} {'gap':>8} "
          f"{'mass(b_M)':>10} {'applied':>8} {'achieved':>9}")
    for r_ in results:
        print(f"{r_['g0']:>4.1f} {r_['b_M']:>8.4f} {r_['delta_exact']:>8.4f} "
              f"{r_['gap_dex']:>8.4f} {r_['mass_ratio_if_bM_insample']:>10.4f} "
              f"{r_['applied_offset']:>8.4f} {r_['mass_ratio_achieved']:>9.4f}")

    gaps = [abs(r_['gap_dex']) for r_ in results]
    print(f"\n|gap|: max {max(gaps):.4f} dex, mean {np.mean(gaps):.4f} dex "
          f"(in-sample b_M mass closure within "
          f"{max(abs(1 - r_['mass_ratio_if_bM_insample']) for r_ in results) * 100:.1f}%)")

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = f'results/a4_calibration_functionals_{ts}.json'
    with open(out, 'w') as f:
        json.dump({'timestamp': ts, 'folds': results}, f, indent=2)
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
