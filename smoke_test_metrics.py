"""Smoke test for the extended compute_metrics suite (no simulation data needed).

Run:  python smoke_test_metrics.py
"""

import json
import numpy as np

from classical_models import compute_metrics, print_results, PHASE_SPLIT

rng = np.random.default_rng(0)

# ── Synthetic bimodal target mimicking the nH2 distribution ──────────────────
n = 200_000
mol = rng.random(n) < 0.3
y_true = np.where(mol, rng.normal(0.0, 1.0, n), rng.normal(-10.0, 1.5, n))

# Prediction = truth + known bias + known scatter
TRUE_BIAS, TRUE_SCATTER = 0.3, 0.4
y_pred = y_true + TRUE_BIAS + rng.normal(0.0, TRUE_SCATTER, n)

m = compute_metrics(y_true, y_pred)

def check(name, got, want, tol):
    ok = abs(got - want) < tol
    print(f"  {'OK ' if ok else 'FAIL'} {name:<10} got={got:+.4f}  want~{want:+.4f}")
    assert ok, f"{name}: {got} vs {want}"

print("[1] Recovery of known bias/scatter and identities")
check('bias',    m['bias'],    TRUE_BIAS,    0.01)
check('scatter', m['scatter'], TRUE_SCATTER, 0.01)
check('RMSE^2',  m['RMSE']**2, m['bias']**2 + m['scatter']**2, 1e-9)
check('W1',      m['W1'],      TRUE_BIAS,    0.02)   # pure shift -> W1 = |bias|
# CCC penalises bias: the same prediction without the shift scores higher,
# even though Pearson correlation (and ranking) is identical
m_unbiased = compute_metrics(y_true, y_pred - TRUE_BIAS)
assert 0.0 < m['CCC'] < m_unbiased['CCC'] < 1.0, (m['CCC'], m_unbiased['CCC'])
print(f"  OK  CCC={m['CCC']:.4f} < unbiased CCC={m_unbiased['CCC']:.4f} "
      f"(bias penalised)")

print("\n[2] Threshold accuracies are sane and ordered")
assert 0 < m['frac_01'] < m['frac_03'] < m['frac_05'] < 1
print(f"  OK  f0.1={m['frac_01']:.3f} < f0.3={m['frac_03']:.3f} "
      f"< f0.5={m['frac_05']:.3f}")

print("\n[3] Phase-conditional metrics")
assert abs(m['f_mol'] - np.mean(y_true > PHASE_SPLIT)) < 1e-12
assert abs(m['bias_mol'] - TRUE_BIAS) < 0.02 and abs(m['bias_dif'] - TRUE_BIAS) < 0.02
print(f"  OK  f_mol={m['f_mol']:.3f}  R2_mol={m['R2_mol']:.3f}  "
      f"R2_dif={m['R2_dif']:.3f}")

print("\n[4] NaN guard: fewer than 10 cells in a phase")
y_small = np.full(1000, -10.0) + rng.normal(0, 0.1, 1000)   # all diffuse
m_small = compute_metrics(y_small, y_small + 0.1)
assert np.isnan(m_small['R2_mol']) and np.isnan(m_small['RMSE_mol'])
assert not np.isnan(m_small['R2_dif'])
print("  OK  molecular metrics NaN, diffuse metrics finite")

print("\n[5] fast=True returns only the cheap keys")
m_fast = compute_metrics(y_true, y_pred, fast=True)
assert set(m_fast) == {'R2', 'RMSE', 'MAE', 'bias', 'scatter'}
assert abs(m_fast['R2'] - m['R2']) < 1e-12
print(f"  OK  keys={sorted(m_fast)}")

print("\n[6] Mass ratio reflects the positive bias")
assert m['mass_ratio'] > 1.5   # +0.3 dex -> x2 in mass
print(f"  OK  mass_ratio={m['mass_ratio']:.3f} (+0.3 dex -> ~x2)")

print("\n[7] JSON serialisability (incl. NaN) and float() coercion")
s = json.dumps({k: float(v) for k, v in m_small.items()})
back = json.loads(s)
assert np.isnan(back['R2_mol'])
print(f"  OK  {len(back)} keys round-tripped")

print("\n[8] print_results renders the full table")
print_results("smoke-test model", [m, m], [0.1, 6.4])

print("\n[9] add_skill_scores: 128^3 path and reference self-skill = 0")
from compare_architectures import add_skill_scores
fm_ref  = [compute_metrics(y_true, y_pred) for _ in range(2)]
fm_good = [compute_metrics(y_true, y_true + rng.normal(0, 0.2, n)) for _ in range(2)]
all_results = {'xgb_standard': fm_ref, 'mlp_wide': fm_good}
add_skill_scores(all_results, {}, [], [0.1, 6.4])
assert abs(all_results['xgb_standard'][0]['skill_vs_xgb']) < 1e-12
assert all_results['mlp_wide'][0]['skill_vs_xgb'] > 0.5
print(f"  OK  ref skill={all_results['xgb_standard'][0]['skill_vs_xgb']:.4f}  "
      f"better-model skill={all_results['mlp_wide'][0]['skill_vs_xgb']:.4f}")

print("\n[10] mass_weighted_bias sees density-dependent error that mean bias hides")
from model_helpers import mass_weighted_bias
# Dense (molecular) cells under-predicted by 0.1 dex, diffuse over-predicted
# by the amount that zeroes the unweighted mean residual -> classic
# compression-to-the-mean: cell-mean unbiased, mass budget broken.
DENSE_ERR = -0.1
diffuse_err = -DENSE_ERR * mol.sum() / (~mol).sum()
y_dd = y_true + np.where(mol, DENSE_ERR, diffuse_err)
m_dd = compute_metrics(y_true, y_dd)
mwb  = mass_weighted_bias(y_dd, y_true)
assert abs(m_dd['bias']) < 1e-6, m_dd['bias']            # mean bias ~ 0
assert abs(mwb - DENSE_ERR) < 0.01, mwb                  # mass-wtd bias ~ -0.1
assert m_dd['mass_ratio'] < 0.85                         # ~10^-0.1 = 0.794
m_fix = compute_metrics(y_true, y_dd - mwb)
assert abs(m_fix['mass_ratio'] - 1.0) < 0.02, m_fix['mass_ratio']
print(f"  OK  mean bias={m_dd['bias']:+.4f}  mass-wtd bias={mwb:+.4f}  "
      f"massR {m_dd['mass_ratio']:.3f} -> {m_fix['mass_ratio']:.3f} after subtraction")

print("\nAll smoke tests passed.")
