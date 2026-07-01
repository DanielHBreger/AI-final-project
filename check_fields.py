from data_loader import load_single_cube
import numpy as np

prev = None
for g0 in [0.1, 0.8, 6.4]:
    df = load_single_cube(g0)
    print(f"G0={g0}: T mean={df['T'].mean():.1f} med={np.median(df['T']):.1f}  "
          f"nH med={np.median(df['nH']):.3f}  nHp med={np.median(df['nHp']):.2e}")
    cur = df[['nH', 'T', 'vx']].values
    if prev is not None:
        same_nH = np.allclose(prev[:, 0], cur[:, 0])
        same_T = np.allclose(prev[:, 1], cur[:, 1])
        same_vx = np.allclose(prev[:, 2], cur[:, 2])
        print(f"   identical to previous cube? nH={same_nH}  T={same_T}  vx={same_vx}")
    prev = cur
