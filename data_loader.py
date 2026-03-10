"""
data_loader.py
Load, preprocess, and expose all 7 simulation cubes for ML training.
"""

import os
import glob
import pandas as pd
import numpy as np

COLS = ['ix', 'iy', 'iz', 'nH', 'nH2', 'T', 'vx', 'vy', 'vz',
        'nHp', 'ext', 'fh2', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']

# Features used for all models (nH2 excluded — derived from fh2, causes leakage)
FEATURE_COLS = ['log_nH', 'log_T', 'log_nHp', 'ext', 'log_G0',
                'vx', 'vy', 'vz', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']

TARGET_COL     = 'fh2'
LOG_TARGET_COL = 'log_fh2'

# Small epsilon to guard against log(0)
_EPS = 1e-30


def _parse_g0(dir_name: str) -> float:
    """Parse G0 value from directory name: '0_1' -> 0.1, '3_2' -> 3.2"""
    return float(dir_name.replace('_', '.'))


def load_all_cubes(data_root: str = 'icedrive-dl-182bd/UVonly') -> list[pd.DataFrame]:
    """
    Load all simulation cubes from the UVonly directory.

    Returns a list of DataFrames (one per G0 value), sorted by G0.
    Each DataFrame contains:
      - Raw physical columns (except nH2)
      - log_nH, log_T, log_nHp  (log10-transformed densities / temperature)
      - log_fh2                  (log10-transformed target)
      - G0, log_G0               (UV field strength, linear and log)
    """
    csv_paths = sorted(glob.glob(os.path.join(data_root, '*', '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under '{data_root}'")

    cubes = []
    for csv_path in csv_paths:
        dir_name = os.path.basename(os.path.dirname(csv_path))
        g0 = _parse_g0(dir_name)

        df = pd.read_csv(csv_path, sep=r'\s+', header=None, skiprows=1)
        df.columns = COLS

        # Remove data-leaking column (fh2 = 2*nH2 / (nH + 2*nH2))
        df = df.drop(columns=['nH2'])

        # Log-transform skewed physical quantities
        df['log_nH']  = np.log10(df['nH']  + _EPS)
        df['log_T']   = np.log10(df['T']   + _EPS)
        df['log_nHp'] = np.log10(df['nHp'] + _EPS)

        # Log-transform target (fh2 spans many orders of magnitude in practice)
        df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))

        # UV field strength as feature
        df['G0']     = g0
        df['log_G0'] = np.log10(g0)

        cubes.append(df)
        print(f"  G0={g0:<5.1f} | rows={len(df):,} | "
              f"fh2=[{df['fh2'].min():.2e}, {df['fh2'].max():.2e}]")

    # Sort by G0 so fold index == sorted G0 index
    cubes.sort(key=lambda d: d['G0'].iloc[0])
    return cubes


def get_fold_labels(cubes: list[pd.DataFrame]) -> np.ndarray:
    """Return an integer fold label array (one label per row, matching cube index)."""
    return np.concatenate([np.full(len(df), i, dtype=int) for i, df in enumerate(cubes)])


def get_X_y(cubes: list[pd.DataFrame],
            use_log_target: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack all cubes into flat (N, F) feature and (N,) target arrays.

    Returns:
        X      : float32 array of shape (N, len(FEATURE_COLS))
        y      : float32 array of shape (N,)  — log_fh2 or fh2
        folds  : int array of shape (N,) — cube index for leave-one-out CV
    """
    target = LOG_TARGET_COL if use_log_target else TARGET_COL
    X      = np.concatenate([df[FEATURE_COLS].values for df in cubes], axis=0).astype(np.float32)
    y      = np.concatenate([df[target].values        for df in cubes], axis=0).astype(np.float32)
    folds  = get_fold_labels(cubes)
    return X, y, folds


def cube_to_volumes(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    """
    Reshape DataFrame columns into 128x128x128 numpy volumes.

    Grid indices in CSVs are 1-based (1..128), so we subtract 1.
    Returns a dict mapping column name -> (128, 128, 128) array.
    """
    ix = df['ix'].values.astype(int) - 1
    iy = df['iy'].values.astype(int) - 1
    iz = df['iz'].values.astype(int) - 1
    volumes = {}
    for col in cols:
        vol = np.zeros((128, 128, 128), dtype=np.float32)
        vol[ix, iy, iz] = df[col].values.astype(np.float32)
        volumes[col] = vol
    return volumes


def get_g0_values(cubes: list[pd.DataFrame]) -> list[float]:
    """Return sorted list of G0 values, one per cube."""
    return [df['G0'].iloc[0] for df in cubes]


def load_single_cube(g0: float, data_root: str = 'icedrive-dl-182bd/UVonly') -> pd.DataFrame:
    """Load a single simulation cube for the given G0 value.

    Applies the same preprocessing as load_all_cubes (drop nH2, add log_fh2)
    but skips columns that are only needed for training (log_nH, log_T, etc.).
    """
    dir_name = f"{g0:.1f}".replace('.', '_')
    csv_paths = sorted(glob.glob(os.path.join(data_root, dir_name, '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV found for G0={g0} (looked for: {data_root}/{dir_name}/*.csv)")
    df = pd.read_csv(csv_paths[0], sep=r'\s+', header=None, skiprows=1)
    df.columns = COLS
    df = df.drop(columns=['nH2'])
    df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))
    return df


# ── EDA when run directly ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading cubes...")
    cubes = load_all_cubes()

    all_fh2 = np.concatenate([df['fh2'].values for df in cubes])
    all_log_fh2 = np.concatenate([df['log_fh2'].values for df in cubes])

    print("\n--- fh2 distribution (raw) ---")
    print(f"  min    : {all_fh2.min():.4e}")
    print(f"  max    : {all_fh2.max():.4e}")
    print(f"  mean   : {all_fh2.mean():.4e}")
    print(f"  median : {np.median(all_fh2):.4e}")
    print(f"  < 0.01 : {(all_fh2 < 0.01).mean() * 100:.1f}%")
    print(f"  > 0.99 : {(all_fh2 > 0.99).mean() * 100:.1f}%")

    print("\n--- log10(fh2) distribution ---")
    print(f"  min    : {all_log_fh2.min():.2f}")
    print(f"  max    : {all_log_fh2.max():.2f}")
    print(f"  mean   : {all_log_fh2.mean():.2f}")
    print(f"  std    : {all_log_fh2.std():.2f}")

    X, y, folds = get_X_y(cubes, use_log_target=True)
    print(f"\nFeature matrix : {X.shape}  dtype={X.dtype}")
    print(f"Target (log)   : {y.shape}  range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"Folds          : {np.bincount(folds).tolist()}")
    print(f"Features       : {FEATURE_COLS}")
