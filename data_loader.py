"""
data_loader.py
Load, preprocess, and expose all 7 simulation cubes for ML training.
"""

import argparse
import hashlib
import os
import glob
import pandas as pd
import numpy as np

COLS = ['ix', 'iy', 'iz', 'nH', 'nH2', 'T', 'vx', 'vy', 'vz',
        'nHp', 'ext', 'fh2', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']

# Features used for all models.
# fh2 (H2 self-shielding factor) is included: it is an independent physical
# quantity — not algebraically derived from nH2 — so it is not data leakage.
FEATURE_COLS = ['log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2', 'log_G0',
                'vx', 'vy', 'vz', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']

TARGET_COL     = 'nH2'
LOG_TARGET_COL = 'log_nH2'


def get_feature_cols(drop: set[str] | None = None) -> list[str]:
    """Return FEATURE_COLS, optionally excluding the named columns."""
    if not drop:
        return FEATURE_COLS
    return [c for c in FEATURE_COLS if c not in drop]


def add_drop_args(parser: argparse.ArgumentParser) -> None:
    """Add all --no-X feature-exclusion flags to an ArgumentParser."""
    parser.add_argument('--no-fh2', action='store_true', help='Exclude log_fh2 from features.')
    parser.add_argument('--no-nH',  action='store_true', help='Exclude log_nH from features.')
    parser.add_argument('--no-T',   action='store_true', help='Exclude log_T from features.')
    parser.add_argument('--no-nHp', action='store_true', help='Exclude log_nHp from features.')
    parser.add_argument('--no-ext', action='store_true', help='Exclude ext from features.')
    parser.add_argument('--no-G0',  action='store_true', help='Exclude log_G0 from features.')
    parser.add_argument('--no-vel', action='store_true', help='Exclude vx/vy/vz from features.')
    parser.add_argument('--no-B',   action='store_true', help='Exclude magnetic field components.')


def build_drop_set(args: argparse.Namespace) -> set[str]:
    """Build the feature drop set from parsed --no-X arguments."""
    drop: set[str] = set()
    if args.no_fh2: drop.add('log_fh2')
    if args.no_nH:  drop.add('log_nH')
    if args.no_T:   drop.add('log_T')
    if args.no_nHp: drop.add('log_nHp')
    if args.no_ext: drop.add('ext')
    if args.no_G0:  drop.add('log_G0')
    if args.no_vel: drop.update(['vx', 'vy', 'vz'])
    if args.no_B:   drop.update(['bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr'])
    return drop

# Small epsilon to guard against log(0)
_EPS = 1e-30


def _parse_g0(dir_name: str) -> float:
    """Parse G0 value from directory name: '0_1' -> 0.1, '3_2' -> 3.2"""
    return float(dir_name.replace('_', '.'))


def load_all_cubes(data_root: str = 'data/UVonly') -> list[pd.DataFrame]:
    """
    Load all simulation cubes from the UVonly directory.

    Returns a list of DataFrames (one per G0 value), sorted by G0.
    Each DataFrame contains:
      - Raw physical columns
      - log_nH, log_T, log_nHp, log_fh2  (log10-transformed quantities)
      - log_nH2                            (log10-transformed target)
      - G0, log_G0                         (UV field strength, linear and log)
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

        # Log-transform skewed physical quantities
        df['log_nH']  = np.log10(df['nH']  + _EPS)
        df['log_T']   = np.log10(df['T']   + _EPS)
        df['log_nHp'] = np.log10(df['nHp'] + _EPS)
        # fh2 is the H2 self-shielding factor (independent of nH2, not leakage)
        df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))

        # Log-transform target (nH2 spans many orders of magnitude)
        df['log_nH2'] = np.log10(df['nH2'].clip(lower=_EPS))

        # UV field strength as feature
        df['G0']     = g0
        df['log_G0'] = np.log10(g0)

        cubes.append(df)
        print(f"  G0={g0:<5.1f} | rows={len(df):,} | "
              f"nH2=[{df['nH2'].min():.2e}, {df['nH2'].max():.2e}] | "
              f"fh2=[{df['fh2'].min():.2e}, {df['fh2'].max():.2e}]")

    # Sort by G0 so fold index == sorted G0 index
    cubes.sort(key=lambda d: d['G0'].iloc[0])
    return cubes


def get_fold_labels(cubes: list[pd.DataFrame]) -> np.ndarray:
    """Return an integer fold label array (one label per row, matching cube index)."""
    return np.concatenate([np.full(len(df), i, dtype=int) for i, df in enumerate(cubes)])


def get_X_y(cubes: list[pd.DataFrame],
            use_log_target: bool = True,
            feature_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stack all cubes into flat (N, F) feature and (N,) target arrays.

    Returns:
        X      : float32 array of shape (N, len(feature_cols))
        y      : float32 array of shape (N,)  — log_nH2 or nH2
        folds  : int array of shape (N,) — cube index for leave-one-out CV
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    target = LOG_TARGET_COL if use_log_target else TARGET_COL
    X      = np.concatenate([df[feature_cols].values for df in cubes], axis=0).astype(np.float32)
    y      = np.concatenate([df[target].values       for df in cubes], axis=0).astype(np.float32)
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


def compute_data_checksum(data_root: str = 'data/UVonly') -> dict:
    """Compute a reproducibility fingerprint of the dataset CSV files.

    Hashes the first 8 KB + file size of every CSV to detect any change in
    the input data without reading the full files.  Include the returned dict
    in every experiment log so readers can verify they used the same data.
    """
    csv_paths = sorted(glob.glob(os.path.join(data_root, '*', '*.csv')))
    h = hashlib.sha256()
    file_info = []
    for path in csv_paths:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(8192)
        h.update(path.encode())
        h.update(size.to_bytes(8, 'little'))
        h.update(head)
        file_info.append({'path': os.path.relpath(path).replace('\\', '/'),
                          'size_bytes': size})
    return {
        'sha256_partial': h.hexdigest(),
        'n_files': len(csv_paths),
        'files': file_info,
    }


def get_g0_values(cubes: list[pd.DataFrame]) -> list[float]:
    """Return sorted list of G0 values, one per cube."""
    return [df['G0'].iloc[0] for df in cubes]


def load_single_cube(g0: float, data_root: str = 'data/UVonly') -> pd.DataFrame:
    """Load a single simulation cube for the given G0 value.

    Applies the same preprocessing as load_all_cubes (log_fh2, log_nH2)
    but skips columns that are only needed for training (log_nH, log_T, etc.).
    """
    dir_name = f"{g0:.1f}".replace('.', '_')
    csv_paths = sorted(glob.glob(os.path.join(data_root, dir_name, '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV found for G0={g0} (looked for: {data_root}/{dir_name}/*.csv)")
    df = pd.read_csv(csv_paths[0], sep=r'\s+', header=None, skiprows=1)
    df.columns = COLS
    df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))
    df['log_nH2'] = np.log10(df['nH2'].clip(lower=_EPS))
    return df


# ── EDA when run directly ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Loading cubes...")
    cubes = load_all_cubes()

    all_nH2 = np.concatenate([df['nH2'].values for df in cubes])
    all_log_nH2 = np.concatenate([df['log_nH2'].values for df in cubes])

    print("\n--- nH2 distribution (raw) ---")
    print(f"  min    : {all_nH2.min():.4e}")
    print(f"  max    : {all_nH2.max():.4e}")
    print(f"  mean   : {all_nH2.mean():.4e}")
    print(f"  median : {np.median(all_nH2):.4e}")

    print("\n--- log10(nH2) distribution ---")
    print(f"  min    : {all_log_nH2.min():.2f}")
    print(f"  max    : {all_log_nH2.max():.2f}")
    print(f"  mean   : {all_log_nH2.mean():.2f}")
    print(f"  std    : {all_log_nH2.std():.2f}")

    X, y, folds = get_X_y(cubes, use_log_target=True)
    print(f"\nFeature matrix : {X.shape}  dtype={X.dtype}")
    print(f"Target (log)   : {y.shape}  range=[{y.min():.2f}, {y.max():.2f}]")
    print(f"Folds          : {np.bincount(folds).tolist()}")
    print(f"Features       : {FEATURE_COLS}")
