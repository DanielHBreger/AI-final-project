import argparse
import os
import glob
import pandas as pd
import numpy as np

COLS = ['ix', 'iy', 'iz', 'nH', 'nH2', 'T', 'vx', 'vy', 'vz',
        'nHp', 'ext', 'fh2', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']

# fh2 (H2 self-shielding factor) is NOT leakage — it's physically independent of nH2
FEATURE_COLS = ['log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2', 'log_G0',
                'vx', 'vy', 'vz', 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr']

TARGET_COL = 'nH2'
LOG_TARGET_COL = 'log_nH2'

_EPS = 1e-30


def get_feature_cols(drop: set[str] | None = None) -> list[str]:
    if not drop:
        return FEATURE_COLS
    return [c for c in FEATURE_COLS if c not in drop]


def add_drop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--no-fh2', action='store_true', help='Exclude log_fh2 from features.')
    parser.add_argument('--no-nH',  action='store_true', help='Exclude log_nH from features.')
    parser.add_argument('--no-T',   action='store_true', help='Exclude log_T from features.')
    parser.add_argument('--no-nHp', action='store_true', help='Exclude log_nHp from features.')
    parser.add_argument('--no-ext', action='store_true', help='Exclude ext from features.')
    parser.add_argument('--no-G0',  action='store_true', help='Exclude log_G0 from features.')
    parser.add_argument('--no-vel', action='store_true', help='Exclude vx/vy/vz from features.')
    parser.add_argument('--no-B',   action='store_true', help='Exclude magnetic field components.')


def build_drop_set(args: argparse.Namespace) -> set[str]:
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


def _parse_g0(dir_name: str) -> float:
    # '0_1' -> 0.1
    return float(dir_name.replace('_', '.'))


def load_all_cubes(data_root: str = 'icedrive-dl-182bd/UVonly') -> list[pd.DataFrame]:
    csv_paths = sorted(glob.glob(os.path.join(data_root, '*', '*.csv')))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under '{data_root}'")

    cubes = []
    for csv_path in csv_paths:
        dir_name = os.path.basename(os.path.dirname(csv_path))
        g0 = _parse_g0(dir_name)

        df = pd.read_csv(csv_path, sep=r'\s+', header=None, skiprows=1)
        df.columns = COLS

        df['log_nH']  = np.log10(df['nH']  + _EPS)
        df['log_T']   = np.log10(df['T']   + _EPS)
        df['log_nHp'] = np.log10(df['nHp'] + _EPS)
        df['log_fh2'] = np.log10(df['fh2'].clip(lower=_EPS))
        df['log_nH2'] = np.log10(df['nH2'].clip(lower=_EPS))
        df['G0']      = g0
        df['log_G0']  = np.log10(g0)

        cubes.append(df)
        print(f"  G0={g0:<5.1f} | rows={len(df):,} | "
              f"nH2=[{df['nH2'].min():.2e}, {df['nH2'].max():.2e}] | "
              f"fh2=[{df['fh2'].min():.2e}, {df['fh2'].max():.2e}]")

    cubes.sort(key=lambda d: d['G0'].iloc[0])
    return cubes


def get_fold_labels(cubes: list[pd.DataFrame]) -> np.ndarray:
    return np.concatenate([np.full(len(df), i, dtype=int) for i, df in enumerate(cubes)])


def get_X_y(cubes: list[pd.DataFrame],
            use_log_target: bool = True,
            feature_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    target = LOG_TARGET_COL if use_log_target else TARGET_COL
    X = np.concatenate([df[feature_cols].values for df in cubes], axis=0).astype(np.float32)
    y = np.concatenate([df[target].values for df in cubes], axis=0).astype(np.float32)
    folds = get_fold_labels(cubes)
    return X, y, folds


def cube_to_volumes(df: pd.DataFrame, cols: list[str]) -> dict[str, np.ndarray]:
    # grid indices are 1-based in the CSVs
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
    return [df['G0'].iloc[0] for df in cubes]


def load_single_cube(g0: float, data_root: str = 'icedrive-dl-182bd/UVonly') -> pd.DataFrame:
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
