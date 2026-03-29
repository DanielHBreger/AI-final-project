"""
Generate the 48 symmetry operations of the octahedral group Oh and apply
them to 3D simulation volumes.

Each operation is a 3x3 signed-permutation matrix R mapping
grid axes: R @ [x, y, z]^T = [x', y', z']^T.

Vector fields transform as:
  Polar vectors (v):  v' =  R @ v
  Axial vectors (b):  b' = det(R) * R @ b

UV field caveat: if the UV field illuminates along +z, only the 8
z-preserving operations are physically valid (safe_only=True, default).
Use safe_only=False only if you're assuming an isotropic UV field.
"""

from __future__ import annotations
import numpy as np
from itertools import product


def _build_oh_group() -> list[np.ndarray]:
    """Return all 48 3x3 integer matrices of the octahedral group Oh."""
    ops = []
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        for signs in product([-1, 1], repeat=3):
            R = np.zeros((3, 3), dtype=int)
            for new_ax, (old_ax, s) in enumerate(zip(perm, signs)):
                R[new_ax, old_ax] = s
            ops.append(R)
    unique = []
    for R in ops:
        if not any(np.array_equal(R, U) for U in unique):
            unique.append(R)
    assert len(unique) == 48, f"Expected 48 ops, got {len(unique)}"
    return unique


_OH_GROUP = _build_oh_group()

_Z_PRESERVING = [R for R in _OH_GROUP
                 if R[2, 2] == 1 and R[0, 2] == 0 and R[1, 2] == 0]
assert len(_Z_PRESERVING) == 8, f"Expected 8 z-preserving ops, got {len(_Z_PRESERVING)}"


def get_symmetry_ops(safe_only: bool = True) -> list[np.ndarray]:
    return _Z_PRESERVING if safe_only else _OH_GROUP


def _apply_grid_op(R: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """Permute and flip axes of a volume according to signed-permutation matrix R."""
    perm  = []
    signs = []
    for new_ax in range(3):
        old_ax = int(np.nonzero(R[new_ax])[0][0])
        perm.append(old_ax)
        signs.append(int(R[new_ax, old_ax]))

    out = np.transpose(vol, perm)
    for new_ax, s in enumerate(signs):
        if s == -1:
            out = np.flip(out, axis=new_ax)
    return np.ascontiguousarray(out)


def augment_cube(volumes: dict[str, np.ndarray],
                 R: np.ndarray) -> dict[str, np.ndarray]:
    """
    Apply symmetry operation R to one cube's volume dict.

    Scalars: just reindex. Polar vectors: v' = R @ v.
    Axial (B-field): b' = det(R) * R @ b, with left/right swap on reflected axes.
    """
    det = int(round(np.linalg.det(R)))  # +1 rotation, -1 improper

    perm, signs = [], []
    for new_ax in range(3):
        old_ax = int(np.nonzero(R[new_ax])[0][0])
        perm.append(old_ax)
        signs.append(int(R[new_ax, old_ax]))

    axis_label = ['x', 'y', 'z']
    out = {}

    for col in ('log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2', 'log_nH2', 'log_G0'):
        if col in volumes:
            out[col] = _apply_grid_op(R, volumes[col])

    v_old = {ax: volumes[f'v{ax}'] for ax in axis_label if f'v{ax}' in volumes}
    for new_ax in range(3):
        old_lbl = axis_label[perm[new_ax]]
        new_lbl = axis_label[new_ax]
        if old_lbl in v_old:
            out[f'v{new_lbl}'] = signs[new_ax] * _apply_grid_op(R, v_old[old_lbl])

    b_old = {}
    for ax in axis_label:
        for side in ('l', 'r'):
            key = f'b{ax}{side}'
            if key in volumes:
                b_old[key] = volumes[key]

    for new_ax in range(3):
        old_lbl = axis_label[perm[new_ax]]
        new_lbl = axis_label[new_ax]
        s = signs[new_ax]
        sign_factor = det * s

        bl_key_old = f'b{old_lbl}l'
        br_key_old = f'b{old_lbl}r'
        bl_key_new = f'b{new_lbl}l'
        br_key_new = f'b{new_lbl}r'

        if bl_key_old in b_old and br_key_old in b_old:
            grid_bl = _apply_grid_op(R, b_old[bl_key_old])
            grid_br = _apply_grid_op(R, b_old[br_key_old])
            if s == -1:
                # reflection along this axis flips left/right
                grid_bl, grid_br = grid_br, grid_bl
            out[bl_key_new] = sign_factor * grid_bl
            out[br_key_new] = sign_factor * grid_br

    return out


def augment_all_cubes(cube_volumes: list[dict[str, np.ndarray]],
                      safe_only: bool = True) -> list[dict[str, np.ndarray]]:
    """
    Apply all valid symmetry operations to every cube.
    Returns a flat list of augmented volume dicts (identity included).
    """
    ops = get_symmetry_ops(safe_only)
    augmented = []
    for cube_idx, cube_vol in enumerate(cube_volumes):
        for R in ops:
            aug = augment_cube(cube_vol, R)
            aug['_cube_idx'] = cube_idx
            augmented.append(aug)
    print(f"Augmented: {len(cube_volumes)} cubes x {len(ops)} ops = {len(augmented)} total")
    return augmented


def _sanity_check(vol: dict[str, np.ndarray], aug: dict[str, np.ndarray]) -> None:
    for col in ('log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2', 'log_nH2'):
        if col not in vol:
            continue
        orig_mean = vol[col].mean()
        aug_mean  = aug[col].mean()
        assert abs(orig_mean - aug_mean) < 1e-4, (
            f"Sanity fail: {col} mean changed {orig_mean:.4f} -> {aug_mean:.4f}"
        )
    print("  Sanity check passed: scalar statistics unchanged after augmentation.")


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all_cubes, cube_to_volumes, FEATURE_COLS, LOG_TARGET_COL

    print("Loading cubes...")
    cubes = load_all_cubes()

    vol_cols = [c for c in FEATURE_COLS if c not in ('log_G0',)] + [LOG_TARGET_COL, 'log_G0']
    avail = [c for c in vol_cols if c in cubes[0].columns]

    print("Converting first cube to volumes...")
    vol = cube_to_volumes(cubes[0], avail)

    ops = get_symmetry_ops(safe_only=True)
    print(f"Testing {len(ops)} safe (z-preserving) operations on first cube...")
    for R in ops:
        aug = augment_cube(vol, R)
        _sanity_check(vol, aug)

    print(f"\nAll {len(ops)} operations passed.")
    print(f"Oh group: 48 total | z-preserving: {len(_Z_PRESERVING)}")
