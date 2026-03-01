"""
augmentation.py
Generate the 48 symmetry operations of a cube (octahedral group Oh) and apply
them to 3D simulation volumes.

Each operation is represented as a 3x3 signed-permutation matrix R that maps
grid axes: R @ [x, y, z]^T = [x', y', z']^T.

Vector fields transform as:
  Polar vectors  (v):  v' =  R @ v
  Axial vectors  (b):  b' = det(R) * R @ b   (det = +1 for rotations, -1 for reflections)

For the face-centred B-field stored as (bxl, bxr, byl, byr, bzl, bzr):
  - The component axis follows the same rotation as axial vectors.
  - The left/right designation swaps when det(R) = -1 along the relevant axis
    (a reflection reverses which face is "left" and which is "right").

⚠️  UV FIELD CAVEAT
If the UV field illuminates the cube from a fixed direction (e.g. along +z),
only operations that keep that axis invariant are physically valid.
Set SAFE_ONLY = True (default) to restrict to the 8-element subgroup that
preserves the z-axis:  4 rotations (0°/90°/180°/270° around z) + their
reflections across the xy-plane.
Run with SAFE_ONLY = False only if the UV field is assumed isotropic.
"""

from __future__ import annotations
import numpy as np
from itertools import product

# ── Build the 48 signed-permutation matrices ─────────────────────────────────

def _build_oh_group() -> list[np.ndarray]:
    """Return all 48 3x3 integer matrices forming the octahedral group Oh."""
    ops = []
    axes = np.eye(3, dtype=int)
    for perm in [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]:
        for signs in product([-1, 1], repeat=3):
            R = np.zeros((3, 3), dtype=int)
            for new_ax, (old_ax, s) in enumerate(zip(perm, signs)):
                R[new_ax, old_ax] = s
            ops.append(R)
    # Deduplicate (signed-permutation matrices are unique by construction, but verify)
    unique = []
    for R in ops:
        if not any(np.array_equal(R, U) for U in unique):
            unique.append(R)
    assert len(unique) == 48, f"Expected 48 ops, got {len(unique)}"
    return unique


_OH_GROUP = _build_oh_group()

# The 8 z-preserving operations (safe when UV illuminates along z)
_Z_PRESERVING = [R for R in _OH_GROUP
                 if R[2, 2] == 1 and R[0, 2] == 0 and R[1, 2] == 0]
assert len(_Z_PRESERVING) == 8, f"Expected 8 z-preserving ops, got {len(_Z_PRESERVING)}"


def get_symmetry_ops(safe_only: bool = True) -> list[np.ndarray]:
    """
    Return list of 3x3 integer rotation/reflection matrices.

    safe_only=True  → 8 operations preserving the z-axis (UV field direction)
    safe_only=False → all 48 operations of Oh
    """
    return _Z_PRESERVING if safe_only else _OH_GROUP


# ── Apply one operation to a volume dict ─────────────────────────────────────

def _apply_grid_op(R: np.ndarray, vol: np.ndarray) -> np.ndarray:
    """
    Permute and flip axes of a (128, 128, 128) volume according to R.
    R is a signed permutation: new axis i comes from old axis where R[i,j] != 0.
    """
    # Find which old axis maps to each new axis, and its sign
    perm  = []   # perm[i] = old axis that becomes new axis i
    signs = []   # signs[i] = +1 or -1
    for new_ax in range(3):
        old_ax = int(np.nonzero(R[new_ax])[0][0])
        perm.append(old_ax)
        signs.append(int(R[new_ax, old_ax]))

    # Permute axes
    out = np.transpose(vol, perm)

    # Flip axes with sign = -1 (reverses that dimension)
    for new_ax, s in enumerate(signs):
        if s == -1:
            out = np.flip(out, axis=new_ax)

    return np.ascontiguousarray(out)


def augment_cube(volumes: dict[str, np.ndarray],
                 R: np.ndarray) -> dict[str, np.ndarray]:
    """
    Apply a single symmetry operation R to a cube's volume dict.

    Expected keys in `volumes`:
      Scalars : 'log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2', 'log_G0'
      Polar v : 'vx', 'vy', 'vz'
      Axial B : 'bxl', 'bxr', 'byl', 'byr', 'bzl', 'bzr'

    Returns a new volume dict with the same keys.
    """
    det = int(round(np.linalg.det(R)))  # +1 (rotation) or -1 (improper rotation)

    # --- Find axis mapping ---
    # perm[i] = old axis that goes to new axis i; sign[i] = ±1
    perm, signs = [], []
    for new_ax in range(3):
        old_ax = int(np.nonzero(R[new_ax])[0][0])
        perm.append(old_ax)
        signs.append(int(R[new_ax, old_ax]))

    axis_label = ['x', 'y', 'z']

    out = {}

    # Scalars — just reindex the grid
    for col in ('log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2', 'log_G0'):
        if col in volumes:
            out[col] = _apply_grid_op(R, volumes[col])

    # Polar vectors: v'_i = R[i,j] * v_j, applied component-by-component
    # v'_new = sign[new] * v_old[perm[new]]
    v_old = {ax: volumes[f'v{ax}'] for ax in axis_label if f'v{ax}' in volumes}
    for new_ax in range(3):
        old_ax  = perm[new_ax]
        old_lbl = axis_label[old_ax]
        new_lbl = axis_label[new_ax]
        if old_lbl in v_old:
            rotated_grid = _apply_grid_op(R, v_old[old_lbl])
            out[f'v{new_lbl}'] = signs[new_ax] * rotated_grid

    # Axial vectors (B-field):  b'_i = det * sign[new] * b_old[perm[new]]
    # Additionally, reflections swap left/right on the affected face.
    #   bxl/bxr are the left/right faces perpendicular to x.
    #   Under a reflection that negates x (sign=-1), left↔right swap.
    b_old = {}
    for ax in axis_label:
        for side in ('l', 'r'):
            key = f'b{ax}{side}'
            if key in volumes:
                b_old[key] = volumes[key]

    for new_ax in range(3):
        old_ax  = perm[new_ax]
        old_lbl = axis_label[old_ax]
        new_lbl = axis_label[new_ax]
        s = signs[new_ax]
        sign_factor = det * s   # axial vector factor

        bl_key_old = f'b{old_lbl}l'
        br_key_old = f'b{old_lbl}r'
        bl_key_new = f'b{new_lbl}l'
        br_key_new = f'b{new_lbl}r'

        if bl_key_old in b_old and br_key_old in b_old:
            grid_bl = _apply_grid_op(R, b_old[bl_key_old])
            grid_br = _apply_grid_op(R, b_old[br_key_old])

            if s == -1:
                # Reflection along this axis: left ↔ right swap
                grid_bl, grid_br = grid_br, grid_bl

            out[bl_key_new] = sign_factor * grid_bl
            out[br_key_new] = sign_factor * grid_br

    return out


# ── Generate full augmented dataset ──────────────────────────────────────────

def augment_all_cubes(cube_volumes: list[dict[str, np.ndarray]],
                      safe_only: bool = True) -> list[dict[str, np.ndarray]]:
    """
    Apply all valid symmetry operations to every cube.

    Returns a flat list of augmented volume dicts (including the identity).
    Length = n_cubes * n_ops  (7 * 8 = 56 for safe_only, 7 * 48 = 336 otherwise).
    """
    ops = get_symmetry_ops(safe_only)
    augmented = []
    for cube_idx, cube_vol in enumerate(cube_volumes):
        for op_idx, R in enumerate(ops):
            aug = augment_cube(cube_vol, R)
            aug['_cube_idx'] = cube_idx   # track origin for CV splitting
            augmented.append(aug)
    print(f"Augmented: {len(cube_volumes)} cubes × {len(ops)} ops "
          f"= {len(augmented)} total volumes")
    return augmented


# ── Sanity check ─────────────────────────────────────────────────────────────

def _sanity_check(vol: dict[str, np.ndarray], aug: dict[str, np.ndarray]) -> None:
    """Scalars should have identical statistics before and after augmentation."""
    for col in ('log_nH', 'log_T', 'log_nHp', 'ext', 'log_fh2'):
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
    # Only include cols that exist in the dataframe
    avail = [c for c in vol_cols if c in cubes[0].columns]

    print("Converting first cube to volumes...")
    vol = cube_to_volumes(cubes[0], avail)

    ops = get_symmetry_ops(safe_only=True)
    print(f"Testing {len(ops)} safe (z-preserving) operations on first cube...")
    for i, R in enumerate(ops):
        aug = augment_cube(vol, R)
        _sanity_check(vol, aug)

    print(f"\nAll {len(ops)} operations passed.")
    print(f"\nOh group: 48 total | z-preserving: {len(_Z_PRESERVING)}")
