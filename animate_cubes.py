"""
animate_cubes.py
3x2 grid (3 cols, 2 rows) of 3-D scatter plots of ground-truth nH2 in linear
space, rotating 360 deg. Frames are rendered in parallel; panel data lives in
shared memory so workers never copy it (no RAM multiplication).
Saves cube_rotation.gif  (and optionally cube_rotation.mp4 if ffmpeg present).
"""

import io
import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

from data_loader import load_single_cube

# ── config ────────────────────────────────────────────────────────────────────
G0_VALUES = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
ELEV      = 22
FRAMES    = 72          # 360 / 5 deg per frame
FPS       = 20
CMAP      = 'plasma'
OUT_GIF   = 'cube_rotation.gif'
N_WORKERS = 24           # raise if you have RAM headroom; lower to use less RAM

# ── worker globals ─────────────────────────────────────────────────────────────
_panels   = None
_norm     = None
_cfg      = None
_shm_refs = []   # keep SharedMemory handles alive for the life of each worker


def _init_worker(shm_meta, vmin, vmax, cfg):
    """Run once per worker: attach to shared memory, build panel views."""
    global _panels, _norm, _cfg, _shm_refs
    _norm     = LogNorm(vmin=vmin, vmax=vmax)
    _cfg      = cfg
    _panels   = []
    _shm_refs = []
    for pmeta in shm_meta:
        panel = {'g0': pmeta['g0']}
        for key, (name, shape, dtype_str) in pmeta['arrays'].items():
            shm = SharedMemory(name=name)
            _shm_refs.append(shm)                               # keep alive
            panel[key] = np.ndarray(shape, dtype=np.dtype(dtype_str),
                                    buffer=shm.buf)
        _panels.append(panel)


def _render_frame(frame_idx):
    azim = frame_idx * (360.0 / _cfg['frames'])
    fig  = plt.figure(figsize=(16, 10), facecolor='#0d0d0d')
    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.03, top=0.93,
                        wspace=0.05, hspace=0.08)
    for i, p in enumerate(_panels):
        ax = fig.add_subplot(2, 3, i + 1, projection='3d', facecolor='#0d0d0d')
        ax.scatter(p['x'], p['y'], p['z'],
                   c=p['c'], cmap=_cfg['cmap'], s=0.3, alpha=0.5,
                   norm=_norm, depthshade=True)
        ax.set_title(f'G$_0$ = {p["g0"]}', color='white', fontsize=12, pad=2)
        ax.set_xlabel('x', color='#aaaaaa', fontsize=7, labelpad=1)
        ax.set_ylabel('y', color='#aaaaaa', fontsize=7, labelpad=1)
        ax.set_zlabel('z', color='#aaaaaa', fontsize=7, labelpad=1)
        ax.tick_params(colors='#666666', labelsize=5, pad=0)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#333333')
        ax.grid(False)
        ax.view_init(elev=_cfg['elev'], azim=azim)
    cax = fig.add_axes([0.90, 0.10, 0.02, 0.78])
    sm  = plt.cm.ScalarMappable(cmap=_cfg['cmap'], norm=_norm)
    sm.set_array([])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label(r'$n_{\mathrm{H_2}}$  [cm$^{-3}$]', color='white', fontsize=11)
    cb.ax.yaxis.set_tick_params(color='white', labelcolor='white')
    fig.suptitle(r'Ground-truth $n_{\mathrm{H_2}}$ - six UV field strengths',
                 color='white', fontsize=13, y=0.97)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor='#0d0d0d')
    plt.close(fig)
    buf.seek(0)
    return frame_idx, buf.read()


def main():
    # ── load data ─────────────────────────────────────────────────────────────
    print("Loading cubes...")
    panels = []
    for g0 in G0_VALUES:
        df = load_single_cube(g0)
        panels.append({
            'g0': g0,
            'x' : df['ix'].values.astype(np.float32),
            'y' : df['iy'].values.astype(np.float32),
            'z' : df['iz'].values.astype(np.float32),
            'c' : df['nH2'].values.astype(np.float32),
        })
        print(f"  G0={g0}  n={len(df):,} points")

    all_c = np.concatenate([p['c'] for p in panels])
    p90   = float(np.percentile(all_c, 90))
    vmin  = max(p90, 1e-10)
    vmax  = float(np.percentile(all_c, 99.5))
    cfg   = dict(frames=FRAMES, elev=ELEV, cmap=CMAP)

    # ── keep only top-10% density points per panel ────────────────────────────
    for p in panels:
        mask = p['c'] >= p90
        for key in ['x', 'y', 'z', 'c']:
            p[key] = p[key][mask]
        print(f"  G0={p['g0']}  kept {mask.sum():,} / {len(mask):,} points (>= p90)")

    # ── put panel arrays into shared memory (workers attach, no copies) ────────
    shm_blocks = []   # keep alive until pool is done
    shm_meta   = []
    for p in panels:
        pmeta = {'g0': p['g0'], 'arrays': {}}
        for key in ['x', 'y', 'z', 'c']:
            arr = p[key]
            shm = SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            shm_blocks.append(shm)
            pmeta['arrays'][key] = (shm.name, arr.shape, arr.dtype.str)
        shm_meta.append(pmeta)

    n_workers = min(N_WORKERS, FRAMES)
    print(f"Rendering {FRAMES} frames on {n_workers} workers (shared memory)...")

    try:
        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(shm_meta, vmin, vmax, cfg),
        ) as pool:
            results = pool.map(_render_frame, range(FRAMES))
    finally:
        for shm in shm_blocks:
            shm.close()
            try:
                shm.unlink()
            except Exception:
                pass

    results.sort(key=lambda r: r[0])
    frame_images = [Image.open(io.BytesIO(r[1])) for r in results]

    print(f"Assembling {OUT_GIF}...")
    frame_images[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frame_images[1:],
        loop=0,
        duration=1000 // FPS,
        optimize=False,
    )
    print(f"Saved: {OUT_GIF}")

    try:
        import subprocess, tempfile, os
        with tempfile.TemporaryDirectory() as td:
            for i, img in enumerate(frame_images):
                img.save(os.path.join(td, f'frame_{i:04d}.png'))
            subprocess.run([
                'ffmpeg', '-y', '-framerate', str(FPS),
                '-i', os.path.join(td, 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                'cube_rotation.mp4',
            ], check=True, capture_output=True)
        print("Saved: cube_rotation.mp4")
    except Exception:
        pass


if __name__ == '__main__':
    main()
