"""
evaluate.py
Run all models and produce a unified comparison table + visualizations.

Outputs:
  - Console: per-fold and mean±std metrics for all 4 models
  - Console: XGBoost feature importance ranking
  - PyVista window: true vs predicted nH2 slice for the CNN on one held-out cube

Usage:
    python evaluate.py [--cnn-epochs N] [--skip-cnn]
"""

import argparse
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for figure saving
import matplotlib.pyplot as plt

from data_loader    import load_all_cubes, get_X_y, get_g0_values, FEATURE_COLS, LOG_TARGET_COL
from classical_models import (run_linear, run_xgboost, run_mlp,
                               print_feature_importance, compute_metrics)
from train_cnn      import run_cnn_cv


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(all_results: dict[str, list[dict]], g0_values: list[float]) -> None:
    models = list(all_results.keys())
    print("\n" + "="*80)
    print("  SUMMARY — mean +/- std (leave-one-G0-out, nH2 space)")
    print("="*80)
    print(f"  {'Model':<20} {'R²':>10} {'RMSE':>14} {'MAE':>14}")
    print(f"  {'-'*58}")
    for name in models:
        metrics = all_results[name]
        r2s   = [m['R2']   for m in metrics]
        rmses = [m['RMSE'] for m in metrics]
        maes  = [m['MAE']  for m in metrics]
        print(f"  {name:<20} "
              f"{np.mean(r2s):>6.4f}±{np.std(r2s):.4f}  "
              f"{np.mean(rmses):>10.3e}±{np.std(rmses):.3e}  "
              f"{np.mean(maes):>10.3e}±{np.std(maes):.3e}")
    print("="*80)


# ── JSON results log ──────────────────────────────────────────────────────────

def save_results_log(all_results: dict[str, list[dict]],
                     g0_values:   list[float],
                     run_config:  dict,
                     log_path:    str) -> None:
    models_log = {}
    for name, fold_metrics in all_results.items():
        fold_entries = [
            {'fold': i, 'g0': g0, 'metrics': {k: float(v) for k, v in m.items()}}
            for i, (g0, m) in enumerate(zip(g0_values, fold_metrics))
        ]
        summary = {}
        for metric in ('R2', 'RMSE', 'MAE'):
            vals = [m[metric] for m in fold_metrics]
            summary[metric] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
        models_log[name] = {'folds': fold_entries, 'summary': summary}

    log = {'run_config': run_config, 'g0_values': g0_values, 'models': models_log}
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Results log saved -> {log_path}")


# ── Per-G0 R² bar chart ───────────────────────────────────────────────────────

def plot_r2_comparison(all_results: dict[str, list[dict]],
                       g0_values: list[float],
                       save_path: str = 'r2_comparison.png') -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(g0_values))
    width = 0.2
    models = list(all_results.keys())
    colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

    for i, (name, metrics) in enumerate(all_results.items()):
        r2s = [m['R2'] for m in metrics]
        ax.bar(x + i * width, r2s, width, label=name, color=colors[i % len(colors)], alpha=0.8)

    ax.set_xlabel('Held-out G0 value')
    ax.set_ylabel('R² (nH2 space)')
    ax.set_title('Leave-one-G0-out R² by model')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([f'{g:.1f}' for g in g0_values])
    ax.legend()
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved R2 comparison chart -> {save_path}")
    plt.close()


# ── True vs predicted scatter ─────────────────────────────────────────────────

def plot_scatter(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                 model_name: str, g0: float,
                 save_path: str | None = None) -> None:
    y_true = 10.0 ** y_true_log
    y_pred = 10.0 ** y_pred_log

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=0.5, alpha=0.3, color='steelblue', rasterized=True)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=1, label='y=x')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('True nH2')
    ax.set_ylabel('Predicted nH2')
    ax.set_title(f'{model_name}  (G0={g0:.1f} held out)')
    ax.legend()
    plt.tight_layout()
    path = save_path or f'scatter_{model_name.replace(" ", "_")}_G0{g0}.png'
    plt.savefig(path, dpi=150)
    print(f"Saved scatter -> {path}")
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cnn',    action='store_true',
                        help='Skip CNN training (faster, classical models only)')
    parser.add_argument('--cnn-epochs',  type=int, default=150)
    parser.add_argument('--cnn-all-ops', action='store_true',
                        help='Use all 48 Oh ops for CNN augmentation')
    parser.add_argument('--log',         type=str, default=None,
                        help='Path for JSON results log (default: evaluation_TIMESTAMP.json)')
    args = parser.parse_args()

    _ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    _run_config = {
        'timestamp':    datetime.datetime.now().isoformat(timespec='seconds'),
        'skip_cnn':     args.skip_cnn,
        'cnn_epochs':   args.cnn_epochs,
        'cnn_all_ops':  args.cnn_all_ops,
    }

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    cubes   = load_all_cubes()
    g0_vals = get_g0_values(cubes)
    X, y, folds = get_X_y(cubes, use_log_target=True)
    print(f"Total samples: {len(X):,}")

    all_results: dict[str, list[dict]] = {}

    # ── 1. Linear Regression ──────────────────────────────────────────────────
    print("\n[1/4] Linear Regression")
    all_results['Linear Regression'] = run_linear(X, y, folds, g0_vals)

    # ── 2. XGBoost ────────────────────────────────────────────────────────────
    print("\n[2/4] XGBoost")
    xgb_metrics, xgb_models = run_xgboost(X, y, folds, g0_vals)
    all_results['XGBoost'] = xgb_metrics
    print_feature_importance(xgb_models, FEATURE_COLS)

    # ── 3. MLP ────────────────────────────────────────────────────────────────
    print("\n[3/4] MLP")
    all_results['MLP'] = run_mlp(X, y, folds, g0_vals)

    # ── 4. CNN ────────────────────────────────────────────────────────────────
    if not args.skip_cnn:
        print("\n[4/4] 3D U-Net CNN")
        all_results['3D U-Net'] = run_cnn_cv(
            safe_only=not args.cnn_all_ops,
            epochs=args.cnn_epochs,
        )
    else:
        print("\n[4/4] CNN skipped (--skip-cnn)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(all_results, g0_vals)

    # ── Results log ───────────────────────────────────────────────────────────
    log_path = args.log or f'evaluation_{_ts}.json'
    save_results_log(all_results, g0_vals, _run_config, log_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_r2_comparison(all_results, g0_vals)
