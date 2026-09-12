"""
Heatmapy cross-eval — 2 figury dla pracy magisterskiej / promotora:
  1. Perception cross-eval: 3 modele SDF x 3 zasiegi testowe
  2. Topology cross-eval:   4 modele x 4 topologie

Output:
  figures/perception_matrix.png  (2 subplots: success + collisions)
  figures/topology_matrix.png    (2 subplots: success + collisions)

Run: python figures/generate_cross_eval_matrices.py
"""

import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _load_metric(train_dir, model, tag_file, metric_key):
    """Load one metric from an eval_metrics JSON."""
    path = Path(train_dir) / model / tag_file
    if not path.exists():
        return None
    d = json.load(open(path))
    return d['metrics'].get(metric_key, {}).get('mean')


def perception_matrices(train_dir='train_dir'):
    """Return (success 3x3, collision 3x3, row_labels, col_labels)."""
    models = [
        ('paper_baseline_8drones_s0', 'Model bazowy\n(r=∞)'),
        ('perception_limited_r1.0_8drones_s0', 'Ograniczona\npercepcja (r=1.0)'),
        ('perception_limited_r0.2_8drones_s0', 'Ograniczona\npercepcja (r=0.2)'),
    ]
    test_ranges = [
        ('100.0', 'test\nr=∞'),
        ('1.0', 'test\nr=1.0m'),
        ('0.2', 'test\nr=0.2m'),
    ]
    n = len(models)
    m = len(test_ranges)
    succ = np.full((n, m), np.nan)
    coll = np.full((n, m), np.nan)
    for i, (model, _) in enumerate(models):
        for j, (rng, _) in enumerate(test_ranges):
            tag = f'eval_metrics_r{rng}.json'
            s = _load_metric(train_dir, model, tag, 'metric/agent_success_rate')
            c = _load_metric(train_dir, model, tag, 'num_collisions_obst_quad')
            if s is not None:
                succ[i, j] = s
            if c is not None:
                coll[i, j] = c
    return succ, coll, [m[1] for m in models], [t[1] for t in test_ranges]


def topology_matrices(train_dir='train_dir'):
    """Return (success 5x4, collision 5x4, row_labels, col_labels)."""
    models = [
        ('paper_baseline_8drones_s0', 'Model bazowy\n(SDF, r=∞)'),
        ('perception_limited_r1.0_8drones_s0', 'Ograniczona percepcja\n(SDF r=1.0)'),
        ('perception_limited_r0.2_8drones_s0', 'Ograniczona percepcja\n(SDF r=0.2)'),
        ('multiranger_r4.0_8drones_s0', 'Multi-ranger\n(tylko siatka)'),
        # Checkpoint 1B (nie 1.5B) — rowny budzet treningowy z pozostalymi 4 modelami.
        ('multiranger_r4.0_topomix_8drones_s0_1B', 'Multi-ranger\n+ mieszanka topologii'),
    ]
    topologies = [
        ('grid', 'Siatka'),
        ('poisson', 'Poisson'),
        ('cluster', 'Klastry'),
        ('building', 'Budynek\n(BSP)'),
    ]
    n = len(models)
    m = len(topologies)
    succ = np.full((n, m), np.nan)
    coll = np.full((n, m), np.nan)
    for i, (model, _) in enumerate(models):
        for j, (topo, _) in enumerate(topologies):
            tag = f'eval_metrics_topo_{topo}.json'
            s = _load_metric(train_dir, model, tag, 'metric/agent_success_rate')
            c = _load_metric(train_dir, model, tag, 'num_collisions_obst_quad')
            if s is not None:
                succ[i, j] = s
            if c is not None:
                coll[i, j] = c
    return succ, coll, [m[1] for m in models], [t[1] for t in topologies]


# --------------------------------------------------------------------------
# Heatmap drawing
# --------------------------------------------------------------------------
def draw_heatmap(ax, data, row_labels, col_labels, title,
                 cmap, vmin, vmax, fmt='{:.3f}', cbar_label=None,
                 highlight_diagonal=False):
    """Draw one heatmap with annotations."""
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

    # Ticks
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)

    # Grid between cells
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)

    # Annotate cells
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            # Text color: dark on light bg, light on dark bg
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = 'white' if norm_val > 0.55 else '#1a1a1a'
            text = fmt.format(val)
            weight = 'normal'
            if highlight_diagonal and i == j:
                weight = 'bold'
            ax.text(j, i, text, ha='center', va='center',
                   color=color, fontsize=11, fontweight=weight)

    ax.set_title(title, fontsize=12, pad=10, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


# --------------------------------------------------------------------------
# Figure builders
# --------------------------------------------------------------------------
def build_perception_figure(train_dir='train_dir'):
    succ, coll, row_lbls, col_lbls = perception_matrices(train_dir)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    draw_heatmap(
        axes[0], succ, row_lbls, col_lbls,
        title='Skuteczność (wyżej = lepiej)',
        cmap='RdYlGn', vmin=0.4, vmax=1.0,
        fmt='{:.3f}',
        cbar_label='Średni sukces (dolot bez kolizji)',
        highlight_diagonal=True,
    )
    axes[0].set_xlabel('Zasięg czujnika przy TEŚCIE', fontsize=11, labelpad=8)
    axes[0].set_ylabel('Model wytrenowany na:', fontsize=11, labelpad=8)

    draw_heatmap(
        axes[1], coll, row_lbls, col_lbls,
        title='Kolizje z przeszkodami / epizod (niżej = lepiej)',
        cmap='Reds', vmin=0.0, vmax=max(4.0, np.nanmax(coll) * 1.05),
        fmt='{:.2f}',
        cbar_label='Kolizje z przeszkodami / epizod (średnia)',
        highlight_diagonal=True,
    )
    axes[1].set_xlabel('Zasięg czujnika przy TEŚCIE', fontsize=11, labelpad=8)
    axes[1].set_ylabel('Model wytrenowany na:', fontsize=11, labelpad=8)

    plt.tight_layout()
    out = 'figures/perception_matrix.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out}')


def build_topology_figure(train_dir='train_dir'):
    succ, coll, row_lbls, col_lbls = topology_matrices(train_dir)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    draw_heatmap(
        axes[0], succ, row_lbls, col_lbls,
        title='Skuteczność (wyżej = lepiej)',
        cmap='RdYlGn', vmin=0.5, vmax=1.0,
        fmt='{:.3f}',
        cbar_label='Średni sukces (dolot bez kolizji)',
    )
    axes[0].set_xlabel('Topologia rozmieszczenia przeszkód (TEST)',
                       fontsize=11, labelpad=8)
    axes[0].set_ylabel('Model', fontsize=11, labelpad=8)

    draw_heatmap(
        axes[1], coll, row_lbls, col_lbls,
        title='Kolizje z przeszkodami / epizod (niżej = lepiej)',
        cmap='Reds', vmin=0.0, vmax=np.nanmax(coll) * 1.05,
        fmt='{:.2f}',
        cbar_label='Kolizje z przeszkodami / epizod (średnia)',
    )
    axes[1].set_xlabel('Topologia rozmieszczenia przeszkód (TEST)',
                       fontsize=11, labelpad=8)
    axes[1].set_ylabel('Model', fontsize=11, labelpad=8)

    plt.tight_layout()
    out = 'figures/topology_matrix.png'
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out}')


if __name__ == '__main__':
    build_perception_figure()
    build_topology_figure()
