"""
Polished visualization topologii dla pracy magisterskiej / promotora.
Output: figures/topology_maps.png (persistent, w repo)
Run:    python figures/generate_topology_figure.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_art.quadrotor_multi.topology.generators import TOPOLOGY_GENERATORS


SPAWN = np.array([8.0, 8.0])
DENSITY = 0.2
OBST_SIZE = 0.6
ROOM_H = 10.0


TITLES_PL = {
    'grid': 'Siatka regularna\n(baseline paperu Huang et al.)',
    'poisson': 'Rozklad Poissona\n(uniform random z min. odstepem)',
    'cluster': 'Klastry gaussowskie\n(3 grupy przeszkod)',
    'building': 'Wnetrze budynku\n(BSP: sciany pod roznymi katami, drzwi)',
}


def draw_topology(ax, name, seed):
    rng = np.random.RandomState(seed)
    obst_map, pos_arr, _ = TOPOLOGY_GENERATORS[name](
        SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)

    hx, hy = SPAWN[0] / 2, SPAWN[1] / 2

    # Spawn area boundary (dashed)
    ax.add_patch(patches.Rectangle(
        (-hx, -hy), SPAWN[0], SPAWN[1],
        linewidth=1.5, edgecolor='dimgray',
        facecolor='#fafafa', linestyle='--', zorder=0
    ))

    # Faint grid
    for x in np.arange(-hx, hx + 0.1, 1.0):
        ax.axvline(x, color='lightgray', linewidth=0.4, alpha=0.5, zorder=1)
    for y in np.arange(-hy, hy + 0.1, 1.0):
        ax.axhline(y, color='lightgray', linewidth=0.4, alpha=0.5, zorder=1)

    # Obstacles
    for x, y, _ in pos_arr:
        ax.add_patch(patches.Circle(
            (x, y), OBST_SIZE / 2,
            facecolor='#2E5090', edgecolor='#1A2F5C',
            linewidth=0.8, alpha=0.85, zorder=2
        ))

    # Small count annotation in corner
    ax.text(hx - 0.3, -hy + 0.3, f"n={len(pos_arr)}",
            fontsize=8, ha='right', va='bottom',
            color='dimgray',
            bbox=dict(boxstyle='round,pad=0.25',
                     facecolor='white', edgecolor='lightgray', alpha=0.9))

    ax.set_xlim(-hx - 0.4, hx + 0.4)
    ax.set_ylim(-hy - 0.4, hy + 0.4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'topology_maps.png')
    names = list(TOPOLOGY_GENERATORS.keys())
    n_rows = 3

    fig, axes = plt.subplots(n_rows, len(names),
                              figsize=(3.5 * len(names), 3.3 * n_rows))

    for i, name in enumerate(names):
        for row in range(n_rows):
            ax = axes[row, i]
            draw_topology(ax, name, seed=row)
            if row == 0:
                ax.set_title(TITLES_PL[name], fontsize=11, pad=10, fontweight='bold')
            # Row label on leftmost
            if i == 0:
                ax.set_ylabel(f"Seed {row}", fontsize=10, rotation=0,
                               labelpad=30, va='center', color='dimgray')

    fig.suptitle(
        f"Topologie rozmieszczenia przeszkod — obszar {int(SPAWN[0])}x{int(SPAWN[1])}m, "
        f"gestosc {DENSITY}, przeszkody o srednicy {OBST_SIZE}m",
        fontsize=13, y=0.99, fontweight='bold'
    )

    plt.tight_layout(rect=[0.02, 0, 1, 0.97])
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
