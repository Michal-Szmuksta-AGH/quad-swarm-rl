"""
Wizualizacja NxM wariantow POJEDYNCZEJ topologii (default 4x4).

Do zobaczenia jak roznorodne sa mozliwe uklady w ramach jednej topologii —
uzyteczne dla defendability ("czy sample size wystarcza do generalizacji?").

Output: figures/topology_variety_<name>.png
Run:    python figures/topology_variety.py [--topology building] [--density 0.2]
                                            [--rows 4] [--cols 4]

Przyklady:
    python figures/topology_variety.py                              # 4x4 building
    python figures/topology_variety.py --topology grid              # 4x4 grid
    python figures/topology_variety.py --topology poisson --rows 3  # 3x4 poisson
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_art.quadrotor_multi.topology.generators import (
    TOPOLOGY_GENERATORS, generate_topology,
)


SPAWN = np.array([8.0, 8.0])
OBST_SIZE = 0.6
ROOM_H = 10.0


TITLES_PL = {
    'grid': 'Siatka regularna',
    'poisson': 'Rozklad Poissona',
    'cluster': 'Klastry gaussowskie',
    'building': 'Wnetrze budynku (BSP, sciany pod roznymi katami)',
}


def draw_panel(ax, name, seed, density):
    rng = np.random.RandomState(seed)
    _, pos_arr, _, stats = generate_topology(
        name, SPAWN, density, OBST_SIZE, ROOM_H,
        rng=rng, max_retries=10, return_stats=True,
    )

    hx, hy = SPAWN[0] / 2, SPAWN[1] / 2

    # Boundary
    ax.add_patch(patches.Rectangle(
        (-hx, -hy), SPAWN[0], SPAWN[1],
        linewidth=1.2, edgecolor='dimgray',
        facecolor='#fafafa', linestyle='--', zorder=0
    ))

    # Faint grid
    for x in np.arange(-hx, hx + 0.1, 1.0):
        ax.axvline(x, color='lightgray', linewidth=0.3, alpha=0.5, zorder=1)
    for y in np.arange(-hy, hy + 0.1, 1.0):
        ax.axhline(y, color='lightgray', linewidth=0.3, alpha=0.5, zorder=1)

    # Obstacles
    for x, y, _ in pos_arr:
        ax.add_patch(patches.Circle(
            (x, y), OBST_SIZE / 2,
            facecolor='#2E5090', edgecolor='#1A2F5C',
            linewidth=0.6, alpha=0.85, zorder=2
        ))

    # Info badge: seed + count + retry count if > 1 (or fallback flag)
    badge = f"ziarno={seed} n={len(pos_arr)}"
    if stats['fallback']:
        badge += " ⚠fallback"
    elif stats['attempts'] > 1:
        badge += f" (prób: {stats['attempts']})"
    badge_bg = '#FFEBEE' if stats['fallback'] else 'white'
    ax.text(hx - 0.3, -hy + 0.3, badge,
            fontsize=7, ha='right', va='bottom',
            color='dimgray',
            bbox=dict(boxstyle='round,pad=0.2',
                     facecolor=badge_bg, edgecolor='lightgray', alpha=0.9))

    ax.set_xlim(-hx - 0.3, hx + 0.3)
    ax.set_ylim(-hy - 0.3, hy + 0.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--topology', default='building',
                        choices=list(TOPOLOGY_GENERATORS.keys()),
                        help='Ktora topologie wizualizowac (default: building)')
    parser.add_argument('--density', type=float, default=0.2,
                        help='Obstacle density (default: 0.2)')
    parser.add_argument('--rows', type=int, default=4,
                        help='Ile wierszy paneli (default: 4)')
    parser.add_argument('--cols', type=int, default=4,
                        help='Ile kolumn paneli (default: 4)')
    parser.add_argument('--seed_start', type=int, default=0,
                        help='Pierwszy seed do uzycia (default: 0)')
    args = parser.parse_args()

    n = args.rows * args.cols
    fig, axes = plt.subplots(args.rows, args.cols,
                              figsize=(3.0 * args.cols, 3.0 * args.rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n):
        draw_panel(axes[i], args.topology,
                    seed=args.seed_start + i, density=args.density)

    plt.tight_layout()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'topology_variety_{args.topology}.png'
    )
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()
