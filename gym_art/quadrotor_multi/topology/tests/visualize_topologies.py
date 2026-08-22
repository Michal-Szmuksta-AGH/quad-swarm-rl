"""
Visual sanity check dla topology generators.
Run: python -m gym_art.quadrotor_multi.topology.tests.visualize_topologies
Wyjscie: /tmp/topology_validation.png
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gym_art.quadrotor_multi.topology.generators import TOPOLOGY_GENERATORS


SPAWN = np.array([8.0, 8.0])
DENSITY = 0.2
OBST_SIZE = 0.6
ROOM_H = 10.0


def draw_topology(ax, title, name, seed=0):
    rng = np.random.RandomState(seed)
    obst_map, pos_arr, _ = TOPOLOGY_GENERATORS[name](
        SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)

    hx, hy = SPAWN[0] / 2, SPAWN[1] / 2
    # Spawn boundary
    ax.add_patch(patches.Rectangle((-hx, -hy), SPAWN[0], SPAWN[1],
                                    linewidth=2, edgecolor='black',
                                    facecolor='none', linestyle='--'))
    # Grid lines (grid_size=1.0)
    for x in np.arange(-hx, hx + 0.1, 1.0):
        ax.axvline(x, color='gray', linewidth=0.3, alpha=0.4)
    for y in np.arange(-hy, hy + 0.1, 1.0):
        ax.axhline(y, color='gray', linewidth=0.3, alpha=0.4)

    # Obstacles as circles
    for x, y, _ in pos_arr:
        ax.add_patch(patches.Circle((x, y), OBST_SIZE / 2,
                                     color='steelblue', alpha=0.6,
                                     edgecolor='navy', linewidth=1))

    ax.set_xlim(-hx - 0.5, hx + 0.5)
    ax.set_ylim(-hy - 0.5, hy + 0.5)
    ax.set_aspect('equal')
    ax.set_title(f"{title}\n({len(pos_arr)} obstacles)", fontsize=11)
    ax.grid(False)


def main(out_path="/tmp/topology_validation.png"):
    names = list(TOPOLOGY_GENERATORS.keys())
    # 3 rzedy — zebysmy zobaczyli warianty (szczegolnie dla corridor 4 orientations)
    n_rows = 3
    fig, axes = plt.subplots(n_rows, len(names), figsize=(4 * len(names), 3.5 * n_rows))

    for i, name in enumerate(names):
        for row in range(n_rows):
            draw_topology(axes[row, i], f"{name} (seed={row})", name, seed=row)

    fig.suptitle(f"Topology generators — density={DENSITY}, "
                 f"obst_size={OBST_SIZE}m, spawn={SPAWN[0]}x{SPAWN[1]}m",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
