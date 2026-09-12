"""
Wizualne porownanie SDF cutoff range — 3 modele SDF (paper r=∞, r=1.0, r=0.2)
w tej samej mapie przeszkod (grid topology). Kolorowy okrag pokazuje sensor_range,
poza nim SDF zwraca max = sensor_range (dla polityki "brak przeszkody w zasiegu").

Output: figures/sdf_cutoff_comparison.png
Run: python figures/sdf_cutoff_comparison.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Polskie znaki: matplotlib domyslnie DejaVu Sans (obsluguje pl chars)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

from gym_art.quadrotor_multi.topology.generators import grid_topology


SPAWN = np.array([8.0, 8.0])
DENSITY = 0.2
OBST_SIZE = 0.6
ROOM_HEIGHT = 10.0
OBST_RADIUS = OBST_SIZE / 2

DRONE_XY = (0.5, 0.5)  # blisko srodka spawn area
DRONE_YAW = 0.0

# Definicje 3 modeli
MODELS = [
    {'range':  10.0, 'label': 'Model bazowy (bez ograniczenia zasięgu, r = ∞)',
     'short':  'Model bazowy (r = ∞)', 'color': '#1F77B4'},
    {'range':   1.0, 'label': 'Percepcja ograniczona r = 1.0 m',
     'short':  'r = 1.0 m',     'color': '#2CA02C'},
    {'range':   0.2, 'label': 'Percepcja ograniczona r = 0.2 m',
     'short':  'r = 0.2 m',     'color': '#D62728'},
]


def draw_panel(ax, model, obst_pos_arr, dx, dy, dyaw):
    hx, hy = SPAWN[0] / 2, SPAWN[1] / 2

    # Boundary
    ax.add_patch(patches.Rectangle((-hx, -hy), SPAWN[0], SPAWN[1],
                                    linewidth=1.5, edgecolor='black',
                                    facecolor='#fafafa', zorder=0))
    # Grid
    for x in np.arange(-hx, hx + 0.01, 1.0):
        ax.axvline(x, color='lightgray', linewidth=0.3, alpha=0.5, zorder=1)
    for y in np.arange(-hy, hy + 0.01, 1.0):
        ax.axhline(y, color='lightgray', linewidth=0.3, alpha=0.5, zorder=1)

    # Sensor range circle
    range_val = model['range']
    if range_val >= 5.0:  # r = infty
        # Nie rysujemy kregu (widzi wszystko) — tylko cieniowanie calego pokoju
        ax.add_patch(patches.Rectangle((-hx, -hy), SPAWN[0], SPAWN[1],
                                        facecolor=model['color'], alpha=0.10,
                                        zorder=1.5))
        range_label = 'Zasięg = ∞\n(widzi cały pokój)'
    else:
        ax.add_patch(patches.Circle((dx, dy), range_val,
                                     facecolor=model['color'], alpha=0.18,
                                     edgecolor=model['color'], linewidth=2,
                                     linestyle='--', zorder=1.5))
        range_label = f'Zasięg czujnika = {range_val:.1f} m'

    # Obstacles: solid black gdy in-range, light gray gdy out-of-range
    n_in_range = 0
    n_total = len(obst_pos_arr)
    for p in obst_pos_arr:
        ox, oy = p[0], p[1]
        dist = np.hypot(ox - dx, oy - dy)
        edge_dist = dist - OBST_RADIUS  # najblizsza krawedz obstacle
        in_range = edge_dist <= range_val
        if in_range:
            n_in_range += 1
            ax.add_patch(patches.Circle((ox, oy), OBST_RADIUS,
                                         facecolor='#2E2E2E', edgecolor='#111',
                                         linewidth=0.6, alpha=0.9, zorder=3))
        else:
            ax.add_patch(patches.Circle((ox, oy), OBST_RADIUS,
                                         facecolor='#D0D0D0', edgecolor='#AAA',
                                         linewidth=0.5, alpha=0.5, zorder=2))

    # Drone marker + arrow
    ax.plot(dx, dy, marker='o', color='black', markersize=10, zorder=10)
    ax.annotate('',
                xy=(dx + np.cos(dyaw) * 0.4, dy + np.sin(dyaw) * 0.4),
                xytext=(dx, dy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.7),
                zorder=10)

    # Info panel
    ax.text(-hx + 0.15, hy - 0.35,
            f'{range_label}\n'
            f'Widocznych przeszkód: {n_in_range} / {n_total}',
            fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=model['color'], linewidth=1.2, alpha=0.95),
            zorder=6)

    ax.set_xlim(-hx - 0.3, hx + 0.3)
    ax.set_ylim(-hy - 0.3, hy + 0.3)
    ax.set_aspect('equal')
    ax.set_title(model['label'], fontsize=11, fontweight='bold', pad=10,
                  color=model['color'])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    # Generuj ten sam grid topology dla wszystkich 3 modeli (deterministic seed)
    rng = np.random.RandomState(42)
    _, obst_pos_arr, _ = grid_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_HEIGHT, rng=rng)

    dx, dy = DRONE_XY

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    for i, m in enumerate(MODELS):
        draw_panel(axes[i], m, obst_pos_arr, dx, dy, DRONE_YAW)

    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'sdf_cutoff_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
