"""
Wizualne porownanie sposobu percepcji: SDF (paper Huang) vs Multiranger.
Jedna para wykresow dla zageszczonego srodowiska (najciekawszy przypadek).

Kluczowa roznica skali:
  - SDF: 9 wartosci samplowanych w gridzie 3x3 o rozstawie 0.1m — praktycznie
    punktowo wokol drona. Aby to zwizualizowac w skali pokoju 8x8m,
    uzywamy INSET ZOOM (przybliżenie ~16x) pokazujacego pełny grid.
  - Multiranger: 4 rays z max 4m — widoczne w pełnej skali pokoju.

Zaden z sensorow nie widzi innych dronow — spatial awareness dla nich
pochodzi z osobnego neighbor observation channel.

Output: figures/sensor_comparison.png
Run: python figures/sensor_comparison.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs
from gym_art.quadrotor_multi.sensors.multiranger import (
    get_multiranger_obs, SENSOR_BODY_ANGLES,
)


ROOM_DIMS = (8.0, 8.0)
OBST_RADIUS = 0.3
DRONE_RADIUS = 0.046
MULTIRANGER_MAX = 4.0
FOV_DEG = 27.0
NUM_RAYS = 8
ZOOM_SIZE = 0.5    # 50cm wycinek wokol drona w SDF inset

SENSOR_COLORS = ['red', 'green', 'blue', 'orange']
SENSOR_LABELS = ['front', 'left', 'back', 'right']

SDF_CMAP = LinearSegmentedColormap.from_list(
    'sdf', ['#D62728', '#F0BF4B', '#66BB6A'], N=256)


def _draw_scene(ax, drone_xy, drone_yaw, obstacles_xy, neighbor_drones_xy,
                show_neighbor_labels=True):
    hx, hy = ROOM_DIMS[0] / 2, ROOM_DIMS[1] / 2
    ax.add_patch(patches.Rectangle((-hx, -hy), ROOM_DIMS[0], ROOM_DIMS[1],
                                    linewidth=1.5, edgecolor='black', facecolor='none'))
    # Faint grid
    for x in np.arange(-hx, hx + 0.01, 1.0):
        ax.axvline(x, color='lightgray', linewidth=0.3, alpha=0.4, zorder=0)
    for y in np.arange(-hy, hy + 0.01, 1.0):
        ax.axhline(y, color='lightgray', linewidth=0.3, alpha=0.4, zorder=0)

    for ox, oy in obstacles_xy:
        ax.add_patch(patches.Circle((ox, oy), OBST_RADIUS,
                                     facecolor='#3A3A3A', edgecolor='#111',
                                     linewidth=0.5, alpha=0.85, zorder=2))
    for nx, ny in neighbor_drones_xy:
        ax.add_patch(patches.Circle((nx, ny), DRONE_RADIUS * 4,
                                     facecolor='#87CEEB', edgecolor='#333',
                                     linewidth=1.0, alpha=0.7, zorder=2))
        if show_neighbor_labels:
            ax.plot(nx, ny, marker='X', color='darkred', markersize=8,
                    markeredgewidth=1.0, zorder=3)
            ax.text(nx, ny - 0.4, 'sasiad\n(niewidoczny)', fontsize=6, ha='center',
                    color='darkred', style='italic', zorder=3)


def _draw_drone(ax, dx, dy, dyaw, marker_size=8, arrow_len=0.35):
    ax.plot(dx, dy, marker='o', color='black', markersize=marker_size, zorder=10)
    ax.annotate('', xy=(dx + np.cos(dyaw) * arrow_len, dy + np.sin(dyaw) * arrow_len),
                xytext=(dx, dy),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=10)


def draw_sdf_panel(ax, drone_xy, drone_yaw, obstacles_xy, neighbor_drones_xy,
                    sensor_range=100.0, resolution=0.1):
    _draw_scene(ax, drone_xy, drone_yaw, obstacles_xy, neighbor_drones_xy)
    dx, dy = drone_xy

    # SDF computation
    quad_poses = np.array([[dx, dy]], dtype=np.float64)
    obst_poses = np.array(obstacles_xy, dtype=np.float64) if obstacles_xy \
        else np.zeros((0, 2))
    quads_sdf_obs = np.zeros((1, 9), dtype=np.float64)
    quads_sdf_obs = get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs,
                                       OBST_RADIUS, resolution=resolution,
                                       sensor_range=sensor_range)

    _draw_drone(ax, dx, dy, drone_yaw)

    # Info label
    min_read = float(quads_sdf_obs.min())
    ax.text(-ROOM_DIMS[0]/2 + 0.15, ROOM_DIMS[1]/2 - 0.4,
            f'SDF: 9 wartości (siatka 3×3, rozstaw 0.1 m)\n'
            f'min = {min_read:.2f} m  (siatka pokazana w powiększeniu →)',
            fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='gray'), zorder=6)

    # ------------------ INSET ZOOM ------------------
    axins = ax.inset_axes([0.58, 0.05, 0.4, 0.4])
    # Zoomed area bounds
    zx0, zx1 = dx - ZOOM_SIZE / 2, dx + ZOOM_SIZE / 2
    zy0, zy1 = dy - ZOOM_SIZE / 2, dy + ZOOM_SIZE / 2

    # Redraw obstacles that overlap zoom area (dla context)
    for ox, oy in obstacles_xy:
        # Jeśli obstacle jest w zoomed area (edge-inclusive)
        if abs(ox - dx) < ZOOM_SIZE / 2 + OBST_RADIUS and \
           abs(oy - dy) < ZOOM_SIZE / 2 + OBST_RADIUS:
            axins.add_patch(patches.Circle((ox, oy), OBST_RADIUS,
                                            facecolor='#3A3A3A', edgecolor='#111',
                                            linewidth=0.6, alpha=0.85, zorder=2))

    # Faint sub-grid co 0.1m w insecie (=SDF resolution)
    for gx in np.arange(zx0, zx1 + 0.01, 0.1):
        axins.axvline(gx, color='lightgray', linewidth=0.4, alpha=0.4, zorder=1)
    for gy in np.arange(zy0, zy1 + 0.01, 0.1):
        axins.axhline(gy, color='lightgray', linewidth=0.4, alpha=0.4, zorder=1)

    # Punkty SDF w insecie — duze markery
    for g_i, ox in enumerate([-resolution, 0, resolution]):
        for g_j, oy in enumerate([-resolution, 0, resolution]):
            sx = dx + ox
            sy = dy + oy
            g_id = g_i * 3 + g_j
            sdf_val = quads_sdf_obs[0, g_id]
            display_max = min(sensor_range, 3.0)
            v_norm = min(sdf_val / display_max, 1.0)
            color = SDF_CMAP(v_norm)
            axins.plot(sx, sy, marker='o', color=color, markersize=22,
                        markeredgecolor='black', markeredgewidth=0.8, zorder=5)
            # Wartosc labele
            axins.text(sx + 0.028, sy, f'{sdf_val:.2f}',
                        fontsize=8, ha='left', va='center', fontweight='bold',
                        color='#111',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white', alpha=0.95, edgecolor='none'),
                        zorder=6)

    _draw_drone(axins, dx, dy, drone_yaw, marker_size=10, arrow_len=0.08)

    axins.set_xlim(zx0, zx1)
    axins.set_ylim(zy0, zy1)
    axins.set_xticks([])
    axins.set_yticks([])
    zoom_ratio = ROOM_DIMS[0] / ZOOM_SIZE
    axins.set_title(f'Powiększenie ×{zoom_ratio:.0f} — siatka SDF 3×3 (rozstaw 0.1 m)',
                     fontsize=9, color='blue')
    for spine in axins.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(1.5)

    # Connector: rectangle w main + linie do inset
    ax.indicate_inset_zoom(axins, edgecolor='blue', linewidth=1.5,
                            alpha=0.7)


def draw_multiranger_panel(ax, drone_xy, drone_yaw, obstacles_xy, neighbor_drones_xy):
    _draw_scene(ax, drone_xy, drone_yaw, obstacles_xy, neighbor_drones_xy)
    dx, dy = drone_xy

    quads = np.array([drone_xy], dtype=np.float32)
    yaws = np.array([drone_yaw], dtype=np.float32)
    obst = np.array(obstacles_xy, dtype=np.float32) if obstacles_xy \
        else np.zeros((0, 2), dtype=np.float32)
    room = np.array(ROOM_DIMS, dtype=np.float32)
    fov_rad = np.deg2rad(FOV_DEG)
    obs = get_multiranger_obs(quads, yaws, obst, OBST_RADIUS, room,
                               MULTIRANGER_MAX, 0.0, fov_rad, NUM_RAYS)

    for s_id in range(4):
        body_ang = SENSOR_BODY_ANGLES[s_id]
        base_ang = drone_yaw + body_ang
        color = SENSOR_COLORS[s_id]
        dist = obs[0, s_id]

        ang_l = base_ang + fov_rad / 2
        ang_r = base_ang - fov_rad / 2
        ex_l = dx + np.cos(ang_l) * dist
        ey_l = dy + np.sin(ang_l) * dist
        ex_r = dx + np.cos(ang_r) * dist
        ey_r = dy + np.sin(ang_r) * dist
        ax.add_patch(patches.Polygon([(dx, dy), (ex_l, ey_l), (ex_r, ey_r)],
                                      color=color, alpha=0.15, zorder=3))
        for edge in [-fov_rad / 2, fov_rad / 2]:
            a = base_ang + edge
            ax.plot([dx, dx + np.cos(a) * dist], [dy, dy + np.sin(a) * dist],
                    color=color, linewidth=0.7, alpha=0.4, linestyle=':', zorder=3)
        cx = dx + np.cos(base_ang) * dist
        cy = dy + np.sin(base_ang) * dist
        ax.plot([dx, cx], [dy, cy], color=color, linewidth=1.5, alpha=0.85, zorder=4)
        mx = dx + np.cos(base_ang) * dist * 0.65
        my = dy + np.sin(base_ang) * dist * 0.65
        ax.text(mx, my, f'{dist:.2f}m', color=color, fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          alpha=0.9, edgecolor='none'), zorder=5)

    _draw_drone(ax, dx, dy, drone_yaw, marker_size=10, arrow_len=0.4)

    ax.text(-ROOM_DIMS[0]/2 + 0.15, ROOM_DIMS[1]/2 - 0.4,
            f'Multiranger: 4 wartości\n'
            f'(front, left, back, right, max = 4 m)',
            fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='gray'), zorder=6)


def _finalize(ax, title):
    hx, hy = ROOM_DIMS[0] / 2, ROOM_DIMS[1] / 2
    ax.set_xlim(-hx - 0.3, hx + 0.3)
    ax.set_ylim(-hy - 0.3, hy + 0.3)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    # Jeden scenariusz: zageszczone srodowisko. Dron blisko obstacle (ale nie na nim)
    # zeby SDF pokazal positive gradient (bez sasiada — bez confusion).
    drone_xy = (0, 0)
    drone_yaw = np.pi / 4
    obstacles = [(0.55, 0.05), (-1.5, 1.5), (0.5, -2.0),
                 (2.5, -1.5), (-2.5, -0.5), (1.5, 2.5)]
    neighbor_drones = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    draw_sdf_panel(axes[0], drone_xy, drone_yaw, obstacles, neighbor_drones)
    _finalize(axes[0], 'SDF (konfiguracja bazowa)\n'
                        '9 wartości w siatce 3×3 wokół drona (rozstaw 0.1 m)')

    draw_multiranger_panel(axes[1], drone_xy, drone_yaw, obstacles, neighbor_drones)
    _finalize(axes[1], 'Multiranger (model przyjęty w pracy)\n'
                        '4 wartości (front, left, back, right, max 4 m)')

    plt.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'sensor_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
