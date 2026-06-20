"""
Visual sanity check: render drones with their Multiranger sensor rays.
Run: python -m gym_art.quadrotor_multi.sensors.tests.visualize_multiranger
Outputs: /tmp/multiranger_validation.png

Compare visually:
  - rays should emanate from drones in correct body-frame directions
  - rays should stop at obstacles or walls
  - colored dots mark where each ray ends (hit point)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from gym_art.quadrotor_multi.sensors.multiranger import (
    get_multiranger_obs, SENSOR_BODY_ANGLES,
)


SENSOR_COLORS = ['red', 'green', 'blue', 'orange']      # front, left, back, right
SENSOR_LABELS = ['front (+x body)', 'left (+y body)', 'back (-x body)', 'right (-y body)']


def draw_scene(ax, title, drone_xy, drone_yaw, obstacles_xy, obst_r, room_dims,
               max_range, fov_deg=27.0, num_rays=8):
    half_x = room_dims[0] / 2
    half_y = room_dims[1] / 2

    # Room walls
    room_rect = patches.Rectangle((-half_x, -half_y), room_dims[0], room_dims[1],
                                   linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(room_rect)

    # Obstacles
    for ox, oy in obstacles_xy:
        ax.add_patch(patches.Circle((ox, oy), obst_r, color='gray', alpha=0.5))

    # Compute sensor obs for the single drone using FULL cone
    quads = np.array([drone_xy], dtype=np.float32)
    yaws = np.array([drone_yaw], dtype=np.float32)
    obst = np.array(obstacles_xy, dtype=np.float32) if obstacles_xy else np.zeros((0, 2), dtype=np.float32)
    room_arr = np.array(room_dims, dtype=np.float32)
    fov_rad = np.deg2rad(fov_deg)
    obs = get_multiranger_obs(quads, yaws, obst, obst_r, room_arr, max_range,
                              0.0, fov_rad, num_rays)

    dx, dy = drone_xy
    for s_id in range(4):
        body_ang = SENSOR_BODY_ANGLES[s_id]
        base_world_ang = drone_yaw + body_ang
        color = SENSOR_COLORS[s_id]
        distance = obs[0, s_id]

        # Draw cone boundary rays (faint)
        for edge_offset in [-fov_rad / 2, fov_rad / 2]:
            ang = base_world_ang + edge_offset
            ex = dx + np.cos(ang) * distance
            ey = dy + np.sin(ang) * distance
            ax.plot([dx, ex], [dy, ey], color=color, linewidth=0.7,
                    alpha=0.3, linestyle=':')

        # Fill cone with light translucent triangle
        ang_left = base_world_ang + fov_rad / 2
        ang_right = base_world_ang - fov_rad / 2
        ex_l = dx + np.cos(ang_left) * distance
        ey_l = dy + np.sin(ang_left) * distance
        ex_r = dx + np.cos(ang_right) * distance
        ey_r = dy + np.sin(ang_right) * distance
        cone = patches.Polygon([(dx, dy), (ex_l, ey_l), (ex_r, ey_r)],
                                color=color, alpha=0.08)
        ax.add_patch(cone)

        # Central (axis) ray solid — this is the "indicator", actual reading is min over cone
        cx = dx + np.cos(base_world_ang) * distance
        cy = dy + np.sin(base_world_ang) * distance
        ax.plot([dx, cx], [dy, cy], color=color, linewidth=1.5, alpha=0.85)
        ax.scatter([cx], [cy], color=color, s=40, zorder=5,
                   edgecolors='black', linewidths=0.5)

        # Distance label
        mid_x = dx + np.cos(base_world_ang) * distance * 0.7
        mid_y = dy + np.sin(base_world_ang) * distance * 0.7
        ax.text(mid_x, mid_y, f"{distance:.2f}", color=color,
                fontsize=7, ha='center', bbox=dict(boxstyle='round,pad=0.2',
                facecolor='white', alpha=0.85, edgecolor='none'))

    # Drone marker + heading arrow
    ax.scatter([dx], [dy], color='black', s=80, marker='o', zorder=10)
    arrow_dx = np.cos(drone_yaw) * 0.3
    arrow_dy = np.sin(drone_yaw) * 0.3
    ax.annotate('', xy=(dx + arrow_dx, dy + arrow_dy), xytext=(dx, dy),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

    ax.set_xlim(-half_x - 0.5, half_x + 0.5)
    ax.set_ylim(-half_y - 0.5, half_y + 0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)


def main(out_path="/tmp/multiranger_validation.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Scenario 1: empty room, drone at center, yaw=0 — sees 4 walls
    draw_scene(axes[0, 0], "Empty room, yaw=0\n(expect 4 walls)",
               drone_xy=(0, 0), drone_yaw=0.0, obstacles_xy=[],
               obst_r=0.3, room_dims=(8, 8), max_range=4.0)

    # Scenario 2: yaw=0, obstacle in front
    draw_scene(axes[0, 1], "Obstacle in front, yaw=0\n(red ray hits cylinder)",
               drone_xy=(0, 0), drone_yaw=0.0, obstacles_xy=[(2.5, 0)],
               obst_r=0.5, room_dims=(8, 8), max_range=4.0)

    # Scenario 3: yaw=90, same obstacle — now on RIGHT sensor
    draw_scene(axes[0, 2], "Same obstacle, drone yaw=90deg\n(now orange/right sees it)",
               drone_xy=(0, 0), drone_yaw=np.pi / 2, obstacles_xy=[(2.5, 0)],
               obst_r=0.5, room_dims=(8, 8), max_range=4.0)

    # Scenario 4: cluttered field
    np.random.seed(0)
    obstacles = [(np.random.uniform(-3, 3), np.random.uniform(-3, 3)) for _ in range(8)]
    draw_scene(axes[1, 0], "Cluttered field\n(rays should stop at nearest hit)",
               drone_xy=(0, 0), drone_yaw=np.pi / 4, obstacles_xy=obstacles,
               obst_r=0.4, room_dims=(8, 8), max_range=4.0)

    # Scenario 5: drone near corner, yaw=45 deg
    draw_scene(axes[1, 1], "Drone near corner, yaw=45deg\n(asymmetric wall distances)",
               drone_xy=(2.5, 2.5), drone_yaw=np.pi / 4, obstacles_xy=[],
               obst_r=0.3, room_dims=(8, 8), max_range=4.0)

    # Scenario 6: wall occluding distant obstacle
    draw_scene(axes[1, 2], "Wall occludes obstacle behind it\n(red sees wall, not obstacle)",
               drone_xy=(0, 0), drone_yaw=0.0, obstacles_xy=[(5, 0)],  # outside 8x8 room
               obst_r=0.5, room_dims=(8, 8), max_range=4.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    print(f"Wrote: {out_path}")
    print("Inspect this PNG visually.")


if __name__ == "__main__":
    main()
