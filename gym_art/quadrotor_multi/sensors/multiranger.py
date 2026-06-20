"""
Bitcraze Multiranger Deck simulation: 4 VL53L1x ToF sensors at 0deg, 90deg,
180deg, 270deg in drone body frame. Each is a single-zone sensor reporting
ONE distance — the minimum distance to any object within its FOV cone.

Hardware reference:
  - VL53L1x:        range 0.04-4m, FOV 27deg (datasheet), accuracy +-3% @ 1m
  - Multi-ranger:   https://store.bitcraze.io/products/multi-ranger-deck
  - Crazyflie 2.1:  9.2cm body, IMU-fused odometry from FlowDeck v2

Simulation modeling choices:
  - FOV cone sampled by N rays evenly spread across the 27deg field. Per sensor
    we report min(distances) across all rays — this matches the single-zone
    physical behavior of VL53L1x (the sensor integrates photons from the entire
    cone and reports the dominant reflector). N=8 is the default, matching the
    horizontal resolution used by LEARN (Chiu et al. 2025) when simulating the
    VL53L5CX. Tunable via --quads_multiranger_num_rays.
  - Wall detection included. Room is a 2D axis-aligned box: walls at +-half-dim.
    LEARN's TOF training code does NOT handle walls; we do.
  - Optional gaussian output noise (datasheet: sigma ~= 0.03 * distance at 1m).
  - 2D simulation: assumes obstacles span floor-to-ceiling (paper's assumption).
"""

import numpy as np
from numba import njit

# Sensor angles in DRONE BODY FRAME (rad).
# Bitcraze convention: +x = forward, +y = left, so:
#   sensor 0 = front  (+x body)
#   sensor 1 = left   (+y body)
#   sensor 2 = back   (-x body)
#   sensor 3 = right  (-y body)
SENSOR_BODY_ANGLES = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])


@njit(cache=True)
def _ray_cylinder_distance(ray_ox, ray_oy, ray_dx, ray_dy,
                           cyl_cx, cyl_cy, cyl_radius, max_dist):
    """
    Distance along a unit ray from its origin to the first entry point of an
    infinite-height vertical cylinder, or max_dist if the ray misses.

    Solves the quadratic |O + tD - C|^2 = r^2 for unit |D|.
    """
    ox = ray_ox - cyl_cx
    oy = ray_oy - cyl_cy
    c = ox * ox + oy * oy - cyl_radius * cyl_radius
    # If origin is INSIDE the cylinder, the drone is in collision with this
    # obstacle. A real ToF reports 0 (or saturation low) in this regime — it
    # cannot see anything because it is physically touching/embedded in the
    # obstacle. Returning 0 here gives the policy a correct "collision" signal
    # consistent with the collision-detection subsystem.
    if c <= 0.0:
        return 0.0
    b = 2.0 * (ox * ray_dx + oy * ray_dy)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return max_dist
    sqrt_disc = np.sqrt(disc)
    # Outside the cylinder: t1 (smaller root) is the entry point if positive.
    t1 = (-b - sqrt_disc) * 0.5
    if t1 > 0.0:
        return t1 if t1 < max_dist else max_dist
    # t1 <= 0 here means the cylinder is behind us along the ray — miss.
    return max_dist


@njit(cache=True)
def _ray_wall_distance(ray_ox, ray_oy, ray_dx, ray_dy,
                       half_room_x, half_room_y, max_dist):
    """First room-boundary hit along the ray. Walls at x=+-half_x, y=+-half_y."""
    min_dist = max_dist
    eps = 1e-9
    if ray_dx > eps:
        t = (half_room_x - ray_ox) / ray_dx
        if 0.0 < t < min_dist:
            min_dist = t
    elif ray_dx < -eps:
        t = (-half_room_x - ray_ox) / ray_dx
        if 0.0 < t < min_dist:
            min_dist = t
    if ray_dy > eps:
        t = (half_room_y - ray_oy) / ray_dy
        if 0.0 < t < min_dist:
            min_dist = t
    elif ray_dy < -eps:
        t = (-half_room_y - ray_oy) / ray_dy
        if 0.0 < t < min_dist:
            min_dist = t
    return min_dist


@njit(cache=True)
def get_multiranger_obs(quad_poses_xy, quad_yaws, obst_poses_xy, obst_radius,
                        room_dims_xy, max_range, noise_std, fov_rad, num_rays):
    """
    Compute Multiranger Deck observations for N drones with FOV cone sampling.

    Each sensor's reading = min distance over `num_rays` rays uniformly spread
    across its `fov_rad` field-of-view cone. This matches single-zone ToF
    physical behavior (the sensor reports the closest reflector within its FOV).

    Args:
        quad_poses_xy: (N, 2) drone x,y positions in world frame.
        quad_yaws: (N,) drone yaw angles in world frame (rad).
        obst_poses_xy: (M, 2) cylindrical obstacle centers.
        obst_radius: float, common obstacle radius (m).
        room_dims_xy: (2,) [width, depth] of the room (m), centered at origin.
        max_range: float, sensor max range (4.0 for VL53L1x).
        noise_std: float, gaussian sigma applied to output (0.0 = no noise).
        fov_rad: float, FOV cone half-angle * 2 in radians (VL53L1x ~= 0.471 rad ~= 27deg).
        num_rays: int, number of rays per sensor across the cone (>= 1).

    Returns:
        (N, 4) float32 array of distances, ordered [front, left, back, right]
        in drone body frame. Each value is clipped to [0, max_range].
    """
    n_quads = quad_poses_xy.shape[0]
    n_obst = obst_poses_xy.shape[0]
    out = np.full((n_quads, 4), max_range, dtype=np.float32)

    half_x = 0.5 * room_dims_xy[0]
    half_y = 0.5 * room_dims_xy[1]

    # Precompute ray angle offsets within the FOV cone.
    # num_rays=1  → single ray on axis (offset 0).
    # num_rays>1  → uniform spread across [-fov/2, +fov/2] inclusive.
    if num_rays <= 1:
        ray_offsets = np.zeros(1)
        n_actual = 1
    else:
        ray_offsets = np.empty(num_rays)
        step = fov_rad / (num_rays - 1)
        for r in range(num_rays):
            ray_offsets[r] = -0.5 * fov_rad + r * step
        n_actual = num_rays

    for q_id in range(n_quads):
        ox = quad_poses_xy[q_id, 0]
        oy = quad_poses_xy[q_id, 1]
        yaw = quad_yaws[q_id]
        for s_id in range(4):
            base_ang = yaw + SENSOR_BODY_ANGLES[s_id]
            min_dist = max_range
            # Sample multiple rays across the FOV cone; report MIN distance.
            for r_id in range(n_actual):
                ang = base_ang + ray_offsets[r_id]
                dx = np.cos(ang)
                dy = np.sin(ang)
                for o_id in range(n_obst):
                    d = _ray_cylinder_distance(
                        ox, oy, dx, dy,
                        obst_poses_xy[o_id, 0], obst_poses_xy[o_id, 1],
                        obst_radius, max_range)
                    if d < min_dist:
                        min_dist = d
                d_wall = _ray_wall_distance(ox, oy, dx, dy, half_x, half_y, max_range)
                if d_wall < min_dist:
                    min_dist = d_wall
            out[q_id, s_id] = min_dist

    if noise_std > 0.0:
        noise = np.random.normal(0.0, noise_std, out.shape).astype(np.float32)
        out = out + noise
        # Clip to valid sensor range
        for q_id in range(n_quads):
            for s_id in range(4):
                v = out[q_id, s_id]
                if v < 0.0:
                    out[q_id, s_id] = 0.0
                elif v > max_range:
                    out[q_id, s_id] = max_range

    return out
