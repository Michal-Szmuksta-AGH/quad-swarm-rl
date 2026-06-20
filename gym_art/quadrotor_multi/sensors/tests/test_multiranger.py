"""
Comprehensive validation of the Multiranger sensor implementation.
Run: python -m gym_art.quadrotor_multi.sensors.tests.test_multiranger
"""

import numpy as np

from gym_art.quadrotor_multi.sensors.multiranger import (
    get_multiranger_obs,
    _ray_cylinder_distance,
    _ray_wall_distance,
    SENSOR_BODY_ANGLES,
)


# Bitcraze convention: sensor 0=front (+x), 1=left (+y), 2=back (-x), 3=right (-y)
S_FRONT, S_LEFT, S_BACK, S_RIGHT = 0, 1, 2, 3


def _approx(a, b, tol=1e-4):
    return abs(a - b) < tol


# --------------------------------------------------------------------------
# 1. Ray-cylinder primitive
# --------------------------------------------------------------------------
def test_ray_cylinder_basic():
    """Ray from origin pointing +x hits cylinder at (2, 0, r=0.3) at d=1.7."""
    d = _ray_cylinder_distance(0, 0, 1, 0, 2.0, 0.0, 0.3, 10.0)
    assert _approx(d, 1.7), f"expected 1.7, got {d}"


def test_ray_cylinder_miss():
    """Ray points away from cylinder — no hit."""
    # Cylinder at (2, 0), ray going -x
    d = _ray_cylinder_distance(0, 0, -1, 0, 2.0, 0.0, 0.3, 10.0)
    assert _approx(d, 10.0), f"expected 10.0 (miss), got {d}"


def test_ray_cylinder_inside():
    """Ray origin INSIDE cylinder — must return 0 (drone in collision)."""
    # Drone at (1, 0), cylinder at (1, 0) with radius 0.5 → drone inside
    d = _ray_cylinder_distance(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 10.0)
    assert d == 0.0, f"inside cylinder must return 0, got {d}"


def test_ray_cylinder_on_surface():
    """Ray origin AT cylinder surface, pointing outward — distance ~0."""
    # Drone at (0.5, 0), cylinder at (0, 0) radius 0.5 → on surface
    # Ray pointing AWAY (+x): t=0 is the surface, no further hit on this circle
    d = _ray_cylinder_distance(0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 10.0)
    # On surface: c == 0, our code returns 0 (treated as inside).
    assert d == 0.0, f"surface case returns 0, got {d}"


def test_ray_cylinder_tangent_miss():
    """Ray that just barely misses the cylinder (tangent line outside)."""
    # Cylinder at (2, 1), r=0.3. Ray from (0,0) along +x: closest point on ray is (2,0),
    # distance from cyl center is 1.0 > 0.3, so miss.
    d = _ray_cylinder_distance(0, 0, 1, 0, 2.0, 1.0, 0.3, 10.0)
    assert _approx(d, 10.0), f"tangent miss expected 10.0, got {d}"


def test_ray_cylinder_far_clipping():
    """Cylinder farther than max_dist — return max_dist."""
    d = _ray_cylinder_distance(0, 0, 1, 0, 100.0, 0.0, 0.3, 4.0)
    assert _approx(d, 4.0), f"far cylinder should clip to max_dist, got {d}"


def test_ray_cylinder_diagonal():
    """Ray pointing diagonal NE, cylinder in NE direction."""
    # 45 deg ray, cylinder at (sqrt(2), sqrt(2)) ~ (1.414, 1.414), r=0.3
    # Distance to cylinder CENTER along the ray = 2.0
    # Distance to SURFACE = 2.0 - 0.3 = 1.7
    s = np.sqrt(2) / 2
    d = _ray_cylinder_distance(0, 0, s, s, 1.41421356, 1.41421356, 0.3, 10.0)
    assert _approx(d, 1.7, tol=1e-3), f"diagonal: expected 1.7, got {d}"


# --------------------------------------------------------------------------
# 2. Ray-wall primitive
# --------------------------------------------------------------------------
def test_ray_wall_axis_aligned():
    """Drone at origin, room 10x10 → wall 5m in each direction."""
    # +x wall
    d = _ray_wall_distance(0, 0, 1, 0, 5.0, 5.0, 100.0)
    assert _approx(d, 5.0), f"+x wall: expected 5.0, got {d}"
    # -x wall
    d = _ray_wall_distance(0, 0, -1, 0, 5.0, 5.0, 100.0)
    assert _approx(d, 5.0), f"-x wall: expected 5.0, got {d}"
    # +y wall
    d = _ray_wall_distance(0, 0, 0, 1, 5.0, 5.0, 100.0)
    assert _approx(d, 5.0), f"+y wall: expected 5.0, got {d}"


def test_ray_wall_off_center():
    """Drone off-center → asymmetric distances to walls."""
    # Drone at (2, 0) in 10x10 room → +x wall is 3m away, -x is 7m
    d = _ray_wall_distance(2, 0, 1, 0, 5.0, 5.0, 100.0)
    assert _approx(d, 3.0), f"+x: expected 3.0, got {d}"
    d = _ray_wall_distance(2, 0, -1, 0, 5.0, 5.0, 100.0)
    assert _approx(d, 7.0), f"-x: expected 7.0, got {d}"


def test_ray_wall_diagonal_takes_min():
    """Ray going diagonally NE — should hit nearest wall (top OR right, whichever first)."""
    # Drone at (0, 0), going 45 deg, room 10x10 (walls at +-5)
    # +x wall hit: t = 5/cos(45) = 7.07
    # +y wall hit: t = 5/sin(45) = 7.07
    # min = 7.07
    s = np.sqrt(2) / 2
    d = _ray_wall_distance(0, 0, s, s, 5.0, 5.0, 100.0)
    assert _approx(d, 5 / s, tol=1e-3), f"diag: expected 7.07, got {d}"


# --------------------------------------------------------------------------
# 3. Body-frame convention (the most error-prone part)
# --------------------------------------------------------------------------
def test_body_frame_yaw_zero():
    """At yaw=0, drone is aligned with world axes. Obstacle at world +x hits FRONT sensor."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)  # world +x at d=2
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    # Front sensor (S_FRONT=0) should see obstacle, others should see only walls (10m, clipped to 4)
    assert _approx(obs[0, S_FRONT], 1.7, tol=0.01), f"front: expected 1.7, got {obs[0, S_FRONT]}"
    assert _approx(obs[0, S_LEFT], 4.0), f"left should see far wall, got {obs[0, S_LEFT]}"
    assert _approx(obs[0, S_BACK], 4.0), f"back should see far wall, got {obs[0, S_BACK]}"
    assert _approx(obs[0, S_RIGHT], 4.0), f"right should see far wall, got {obs[0, S_RIGHT]}"


def test_body_frame_yaw_90():
    """At yaw=90deg (drone facing +y world), world +x is on drone's RIGHT (-y body)."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([np.pi / 2], dtype=np.float32)  # face +y world
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)  # at world +x
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    # Drone facing +y world. World +x is to drone's right (-y body, sensor index 3).
    assert _approx(obs[0, S_RIGHT], 1.7, tol=0.01), \
        f"right (sensor 3) should see obst at yaw 90: got {obs[0, S_RIGHT]}, full={obs[0]}"
    # Front and back should NOT see this obstacle
    assert _approx(obs[0, S_FRONT], 4.0), f"front sees wall: got {obs[0, S_FRONT]}"


def test_body_frame_yaw_180():
    """yaw=180 → drone faces -x world, so world +x obstacle is BEHIND drone."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([np.pi], dtype=np.float32)
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert _approx(obs[0, S_BACK], 1.7, tol=0.01), \
        f"back should see obstacle at yaw 180: got {obs[0, S_BACK]}, full={obs[0]}"


def test_body_frame_yaw_minus_90():
    """yaw=-90 (or 270) → drone faces -y world, world +x is to drone's LEFT (+y body)."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([-np.pi / 2], dtype=np.float32)
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert _approx(obs[0, S_LEFT], 1.7, tol=0.01), \
        f"left should see obstacle at yaw -90: got {obs[0, S_LEFT]}, full={obs[0]}"


# --------------------------------------------------------------------------
# 4. Multi-obstacle: sensor sees NEAREST
# --------------------------------------------------------------------------
def test_multi_obstacle_nearest_wins():
    """Two obstacles in front, sensor reports nearer one."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.array([
        [3.0, 0.0],   # farther
        [1.0, 0.0],   # nearer
    ], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert _approx(obs[0, S_FRONT], 0.7, tol=0.01), \
        f"front: expected 0.7 (nearer obst), got {obs[0, S_FRONT]}"


def test_multi_drone_independent():
    """Two drones at different positions get different sensor readings.

    Note: with 8-ray cone (no ray exactly on axis), expected distance is the
    nearest-ray hit, which is slightly longer than the axis-ray hit (~2.72 vs
    2.70 for an axis-aligned obstacle).
    """
    quad_xy = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0, 0.0], dtype=np.float32)
    obst_xy = np.array([[3.0, 0.0]], dtype=np.float32)  # closer to drone 0
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert _approx(obs[0, S_FRONT], 2.7, tol=0.05), f"drone 0 front: {obs[0, S_FRONT]}"
    assert _approx(obs[1, S_BACK], 1.7, tol=0.05), f"drone 1 back: {obs[1, S_BACK]}"


# --------------------------------------------------------------------------
# 5. Wall + obstacle interaction
# --------------------------------------------------------------------------
def test_wall_blocks_distant_obstacle():
    """Obstacle outside room shouldn't be reachable through wall."""
    # Room 4x4 (walls at +-2), drone at center, obstacle at (3, 0) — outside room!
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.array([[3.0, 0.0]], dtype=np.float32)
    room = np.array([4.0, 4.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    # Wall at +x is 2m away. Obstacle is 3m away (with surface at 2.7m).
    # The ray hits the WALL first at t=2.0, then would hit obstacle at t=2.7.
    # Our impl takes MIN — so front should see wall (2.0), not obstacle.
    assert _approx(obs[0, S_FRONT], 2.0, tol=0.01), \
        f"wall must occlude obstacle behind it: expected 2.0, got {obs[0, S_FRONT]}"


# --------------------------------------------------------------------------
# 6. Noise
# --------------------------------------------------------------------------
def test_noise_changes_output():
    """With noise > 0, repeated calls should yield slightly different outputs."""
    np.random.seed(42)
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs1 = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.05, 0.471, 8)
    obs2 = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.05, 0.471, 8)
    assert not np.allclose(obs1, obs2), "noise should produce different outputs"
    # And values should still be in valid range
    assert (obs1 >= 0).all() and (obs1 <= 4.0).all(), f"out-of-range: {obs1}"


def test_no_noise_is_deterministic():
    """With noise=0, repeated calls are identical."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs1 = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    obs2 = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert np.allclose(obs1, obs2)


# --------------------------------------------------------------------------
# 7. Empty obstacles
# --------------------------------------------------------------------------
def test_empty_obstacle_array():
    """Zero obstacles: sensor sees only walls."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.zeros((0, 2), dtype=np.float32)
    room = np.array([4.0, 4.0], dtype=np.float32)  # 2m to wall in each direction
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert np.allclose(obs[0], 2.0, atol=0.01), f"expected all 2.0, got {obs[0]}"


# --------------------------------------------------------------------------
# 8. Statistical sanity: no NaNs, no negatives, all in range
# --------------------------------------------------------------------------
def test_statistical_sanity():
    """Over many random configs, output always in [0, max_range], no NaN/Inf."""
    np.random.seed(0)
    room = np.array([10.0, 10.0], dtype=np.float32)
    max_range = 4.0
    n_trials = 200
    for trial in range(n_trials):
        n_quads = np.random.randint(1, 16)
        n_obst = np.random.randint(0, 30)
        # Spawn in 8x8 area, drones can be anywhere
        quad_xy = (np.random.rand(n_quads, 2).astype(np.float32) - 0.5) * 6
        yaws = (np.random.rand(n_quads).astype(np.float32) - 0.5) * 2 * np.pi
        obst_xy = (np.random.rand(n_obst, 2).astype(np.float32) - 0.5) * 6 if n_obst > 0 \
            else np.zeros((0, 2), dtype=np.float32)
        obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, max_range, 0.0, 0.471, 8)
        assert obs.shape == (n_quads, 4), f"trial {trial}: shape {obs.shape}"
        assert not np.isnan(obs).any(), f"trial {trial}: NaN found"
        assert not np.isinf(obs).any(), f"trial {trial}: Inf found"
        assert (obs >= 0).all(), f"trial {trial}: negative: {obs.min()}"
        assert (obs <= max_range + 1e-5).all(), f"trial {trial}: out of range: {obs.max()}"


# --------------------------------------------------------------------------
# 9. Comparison to SDF: when far from obstacles, sensors should also report far
# --------------------------------------------------------------------------
def test_consistency_with_sdf():
    """Drone in empty area: Multiranger and SDF agree no obstacle is near."""
    from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs

    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    # One obstacle far away
    obst_xy = np.array([[5.0, 5.0]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    mr = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    # SDF
    sdf = 100 * np.ones((1, 9))
    sdf = get_surround_sdfs(quad_xy, obst_xy, sdf, obst_radius=0.3,
                            resolution=0.1, sensor_range=10.0)
    # Both should agree obstacle is far. Multiranger may saturate at max_range.
    # SDF center cell measures dist from drone to obstacle SURFACE
    # SDF center = sqrt(50) - 0.3 ~ 6.77
    assert sdf[0, 4] > 5.0, f"SDF should report far obstacle: {sdf[0, 4]}"
    # Multiranger front/left/right/back point AWAY from obstacle direction (NE),
    # so they should all hit max_range (no obstacle in line of sight).
    assert (mr[0] >= 3.9).all(), f"multiranger should saturate: {mr[0]}"


# --------------------------------------------------------------------------
# 10. Cone-vs-single-ray (NEW: tests the FOV upgrade)
# --------------------------------------------------------------------------
def test_cone_catches_off_axis_obstacle():
    """Obstacle 11deg off-axis at 2m: single ray MISSES, cone CATCHES.

    Math:
      - Axis ray's closest approach to obstacle center = 2*sin(11deg) ~= 0.382m
      - That's > 0.3m radius, so axis ray misses by ~8cm
      - Cone half-angle = 13.5deg > 11deg, so cone catches it
    """
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    ang_off = np.deg2rad(11.0)
    obst_xy = np.array([[2.0 * np.cos(ang_off), 2.0 * np.sin(ang_off)]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)

    # Single ray (num_rays=1) misses
    obs_axis = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 1)
    assert obs_axis[0, S_FRONT] >= 3.9, \
        f"single ray must MISS off-axis obstacle: got {obs_axis[0, S_FRONT]}"

    # Cone (num_rays=8) catches
    obs_cone = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    assert obs_cone[0, S_FRONT] < 2.5, \
        f"cone must CATCH off-axis obstacle: got {obs_cone[0, S_FRONT]}"


def test_cone_ignores_far_outside():
    """Obstacle 40deg off-axis: well outside 13.5deg cone — sensor must ignore it."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    ang_off = np.deg2rad(40.0)
    obst_xy = np.array([[2.0 * np.cos(ang_off), 2.0 * np.sin(ang_off)]], dtype=np.float32)
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 8)
    # Front cone goes from -13.5 to +13.5 deg, obstacle is at 40deg — well outside.
    # Left cone goes from 90-13.5 to 90+13.5 deg = 76.5 to 103.5 deg, obstacle at 40 — outside.
    assert obs[0, S_FRONT] >= 3.9, f"front shouldn't see: got {obs[0, S_FRONT]}"
    assert obs[0, S_LEFT] >= 3.9, f"left shouldn't see: got {obs[0, S_LEFT]}"


def test_num_rays_one_equals_old_single_ray_behavior():
    """num_rays=1 should give exactly single-axis-ray semantics (backward compat)."""
    quad_xy = np.array([[0.0, 0.0]], dtype=np.float32)
    yaws = np.array([0.0], dtype=np.float32)
    obst_xy = np.array([[2.0, 0.0]], dtype=np.float32)  # directly on axis
    room = np.array([20.0, 20.0], dtype=np.float32)
    obs = get_multiranger_obs(quad_xy, yaws, obst_xy, 0.3, room, 4.0, 0.0, 0.471, 1)
    assert _approx(obs[0, S_FRONT], 1.7, tol=0.001), \
        f"num_rays=1 axis hit: expected 1.7, got {obs[0, S_FRONT]}"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
TESTS = [
    test_ray_cylinder_basic,
    test_ray_cylinder_miss,
    test_ray_cylinder_inside,
    test_ray_cylinder_on_surface,
    test_ray_cylinder_tangent_miss,
    test_ray_cylinder_far_clipping,
    test_ray_cylinder_diagonal,
    test_ray_wall_axis_aligned,
    test_ray_wall_off_center,
    test_ray_wall_diagonal_takes_min,
    test_body_frame_yaw_zero,
    test_body_frame_yaw_90,
    test_body_frame_yaw_180,
    test_body_frame_yaw_minus_90,
    test_multi_obstacle_nearest_wins,
    test_multi_drone_independent,
    test_wall_blocks_distant_obstacle,
    test_noise_changes_output,
    test_no_noise_is_deterministic,
    test_empty_obstacle_array,
    test_statistical_sanity,
    test_consistency_with_sdf,
    test_cone_catches_off_axis_obstacle,
    test_cone_ignores_far_outside,
    test_num_rays_one_equals_old_single_ray_behavior,
]


def run_all():
    passed = 0
    for t in TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(TESTS)} tests passed")
    return passed == len(TESTS)


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)
