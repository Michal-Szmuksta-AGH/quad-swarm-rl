"""
Unit tests dla topology generators.
Run: python -m gym_art.quadrotor_multi.topology.tests.test_generators
"""

import numpy as np

from gym_art.quadrotor_multi.topology.generators import (
    grid_topology, poisson_topology, cluster_topology, building_topology,
    generate_topology, TOPOLOGY_GENERATORS,
    _positions_to_obst_map, _min_dist_for,
    _compute_bare_obst_map, is_free_space_connected, label_components,
    _split_convex_polygon, _polygon_diameter,
)


SPAWN = np.array([8.0, 8.0], dtype=np.float32)
DENSITY = 0.2      # ~13 obstacles in 8x8
OBST_SIZE = 0.6    # radius 0.3, min_dist = 0.72
ROOM_H = 10.0


def _check_no_overlap(positions, obst_size, tol=0.001):
    min_dist = obst_size - tol
    for i, (xi, yi) in enumerate(positions):
        for j in range(i + 1, len(positions)):
            xj, yj = positions[j]
            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            assert d >= min_dist, \
                f"Overlap: obst {i}=({xi:.2f},{yi:.2f}) and {j}=({xj:.2f},{yj:.2f}), d={d:.3f} < {min_dist}"


def _check_in_bounds(positions, spawn_area):
    hx, hy = spawn_area[0] / 2, spawn_area[1] / 2
    for x, y in positions:
        assert -hx <= x <= hx, f"x={x} out of [-{hx}, {hx}]"
        assert -hy <= y <= hy, f"y={y} out of [-{hy}, {hy}]"


def _check_obst_map_consistent(positions, obst_map, spawn_area):
    L = int(spawn_area[0])
    W = int(spawn_area[1])
    for x, y in positions:
        rid = int((x + L / 2) / 1.0)
        cid = int((y + W / 2) / 1.0)
        rid = max(0, min(L - 1, rid))
        cid = max(0, min(W - 1, cid))
        assert obst_map[rid, cid] == 1, f"Obst at ({x:.2f},{y:.2f}) -> cell ({rid},{cid}) not marked"


def _positions_only(obst_pos_arr):
    return [(p[0], p[1]) for p in obst_pos_arr]


def _no_identical(positions):
    """Zaden pair obstacle nie ma zero distance."""
    for i, (xi, yi) in enumerate(positions):
        for j in range(i + 1, len(positions)):
            xj, yj = positions[j]
            d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            assert d > 0.01, f"identical obstacles at {i}={positions[i]} and {j}={positions[j]}"


# --------------------------------------------------------------------------
# Grid tests
# --------------------------------------------------------------------------
def test_grid_correct_count():
    rng = np.random.RandomState(0)
    obst_map, pos_arr, cells = grid_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    expected = int(SPAWN[0] * SPAWN[1] * DENSITY)
    assert len(pos_arr) == expected, f"grid: expected {expected}, got {len(pos_arr)}"


def test_grid_positions_at_cell_centers():
    rng = np.random.RandomState(1)
    _, pos_arr, _ = grid_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    for x, y, z in pos_arr:
        frac_x = abs(x) - int(abs(x))
        frac_y = abs(y) - int(abs(y))
        assert abs(frac_x - 0.5) < 0.01, f"x={x} not at cell center"
        assert abs(frac_y - 0.5) < 0.01, f"y={y} not at cell center"


def test_grid_z_is_room_center():
    rng = np.random.RandomState(2)
    _, pos_arr, _ = grid_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    for _, _, z in pos_arr:
        assert abs(z - ROOM_H / 2) < 1e-6


# --------------------------------------------------------------------------
# Poisson tests
# --------------------------------------------------------------------------
def test_poisson_no_overlap():
    rng = np.random.RandomState(0)
    _, pos_arr, _ = poisson_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    _check_no_overlap(_positions_only(pos_arr), OBST_SIZE)


def test_poisson_in_bounds():
    rng = np.random.RandomState(1)
    _, pos_arr, _ = poisson_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    _check_in_bounds(_positions_only(pos_arr), SPAWN)


def test_poisson_reasonable_count():
    rng = np.random.RandomState(2)
    _, pos_arr, _ = poisson_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    expected = int(SPAWN[0] * SPAWN[1] * DENSITY)
    assert len(pos_arr) >= 0.8 * expected


def test_poisson_obst_map_consistent():
    rng = np.random.RandomState(3)
    obst_map, pos_arr, _ = poisson_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    _check_obst_map_consistent(_positions_only(pos_arr), obst_map, SPAWN)


# --------------------------------------------------------------------------
# Cluster tests
# --------------------------------------------------------------------------
def test_cluster_no_overlap():
    rng = np.random.RandomState(0)
    _, pos_arr, _ = cluster_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    _check_no_overlap(_positions_only(pos_arr), OBST_SIZE)


def test_cluster_positions_clustered():
    rng = np.random.RandomState(0)
    _, cluster_pos_arr, _ = cluster_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H,
                                              n_clusters=3, cluster_std=0.8, rng=rng)
    _, poisson_pos_arr, _ = poisson_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)

    def avg_nearest(pos_arr):
        pos = np.array([(p[0], p[1]) for p in pos_arr])
        dists = []
        for i in range(len(pos)):
            others = np.delete(pos, i, axis=0)
            d = np.min(np.linalg.norm(others - pos[i], axis=1))
            dists.append(d)
        return np.mean(dists)

    cluster_nn = avg_nearest(cluster_pos_arr)
    poisson_nn = avg_nearest(poisson_pos_arr)
    assert cluster_nn < poisson_nn


# --------------------------------------------------------------------------
# Polygon split helpers
# --------------------------------------------------------------------------
def test_split_convex_polygon_square_horizontal():
    """Poziomy split kwadratu daje dwa prostokaty i dwa punkty przeciecia."""
    square = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    # Line: y = 0 (horizontal), normal = (0, 1), point = (0, 0)
    pos, neg, inters = _split_convex_polygon(square, (0, 0), (0, 1))
    assert len(inters) == 2, f"Expected 2 intersections, got {len(inters)}"
    # Positive side = upper half (y >= 0), negative = lower
    assert all(y >= -0.01 for x, y in pos), f"positive side has y<0: {pos}"
    assert all(y <= 0.01 for x, y in neg), f"negative side has y>0: {neg}"


def test_split_convex_polygon_diagonal():
    """Diagonalny split (theta=45°) kwadratu daje dwa trojkatne sub-polygony.

    Line_p ma maly offset od diagonali zeby uniknac edge case gdzie linia
    przechodzi dokladnie przez wierzcholki (a wtedy intersections=0 bo
    d_curr*d_next nigdy nie zmienia znaku).
    """
    square = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    normal = (-np.sqrt(2) / 2, np.sqrt(2) / 2)
    # Offset lekko od diagonali (unika trafienia w wierzcholki)
    pos, neg, inters = _split_convex_polygon(square, (0.3, 0.3), normal)
    assert len(inters) == 2, f"Expected 2 intersections, got {len(inters)}: {inters}"
    assert len(pos) >= 3 and len(neg) >= 3


def test_polygon_diameter_square():
    square = [(-2, -2), (2, -2), (2, 2), (-2, 2)]
    d = _polygon_diameter(square)
    # Diagonal of 4x4 square = 4*sqrt(2) ≈ 5.657
    assert abs(d - 4 * np.sqrt(2)) < 0.01


# --------------------------------------------------------------------------
# Building tests
# --------------------------------------------------------------------------
def test_building_no_identical():
    rng = np.random.RandomState(0)
    _, pos_arr, _ = building_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    _no_identical(_positions_only(pos_arr))


def test_building_in_bounds():
    for seed in range(10):
        rng = np.random.RandomState(seed)
        _, pos_arr, _ = building_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
        _check_in_bounds(_positions_only(pos_arr), SPAWN)


def test_building_depth_scales_with_density():
    """Wyzsza density -> wiecej scian (wiecej podzialow BSP)."""
    counts_low = []
    counts_high = []
    for seed in range(10):
        rng_lo = np.random.RandomState(seed)
        rng_hi = np.random.RandomState(seed + 100)
        _, pos_lo, _ = building_topology(SPAWN, 0.10, OBST_SIZE, ROOM_H, rng=rng_lo)
        _, pos_hi, _ = building_topology(SPAWN, 0.35, OBST_SIZE, ROOM_H, rng=rng_hi)
        counts_low.append(len(pos_lo))
        counts_high.append(len(pos_hi))
    assert np.mean(counts_high) > np.mean(counts_low), \
        f"BSP depth->density: high mean {np.mean(counts_high):.1f} not > low {np.mean(counts_low):.1f}"


def test_building_walls_not_all_axis_aligned():
    """KLUCZOWY test: ściany budynku NIE są wszystkie tylko pionowe/poziome.

    Zbiera positions z 30 seedów, dopasowuje linie do collinear grup (min 4 pts),
    liczy kąty. Oczekuje że >20% linii ma kąt NIE w {0°, 90°} (allowing ±10°
    tolerance).
    """
    all_angles = []
    for seed in range(30):
        rng = np.random.RandomState(seed)
        _, pos_arr, _ = building_topology(SPAWN, 0.25, OBST_SIZE, ROOM_H,
                                           max_depth=3, rng=rng)
        positions = np.array(_positions_only(pos_arr))
        if len(positions) < 4:
            continue

        # For each pair of nearby points (< 1m apart), record angle
        # Then filter to only points where local direction repeats (part of a wall)
        # Simpler: use pairwise angles of neighbor pairs
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dp = positions[j] - positions[i]
                d = np.linalg.norm(dp)
                if 0.4 < d < 0.7:  # neighbor distance ~= wall_spacing
                    angle = np.degrees(np.arctan2(dp[1], dp[0])) % 180
                    all_angles.append(angle)

    if not all_angles:
        return  # too few obstacles to conclude
    # Fraction of angles NOT close to 0° or 90° (within ±10°)
    non_axis = sum(1 for a in all_angles
                    if not (a < 10 or a > 170 or (80 < a < 100)))
    frac = non_axis / len(all_angles)
    assert frac > 0.20, \
        f"Building walls too axis-aligned: only {frac:.1%} non-axis (out of {len(all_angles)} pairs)"


def test_building_connectivity_via_retry_loop():
    """Building via generate_topology (z retry) osiaga wysoki success rate przy
    operacyjnym density=0.2.

    Random-angle BSP naturalnie generuje trudne layouty (drzwi ustawione względem
    następnych ścian), więc one-shot connectivity rate to ~43%. Retry loop (max 10
    prób) naprawia to do 98%+ przy density=0.2.
    """
    success_count = 0
    N = 30
    for seed in range(N):
        rng = np.random.RandomState(seed)
        _, _, _, stats = generate_topology(
            'building', SPAWN, 0.20, OBST_SIZE, ROOM_H,
            rng=rng, return_stats=True, max_retries=10)
        if stats['fallback'] == 0:
            success_count += 1
    # Retry loop should get > 85% success at density=0.2
    assert success_count >= int(0.85 * N), \
        f"Building via retry loop: only {success_count}/{N} succeeded"


def test_building_multiple_walls_present():
    """Building powinien generowac wiele scian (nie tylko jedna)."""
    rng = np.random.RandomState(0)
    _, pos_arr, _ = building_topology(SPAWN, 0.25, OBST_SIZE, ROOM_H,
                                       max_depth=3, rng=rng)
    positions = _positions_only(pos_arr)
    assert len(positions) > 10, \
        f"Building at depth=3 should have >10 obstacles, got {len(positions)}"


# --------------------------------------------------------------------------
# Dispatch tests
# --------------------------------------------------------------------------
def test_dispatch_all_topologies():
    rng = np.random.RandomState(42)
    for name in TOPOLOGY_GENERATORS.keys():
        obst_map, pos_arr, cells = generate_topology(
            name, SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
        assert obst_map.shape == (int(SPAWN[0]), int(SPAWN[1])), \
            f"{name}: obst_map shape {obst_map.shape}"
        assert len(pos_arr) > 0, f"{name}: no obstacles generated"
        assert cells.shape == (int(SPAWN[0]) * int(SPAWN[1]), 2), \
            f"{name}: cells shape {cells.shape}"
        _check_in_bounds(_positions_only(pos_arr), SPAWN)


def test_dispatch_only_four_topologies():
    """Po v3 rewrite tylko 4 topologie zostaly."""
    assert set(TOPOLOGY_GENERATORS.keys()) == {'grid', 'poisson', 'cluster', 'building'}, \
        f"Expected exactly 4 topologies, got {list(TOPOLOGY_GENERATORS.keys())}"


def test_dispatch_invalid_raises():
    try:
        generate_topology('nonexistent', SPAWN, DENSITY, OBST_SIZE, ROOM_H)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'nonexistent' in str(e)


def test_dispatch_old_names_removed():
    """Stare nazwy corridor/l_corridor NIE powinny juz istniec w dispatch."""
    for old_name in ['corridor', 'l_corridor']:
        try:
            generate_topology(old_name, SPAWN, DENSITY, OBST_SIZE, ROOM_H)
            assert False, f"Old topology '{old_name}' should have been removed"
        except ValueError:
            pass  # expected


# --------------------------------------------------------------------------
# Connectivity (BFS)
# --------------------------------------------------------------------------
def test_connectivity_empty_map():
    m = np.zeros((8, 8), dtype=np.int8)
    assert is_free_space_connected(m)
    _, n = label_components(m)
    assert n == 1


def test_connectivity_all_occupied():
    m = np.ones((8, 8), dtype=np.int8)
    assert not is_free_space_connected(m)


def test_connectivity_two_islands():
    m = np.ones((5, 5), dtype=np.int8)
    m[0, 0] = 0
    m[4, 4] = 0
    assert not is_free_space_connected(m)
    _, n = label_components(m)
    assert n == 2


def test_connectivity_one_component_L_shape():
    m = np.ones((5, 5), dtype=np.int8)
    m[:3, 0] = 0
    m[2, :3] = 0
    assert is_free_space_connected(m)


def test_connectivity_wall_dividing_room():
    m = np.zeros((5, 5), dtype=np.int8)
    m[2, :] = 1
    assert not is_free_space_connected(m)
    _, n = label_components(m)
    assert n == 2


def test_connectivity_wall_with_gap():
    m = np.zeros((5, 5), dtype=np.int8)
    m[2, :] = 1
    m[2, 2] = 0
    assert is_free_space_connected(m)


# --------------------------------------------------------------------------
# Bare obst_map (non-dilated)
# --------------------------------------------------------------------------
def test_bare_map_only_marks_containing_cells():
    positions = [(0.5, 0.5), (3.5, 2.5)]
    bare = _compute_bare_obst_map(positions, SPAWN)
    assert bare.sum() == 2
    assert bare[4, 4] == 1
    assert bare[7, 6] == 1


def test_bare_map_less_than_dilated():
    rng = np.random.RandomState(0)
    _, pos_arr, _ = poisson_topology(SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    positions = [(p[0], p[1]) for p in pos_arr]
    bare = _compute_bare_obst_map(positions, SPAWN)
    dilated = _positions_to_obst_map(positions, SPAWN)
    assert bare.sum() <= dilated.sum()


# --------------------------------------------------------------------------
# generate_topology retry / stats
# --------------------------------------------------------------------------
def test_generate_topology_returns_stats_when_requested():
    result = generate_topology('grid', SPAWN, DENSITY, OBST_SIZE, ROOM_H,
                                rng=np.random.RandomState(0), return_stats=True)
    assert len(result) == 4
    _, _, _, stats = result
    assert 'attempts' in stats
    assert 'fallback' in stats


def test_generate_topology_no_stats_by_default():
    result = generate_topology('grid', SPAWN, DENSITY, OBST_SIZE, ROOM_H,
                                rng=np.random.RandomState(0))
    assert len(result) == 3


def test_generate_topology_grid_rarely_retries():
    attempts_list = []
    for seed in range(20):
        _, _, _, stats = generate_topology(
            'grid', SPAWN, DENSITY, OBST_SIZE, ROOM_H,
            rng=np.random.RandomState(seed), return_stats=True)
        attempts_list.append(stats['attempts'])
        assert stats['fallback'] == 0, f"grid seed={seed} hit fallback"
    mean_attempts = np.mean(attempts_list)
    assert mean_attempts < 2.5


def test_generate_topology_keep_rejected():
    _, _, _, stats = generate_topology(
        'building', SPAWN, density=0.4, obst_size=OBST_SIZE, room_height=ROOM_H,
        rng=np.random.RandomState(0), return_stats=True, keep_rejected=True,
        max_retries=5)
    if stats['attempts'] > 1:
        assert 'rejected_maps' in stats
        assert len(stats['rejected_maps']) == stats['attempts'] - 1 or stats['fallback'] == 1


def test_generate_topology_disables_connectivity_check():
    _, _, _, stats = generate_topology(
        'grid', SPAWN, DENSITY, OBST_SIZE, ROOM_H,
        rng=np.random.RandomState(0), return_stats=True,
        ensure_connectivity=False, max_retries=100)
    assert stats['attempts'] == 1


def test_generate_topology_building_dispatch():
    rng = np.random.RandomState(0)
    obst_map, pos_arr, cells = generate_topology(
        'building', SPAWN, DENSITY, OBST_SIZE, ROOM_H, rng=rng)
    assert len(pos_arr) > 0
    _check_in_bounds(_positions_only(pos_arr), SPAWN)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
TESTS = [
    # Grid
    test_grid_correct_count,
    test_grid_positions_at_cell_centers,
    test_grid_z_is_room_center,
    # Poisson
    test_poisson_no_overlap,
    test_poisson_in_bounds,
    test_poisson_reasonable_count,
    test_poisson_obst_map_consistent,
    # Cluster
    test_cluster_no_overlap,
    test_cluster_positions_clustered,
    # Polygon helpers
    test_split_convex_polygon_square_horizontal,
    test_split_convex_polygon_diagonal,
    test_polygon_diameter_square,
    # Building
    test_building_no_identical,
    test_building_in_bounds,
    test_building_depth_scales_with_density,
    test_building_walls_not_all_axis_aligned,
    test_building_connectivity_via_retry_loop,
    test_building_multiple_walls_present,
    # Dispatch
    test_dispatch_all_topologies,
    test_dispatch_only_four_topologies,
    test_dispatch_invalid_raises,
    test_dispatch_old_names_removed,
    # Connectivity
    test_connectivity_empty_map,
    test_connectivity_all_occupied,
    test_connectivity_two_islands,
    test_connectivity_one_component_L_shape,
    test_connectivity_wall_dividing_room,
    test_connectivity_wall_with_gap,
    # Bare map
    test_bare_map_only_marks_containing_cells,
    test_bare_map_less_than_dilated,
    # Dispatch stats
    test_generate_topology_returns_stats_when_requested,
    test_generate_topology_no_stats_by_default,
    test_generate_topology_grid_rarely_retries,
    test_generate_topology_keep_rejected,
    test_generate_topology_disables_connectivity_check,
    test_generate_topology_building_dispatch,
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
