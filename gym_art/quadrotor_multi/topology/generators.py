"""
Topology generators for obstacle placement.

Motywacja: paper bazowy (Huang et al. ICRA 2024) rozmieszcza przeszkody wylacznie
na regularnej siatce. Ta praca bada wplyw sposobu rozmieszczenia (topologii)
na performance polityki, testujac 4 topologie:
  - grid:     baseline (regularna siatka, jak w paperze)
  - poisson:  uniform random z minimalnym odstepem miedzy przeszkodami
  - cluster:  K klastrow gaussowsko-rozproszonych przeszkod
  - building: BSP-based multi-room building interior — sciany pod DOWOLNYMI
              katami (nie tylko 90°), drzwi miedzy pomieszczeniami. Rekursywny
              podzial convex polygons.

Wszystkie generatory zwracaja spojny interface:
  (obst_map, obst_pos_arr, cell_centers)

  obst_map      : (obst_area_length, obst_area_width) binary DILATED map
                  1 = zajete (albo zawiera przeszkode, albo w promieniu
                  safe_spawn_distance od przeszkody -> uzywane przez scenariusze
                  do BEZPIECZNEGO spawn'u dronow z gwarantowanym clearance)
  obst_pos_arr  : list of [x, y, z] world-frame positions (z = room_height/2)
  cell_centers  : (N_cells, 2) — grid cell centers

Connectivity guarantee: `generate_topology()` dispatch verifies (via BFS on
non-dilated map) ze wszystkie wolne komorki tworza JEDEN spojny komponent.
Przy niepowodzeniu regeneruje z nowym seedem, do max_retries prob.
"""

import warnings

import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_cell_centers


def _spawn_bounds(spawn_area):
    """(x_min, x_max, y_min, y_max) — spawn area centered at origin."""
    hx = spawn_area[0] / 2.0
    hy = spawn_area[1] / 2.0
    return -hx, hx, -hy, hy


def _positions_to_obst_map(positions, spawn_area, grid_size=1.0,
                           safe_spawn_distance=0.65):
    """Rasterize (x, y) positions into binary occupancy grid — z dilation.

    Oznacza jako 'zajete' NIE TYLKO komorki zawierajace przeszkode, ale rowniez
    komorki ktorych srodek jest w odleglosci mniejszej niz safe_spawn_distance
    od jakiejkolwiek przeszkody. Dzieki temu scenariusze spawnujace drony w
    'free' komorkach (obst_map == 0) maja gwarantowany min. clearance
    (~safe_spawn_distance - obst_radius - drone_radius).

    Default safe_spawn_distance=0.65:
      dla obst_radius=0.3, drone_radius=0.046 -> min clearance ~0.3m
    """
    obst_area_length = int(spawn_area[0])
    obst_area_width = int(spawn_area[1])
    obst_map = np.zeros((obst_area_length, obst_area_width), dtype=np.float32)
    safe_dist_sq = safe_spawn_distance ** 2
    search_radius = int(np.ceil(safe_spawn_distance / grid_size)) + 1
    for x, y in positions:
        rid_center = int((x + obst_area_length / 2.0) / grid_size)
        cid_center = int((y + obst_area_width / 2.0) / grid_size)
        for drid in range(-search_radius, search_radius + 1):
            for dcid in range(-search_radius, search_radius + 1):
                rid = rid_center + drid
                cid = cid_center + dcid
                if not (0 <= rid < obst_area_length and 0 <= cid < obst_area_width):
                    continue
                cell_x = rid * grid_size + grid_size / 2.0 - obst_area_length / 2.0
                cell_y = cid * grid_size + grid_size / 2.0 - obst_area_width / 2.0
                if (cell_x - x) ** 2 + (cell_y - y) ** 2 < safe_dist_sq:
                    obst_map[rid, cid] = 1
    return obst_map


def _min_dist_for(obst_size):
    """Minimum center-to-center distance between obstacles (no overlap + margin)."""
    return obst_size * 1.2


# --------------------------------------------------------------------------
# 1. GRID (baseline — same as paper)
# --------------------------------------------------------------------------
def grid_topology(spawn_area, density, obst_size, room_height, grid_size=1.0, rng=None):
    """Regular grid — baseline replika logiki z quadrotor_multi.py.

    UWAGA (fix 2026-08-22): oryginalna implementacja Huanga mieszala dwa mappingi
    cell -> world coords. obst_map[rid, cid] uzywalo (rid, cid) = (x, y) indexing,
    ale obstacle position brany z cell_centers[rid + L*cid], podczas gdy
    cell_centers iteruje inner-y w REVERSED order — wiec obstacle byl umieszczany
    w cell RÓZNEJ od tej oznaczonej w obst_map. Tu uzywamy bezposredniego
    (rid, cid) -> world coord mapping ktore jest spojne z obst_map indexowaniem
    (i z fixed o_base.py generate_pos_obst_map*).
    """
    if rng is None:
        rng = np.random
    obst_area_length = int(spawn_area[0])
    obst_area_width = int(spawn_area[1])
    num_room_grids = obst_area_length * obst_area_width
    cell_centers = get_cell_centers(obst_area_length, obst_area_width, grid_size)

    n_obst = int(num_room_grids * density)
    obst_index = rng.choice(num_room_grids, size=n_obst, replace=False)

    obst_map = np.zeros((obst_area_length, obst_area_width), dtype=np.float32)
    obst_pos_arr = []
    for obst_id in obst_index:
        rid = obst_id // obst_area_width
        cid = obst_id - rid * obst_area_width
        obst_map[rid, cid] = 1
        # Bezposredni mapping (rid, cid) -> world coord — spojny z obst_map indexowaniem
        pos_x = rid + 0.5 - obst_area_length // 2
        pos_y = cid + 0.5 - obst_area_width // 2
        obst_pos_arr.append([pos_x, pos_y, room_height / 2.0])
    return obst_map, obst_pos_arr, cell_centers


# --------------------------------------------------------------------------
# 2. POISSON DISK SAMPLING (uniform random, minimum spacing)
# --------------------------------------------------------------------------
def poisson_topology(spawn_area, density, obst_size, room_height, grid_size=1.0,
                     rng=None, max_total_attempts=None):
    """Poisson disk sampling — uniform random rozmieszczenie z minimalnym odstepem."""
    if rng is None:
        rng = np.random
    x_min, x_max, y_min, y_max = _spawn_bounds(spawn_area)
    obst_area_length = int(spawn_area[0])
    obst_area_width = int(spawn_area[1])
    num_room_grids = obst_area_length * obst_area_width
    n_target = int(num_room_grids * density)

    min_dist = _min_dist_for(obst_size)
    min_dist_sq = min_dist ** 2

    if max_total_attempts is None:
        max_total_attempts = n_target * 200

    positions = []
    attempts = 0
    while len(positions) < n_target and attempts < max_total_attempts:
        cx = rng.uniform(x_min, x_max)
        cy = rng.uniform(y_min, y_max)
        ok = True
        for px, py in positions:
            if (cx - px) ** 2 + (cy - py) ** 2 < min_dist_sq:
                ok = False
                break
        if ok:
            positions.append((cx, cy))
        attempts += 1

    obst_pos_arr = [[x, y, room_height / 2.0] for x, y in positions]
    obst_map = _positions_to_obst_map(positions, spawn_area, grid_size)
    cell_centers = get_cell_centers(obst_area_length, obst_area_width, grid_size)
    return obst_map, obst_pos_arr, cell_centers


# --------------------------------------------------------------------------
# 3. CLUSTER (K-means style — grupy przeszkod)
# --------------------------------------------------------------------------
def cluster_topology(spawn_area, density, obst_size, room_height, grid_size=1.0,
                     n_clusters=3, cluster_std=1.0, rng=None):
    """K klastrow, kazdy z ~n_target/K przeszkod rozproszonych gaussowsko."""
    if rng is None:
        rng = np.random
    x_min, x_max, y_min, y_max = _spawn_bounds(spawn_area)
    obst_area_length = int(spawn_area[0])
    obst_area_width = int(spawn_area[1])
    num_room_grids = obst_area_length * obst_area_width
    n_target = int(num_room_grids * density)

    margin = 1.5 * cluster_std
    cluster_centers = np.column_stack([
        rng.uniform(x_min + margin, x_max - margin, n_clusters),
        rng.uniform(y_min + margin, y_max - margin, n_clusters)
    ])

    obst_per_cluster = n_target // n_clusters
    remainder = n_target - obst_per_cluster * n_clusters

    min_dist_sq = _min_dist_for(obst_size) ** 2
    positions = []

    for i, (cx, cy) in enumerate(cluster_centers):
        target_here = obst_per_cluster + (1 if i < remainder else 0)
        placed = 0
        attempts = 0
        max_attempts = target_here * 100
        while placed < target_here and attempts < max_attempts:
            x = rng.normal(cx, cluster_std)
            y = rng.normal(cy, cluster_std)
            x = max(x_min, min(x_max, x))
            y = max(y_min, min(y_max, y))
            ok = True
            for px, py in positions:
                if (x - px) ** 2 + (y - py) ** 2 < min_dist_sq:
                    ok = False
                    break
            if ok:
                positions.append((x, y))
                placed += 1
            attempts += 1

    obst_pos_arr = [[x, y, room_height / 2.0] for x, y in positions]
    obst_map = _positions_to_obst_map(positions, spawn_area, grid_size)
    cell_centers = get_cell_centers(obst_area_length, obst_area_width, grid_size)
    return obst_map, obst_pos_arr, cell_centers


# --------------------------------------------------------------------------
# 4. BUILDING (BSP na convex polygons — sciany pod dowolnymi katami)
# --------------------------------------------------------------------------
def _split_convex_polygon(vertices, line_p, line_normal):
    """Split convex polygon by a line.

    Line defined by point `line_p` and unit `line_normal` (perpendicular to line).
    Signed distance of point p to line = dot(p - line_p, line_normal).
    Points with positive signed distance -> 'positive' side, negative -> 'negative'.

    Returns:
        (positive_verts, negative_verts, intersection_points)
        positive_verts, negative_verts: lists of (x, y) tuples forming
            2 sub-polygons (in original CCW/CW order)
        intersection_points: list of 2 tuples where line crosses polygon edges
            (empty or fewer if line does not properly split polygon)
    """
    n = len(vertices)
    if n < 3:
        return list(vertices), [], []

    verts_arr = np.array(vertices, dtype=np.float64)
    line_p = np.array(line_p, dtype=np.float64)
    line_normal = np.array(line_normal, dtype=np.float64)
    signed = (verts_arr - line_p) @ line_normal  # shape (n,)

    positive = []
    negative = []
    intersections = []

    for i in range(n):
        v_curr = verts_arr[i]
        v_next = verts_arr[(i + 1) % n]
        d_curr = signed[i]
        d_next = signed[(i + 1) % n]

        if d_curr >= 0:
            positive.append(tuple(v_curr))
        if d_curr <= 0:
            negative.append(tuple(v_curr))

        # Edge crosses the line if signs differ (strict)
        if (d_curr > 0 and d_next < 0) or (d_curr < 0 and d_next > 0):
            t = d_curr / (d_curr - d_next)
            inter = v_curr + t * (v_next - v_curr)
            inter_tuple = (float(inter[0]), float(inter[1]))
            positive.append(inter_tuple)
            negative.append(inter_tuple)
            intersections.append(inter_tuple)

    return positive, negative, intersections


def _polygon_diameter(vertices):
    """Max pairwise distance between vertices of polygon."""
    if len(vertices) < 2:
        return 0.0
    pts = np.array(vertices)
    diffs = pts[:, None, :] - pts[None, :, :]
    return float(np.max(np.linalg.norm(diffs, axis=2)))


def _rasterize_wall_segment(positions, p1, p2, gap_center_t, gap_width,
                             wall_spacing, hx, hy):
    """Place obstacles along segment p1->p2 with gap centered at param gap_center_t."""
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    seg = p2 - p1
    seg_length = float(np.linalg.norm(seg))
    if seg_length < 1e-6:
        return
    direction = seg / seg_length
    t = 0.0
    while t <= seg_length:
        if abs(t - gap_center_t) < gap_width / 2.0:
            t += wall_spacing
            continue
        p = p1 + t * direction
        if abs(p[0]) <= hx and abs(p[1]) <= hy:
            positions.append((float(p[0]), float(p[1])))
        t += wall_spacing


def _bsp_split(vertices, depth, min_room_size, door_width, wall_spacing,
               rng, walls_out, hx, hy):
    """Recursive BSP split of a convex polygon with random-angle walls.

    Wybiera losowy kat theta ∈ [0, π), losowe przesuniecie perpendicular
    tak by oba sub-polygons mialy min_room_size srednicy. Dodaje sciane
    z losowa dziurka (drzwiami) door_width.
    """
    if depth <= 0 or len(vertices) < 3:
        return
    diameter = _polygon_diameter(vertices)
    if diameter < 2 * min_room_size:
        return

    verts_arr = np.array(vertices, dtype=np.float64)

    # Try several random angles, pick first one where valid split exists
    best_split = None
    for _ in range(6):
        theta = float(rng.uniform(0.0, np.pi))
        normal = np.array([np.cos(theta), np.sin(theta)])
        # Projections along normal
        centroid = verts_arr.mean(axis=0)
        centered_projs = (verts_arr - centroid) @ normal
        proj_min = float(centered_projs.min())
        proj_max = float(centered_projs.max())
        span = proj_max - proj_min
        if span < 2 * min_room_size:
            continue
        # Pick offset ensuring both halves >= min_room_size along normal
        offset = float(rng.uniform(proj_min + min_room_size,
                                     proj_max - min_room_size))
        line_p = centroid + offset * normal

        pos_verts, neg_verts, inters = _split_convex_polygon(
            list(vertices), tuple(line_p), tuple(normal))
        if len(inters) != 2:
            continue
        # Wall segment length
        wall_len = float(np.linalg.norm(np.array(inters[0]) - np.array(inters[1])))
        if wall_len < door_width + wall_spacing:
            continue
        # Both sub-polygons should still have min diameter
        if (_polygon_diameter(pos_verts) < min_room_size or
                _polygon_diameter(neg_verts) < min_room_size):
            continue
        best_split = (pos_verts, neg_verts, inters, wall_len, theta)
        break

    if best_split is None:
        return

    pos_verts, neg_verts, inters, wall_len, theta = best_split

    # Add wall with door gap
    gap_center_t = float(rng.uniform(door_width / 2.0 + 0.1,
                                       wall_len - door_width / 2.0 - 0.1))
    walls_out.append((inters[0], inters[1], gap_center_t))

    # Recurse (parallel subdivisions)
    _bsp_split(pos_verts, depth - 1, min_room_size, door_width, wall_spacing,
               rng, walls_out, hx, hy)
    _bsp_split(neg_verts, depth - 1, min_room_size, door_width, wall_spacing,
               rng, walls_out, hx, hy)


def building_topology(spawn_area, density, obst_size, room_height, grid_size=1.0,
                      min_room_size=2.5, door_width=1.8, wall_spacing=0.5,
                      max_depth=None, rng=None):
    """Wnetrze budynku z scianami pod DOWOLNYMI katami.

    Rekursywny podzial convex polygons (Binary Space Partitioning). Kazdy split
    to LOSOWA linia (kat theta ∈ [0, π), losowe przesuniecie), tworzaca sciane
    z losowym otworem (drzwiami). Sub-pokoje sa convex polygons (nie tylko
    prostokaty), wiec kolejne splity moga miec inne kat wzgledem parentow.

    Rezultat: layout wielopokojowy z scianami pod roznymi katami wzgledem
    siebie — realistyczna geometria wewnetrza budynku, w tym korytarze i
    zakrety (jako naturalny efekt uboczny rekursywnego podzialu).

    Zunifikowany model zastepuje poprzednie corridor + l_corridor + axis-aligned
    building — wszystkie te struktury sa specjalnymi przypadkami random-angle BSP.

    Density -> max_depth mapping (kalibrowane wg one-shot connectivity):
      d <= 0.10: depth 1  (2 pokoje, ~91% one-shot connected)
      d <= 0.20: depth 2  (do 4 pokoi, ~43% one-shot, ~98% via retry)
      d <= 0.30: depth 2  (jak wyzej, wieksze density = wieksza density odrzucanych)
      d  > 0.30: depth 3  (do 8 pokoi — density=0.4 bardzo agresywne)

    Args:
      min_room_size: minimalna srednica sub-pokoju (m) — zapewnia rozsadny
        rozmiar do przelotu drona.
      door_width: szerokosc drzwi (m).
      wall_spacing: odleglosc obstacle w scianie (default 0.5m -> overlap
        ~0.1m dla wizualnie solid wall).
      max_depth: opcjonalne wymuszenie glebokosci BSP. None = z density.
    """
    if rng is None:
        rng = np.random

    hx, hy = spawn_area[0] / 2.0, spawn_area[1] / 2.0
    obst_area_length = int(spawn_area[0])
    obst_area_width = int(spawn_area[1])

    if max_depth is None:
        if density <= 0.10:
            max_depth = 1
        elif density <= 0.30:
            max_depth = 2
        else:
            max_depth = 3

    # Initial polygon = room bounds (CCW)
    initial_vertices = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]

    walls = []  # list of (p1, p2, gap_center_t)
    _bsp_split(initial_vertices, max_depth, min_room_size, door_width,
               wall_spacing, rng, walls, hx, hy)

    positions = []
    for p1, p2, gap_t in walls:
        _rasterize_wall_segment(positions, p1, p2, gap_t, door_width,
                                 wall_spacing, hx, hy)

    obst_pos_arr = [[x, y, room_height / 2.0] for x, y in positions]
    obst_map = _positions_to_obst_map(positions, spawn_area, grid_size)
    cell_centers = get_cell_centers(obst_area_length, obst_area_width, grid_size)
    return obst_map, obst_pos_arr, cell_centers


# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------
TOPOLOGY_GENERATORS = {
    'grid': grid_topology,
    'poisson': poisson_topology,
    'cluster': cluster_topology,
    'building': building_topology,
}


# --------------------------------------------------------------------------
# Connectivity check (BFS on non-dilated map)
# --------------------------------------------------------------------------
def _compute_bare_obst_map(positions, spawn_area, grid_size=1.0):
    """Non-dilated obst_map: 1 tylko dla komorek ZAWIERAJACYCH obstacle center.

    Uzywane wylacznie do BFS connectivity check — zapewnia ze test bierze pod
    uwage rzeczywista geometrie (mozna przelesc przez ~1m komorke z obstacle'em
    w sasiedniej), a nie super-bezpieczna dilated wersja uzywana do spawn'u.
    """
    obst_area_length = int(spawn_area[0])
    obst_area_width = int(spawn_area[1])
    bare_map = np.zeros((obst_area_length, obst_area_width), dtype=np.int8)
    for x, y in positions:
        rid = int((x + obst_area_length / 2.0) / grid_size)
        cid = int((y + obst_area_width / 2.0) / grid_size)
        rid = max(0, min(obst_area_length - 1, rid))
        cid = max(0, min(obst_area_width - 1, cid))
        bare_map[rid, cid] = 1
    return bare_map


def label_components(binary_map):
    """Label connected components w wolnej przestrzeni (binary_map==0).

    4-connected BFS. Zajete komorki (binary_map==1) dostaja etykiete 0.
    Wolne komorki dostaja unikalne id (1, 2, ...).

    Returns:
        (labeled_array int32, n_components int)
    """
    H, W = binary_map.shape
    labeled = np.zeros_like(binary_map, dtype=np.int32)
    current_label = 0
    for r in range(H):
        for c in range(W):
            if binary_map[r, c] == 0 and labeled[r, c] == 0:
                current_label += 1
                queue = [(r, c)]
                labeled[r, c] = current_label
                while queue:
                    cr, cc = queue.pop(0)
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if (0 <= nr < H and 0 <= nc < W
                                and binary_map[nr, nc] == 0
                                and labeled[nr, nc] == 0):
                            labeled[nr, nc] = current_label
                            queue.append((nr, nc))
    return labeled, current_label


def is_free_space_connected(binary_map):
    """True jesli wolne komorki (binary_map==0) tworza JEDEN spojny komponent."""
    if np.all(binary_map == 1):
        return False
    _, n_components = label_components(binary_map)
    return n_components == 1


# --------------------------------------------------------------------------
# Main dispatch with connectivity retry
# --------------------------------------------------------------------------
def generate_topology(topology_name, spawn_area, density, obst_size,
                      room_height, grid_size=1.0, rng=None,
                      ensure_connectivity=True, max_retries=10,
                      return_stats=False, **kwargs):
    """Main dispatch.

    Args:
        ensure_connectivity: jesli True, powtarza generacje az wolna
            przestrzen bedzie spojna (wg BFS na NON-dilated map).
        max_retries: max liczba prob. Po tym warning + zwraca ostatnia probe.
        return_stats: jesli True, zwraca 4-tuple z dictem statystyk.
        kwargs['keep_rejected']: jesli True, stats zawiera listy rejected.

    Returns:
        (obst_map, obst_pos_arr, cell_centers) domyslnie
        (obst_map, obst_pos_arr, cell_centers, stats) gdy return_stats=True
    """
    if topology_name not in TOPOLOGY_GENERATORS:
        raise ValueError(
            f"Unknown topology: '{topology_name}'. "
            f"Available: {list(TOPOLOGY_GENERATORS.keys())}"
        )
    if rng is None:
        rng = np.random

    generator_fn = TOPOLOGY_GENERATORS[topology_name]
    keep_rejected = kwargs.pop('keep_rejected', False)

    stats = {'attempts': 0, 'fallback': 0}
    if keep_rejected:
        stats['rejected_maps'] = []

    obst_map, pos_arr, cells = None, None, None
    for attempt in range(max_retries):
        stats['attempts'] = attempt + 1
        obst_map, pos_arr, cells = generator_fn(
            spawn_area, density, obst_size, room_height,
            grid_size=grid_size, rng=rng, **kwargs
        )

        if not ensure_connectivity:
            break

        bare_map = _compute_bare_obst_map(
            [(p[0], p[1]) for p in pos_arr], spawn_area, grid_size
        )
        if is_free_space_connected(bare_map):
            break

        if keep_rejected:
            stats['rejected_maps'].append((obst_map.copy(),
                                            [(p[0], p[1]) for p in pos_arr],
                                            bare_map.copy()))
    else:
        stats['fallback'] = 1
        warnings.warn(
            f"Topology '{topology_name}' failed connectivity check after "
            f"{max_retries} retries. Returning last attempt (may have "
            f"disconnected free space).",
            RuntimeWarning
        )

    if return_stats:
        return obst_map, pos_arr, cells, stats
    return obst_map, pos_arr, cells
