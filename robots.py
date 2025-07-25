"""robots.py

Geometry utilities and coverage‑path planning helpers for a 2‑D multi‑robot
simulation. All functions are side‑effect free and safe to import without
initialisation.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split

import math
import numpy as np
from typing import List, Union
from shapely.geometry import LineString, LinearRing, Point

from simulation import UAV

Coordinate = Tuple[float, float]
Grid = List[Coordinate]


def compute_power_cell(sim_state: Any, agent_id: int) -> Polygon:
    """Return the *power cell* of ``agent_id`` clipped to the simulation boundary.

    The power cell is the region closer to the agent’s weighted seed than to any
    neighbour’s seed under the *power distance* ``‖q − p‖² − w``.

    Returns
    -------
    shapely.geometry.Polygon
        Closed polygon representing the agent’s power cell within the workspace.
    """
    boundary: Polygon = sim_state.bounding_polygon
    p_i, w_i = sim_state.get_agent_seed_and_weight(agent_id)

    cell: Polygon = boundary  # start with the entire workspace

    for j in sim_state.get_agent_ids():
        if j != agent_id:
            p_j, w_j = sim_state.get_agent_seed_and_weight(j)

            # ------------------------------------------------------------------
            # Half‑plane bisector of two weighted points
            #   (‖q − p_j‖² − w_j) = (‖q − p_i‖² − w_i)  ⇒  A·q = B
            # ------------------------------------------------------------------
            Ax, Ay = p_j[0] - p_i[0], p_j[1] - p_i[1]
            B = (p_j[0] ** 2 + p_j[1] ** 2 - p_i[0] ** 2 - p_i[1] ** 2 + w_i - w_j) * 0.5

            # A point (px, py) on the bisector and a perpendicular direction (dx, dy)
            denom = Ax * Ax + Ay * Ay
            px, py = Ax * B / denom, Ay * B / denom
            dx, dy = -Ay, Ax

            # Construct a finite line segment that certainly intersects the boundary
            extent = max(boundary.bounds[2] - boundary.bounds[0],
                        boundary.bounds[3] - boundary.bounds[1]) * 2
            bisector = LineString([(px + dx * extent, py + dy * extent),
                                (px - dx * extent, py - dy * extent)])

            # Split the current cell and retain the half‑plane containing p_i
            for part in split(cell, bisector).geoms:
                q = part.representative_point()
                if Ax * q.x + Ay * q.y <= B:
                    cell = part
                    break  # done with this neighbour
            # If no part satisfies the inequality, *cell* remains unchanged (degenerate)

    return cell.intersection(boundary)


def polygon_to_grid(cell_poly: Polygon, cell_size: float) -> Grid:
    """Rasterise *cell_poly* to the centres of a square grid.

    Grid cell centres are located at ``((k + 0.5)·cell_size, (ℓ + 0.5)·cell_size)``
    with ``k, ℓ ∈ ℤ``.

    Returns
    -------
    list[tuple[float, float]]
        Coordinates of every cell centre lying *strictly* inside *cell_poly*,
        sorted first by *y* then by *x* for deterministic output.
    """
    half = 0.5 * cell_size
    minx, miny, maxx, maxy = cell_poly.bounds

    ix_min = int(np.ceil((minx - half) / cell_size))
    iy_min = int(np.ceil((miny - half) / cell_size))
    ix_max = int(np.floor((maxx - half) / cell_size))
    iy_max = int(np.floor((maxy - half) / cell_size))

    inside: Grid = []
    for ix in range(ix_min, ix_max + 1):
        x = (ix + 0.5) * cell_size
        for iy in range(iy_min, iy_max + 1):
            y = (iy + 0.5) * cell_size
            if cell_poly.contains(Point(x, y)):
                inside.append((x, y))

    inside.sort(key=lambda p: (p[1], p[0]))  # reproducible ordering
    return inside


_DIRS: Tuple[Coordinate, ...] = (
    (1, 0),  # East
    (-1, 0), # West
    (0, 1),  # North
    (0, -1), # South
)


def _adjacency(cells: Iterable[Coordinate]) -> Dict[Coordinate, List[Coordinate]]:
    """Return the 4‑neighbour adjacency list of *cells*."""
    cell_set = set(cells)
    nbrs: Dict[Coordinate, List[Coordinate]] = defaultdict(list)

    for x, y in cell_set:
        for dx, dy in _DIRS:
            n = (x + dx, y + dy)
            if n in cell_set:
                nbrs[(x, y)].append(n)

    return nbrs


def compute_stc(cells: Grid) -> Grid:
    """Compute a deterministic **Spanning‑Tree Coverage (STC)** tour over *cells*.

    The depth‑first traversal starts from the lexicographically smallest cell and
    records every entry into a cell as well as the corresponding back‑track. The
    resulting tour therefore visits some cells multiple times but ensures full
    coverage while remaining simple to execute.
    """
    if not cells:
        return []

    root = min(cells, key=lambda p: (p[1], p[0]))
    nbrs = _adjacency(cells)

    visited = {root}
    tour: Grid = []

    def dfs(u: Coordinate) -> None:
        tour.append(u)  # enter cell
        for v in sorted(nbrs[u], key=lambda p: (p[1], p[0])):
            if v not in visited:
                visited.add(v)
                dfs(v)
                tour.append(u)  # back‑track

    dfs(root)
    return tour


def offset_stc_path(stc_path: Grid, cell_size: float) -> Grid:
    """Return a smooth, closed offset path surrounding *stc_path*.

    The path is obtained by buffering the open STC polyline by a quarter of the
    grid cell size using round caps and joins.
    """
    radius_factor = 0.25  # quarter of the cell size
    resolution = 4  # number of segments per quarter circle
    if not stc_path:
        return []

    r = cell_size * radius_factor
    line = LineString(stc_path)  # keep path open – round caps form at both ends

    # Initial offset
    poly = line.buffer(r, join_style=1, cap_style=1, resolution=resolution)

    # Morphological closing: dilate then erode by the same radius. This rounds
    # *concave* corners that the first buffering step leaves unchanged.
    poly = poly.buffer(r*0.99, join_style=1, cap_style=1, resolution=resolution)
    poly = poly.buffer(-r*1.0, join_style=1, cap_style=1, resolution=resolution)

    # Exterior ring traces the final closed loop
    ring = poly.exterior
    return ring


def create_speed_profile(
    uav: UAV,
    ring: LinearRing,
    angle_thresh: float = math.radians(5),
) -> List[float]:
    """One speed per sample along `ring`, frozen on curves."""
    L = ring.length
    interval = 1.0  # default distance between samples
    if L == 0 or interval <= 0:
        return []

    # 1) sample distances and points
    n = int(math.floor(L / interval)) + 1
    dists = np.linspace(0, L, n)
    pts = np.array([ring.interpolate(d).coords[0] for d in dists])

    # 2) compute headings of each segment
    deltas = pts[1:] - pts[:-1]                    # shape (n-1, 2)
    headings = np.arctan2(deltas[:,1], deltas[:,0])  # shape (n-1,)

    # 3) detect where heading jumps by more than threshold
    #    prepend a zero so we have one flag per point
    dh = np.abs(np.diff(headings, prepend=headings[0]))
    is_curve = dh >= angle_thresh                # curve‐flag for each segment

    # 4) build speed profile
    speeds = [0.5*(uav.v_min + uav.v_max)]        # seed at mid-range
    for i in range(1, n):
        if is_curve[i-1]:                         # if previous segment was curved
            speeds.append(speeds[-1])             # freeze speed
        else:
            # random ±Δv walk on straight
            step = np.random.choice([-uav.delta_v, 0, uav.delta_v])
            cand = speeds[-1] + step
            # clamp
            speeds.append(min(max(cand, uav.v_min), uav.v_max))
    return speeds

def create_const_speed_profile(
    uav: UAV,
    ring: LinearRing,
    speed: float = 15.0,
) -> List[float]:
    """Create a constant speed profile for the UAV along the `ring`."""
    if speed < uav.v_min or speed > uav.v_max:
        raise ValueError(f"Speed {speed} is out of bounds [{uav.v_min}, {uav.v_max}]")

    n = int(math.ceil(ring.length / 1.0))  # number of samples
    return [speed] * n  # constant speed for each sample


def compute_energy_profile(uav: UAV, ring: LinearRing, speeds: List[float]) -> List[float]:
    """Compute energy profile for a UAV flying along `ring` at `interval` spacing."""
    Energy = 0.0
    
    coords = list(ring.coords)          # freeze a copy once
    if len(coords) < 2:                 # guard against degenerate rings
        raise ValueError("ring has < 2 vertices")
    
    if len(speeds) != len(coords) - 1:
        raise ValueError("speeds must match segment count")

    for i in range(len(speeds)):
        # Power = Drag + Thrust
        drag = uav.B * speeds[i] ** 2
        thrust = uav.A / (speeds[i] ** 2) 
        power = drag + thrust

        # Energy = Power * Time
        if i == 0:
            dist = Point(coords[0]).distance(Point(coords[1]))
        else:
            dist = Point(coords[i]).distance(Point(coords[i-1]))
        time = dist / speeds[i]
        Energy += power * time
        
    # Check if in a turn (angle difference between segments is significant)

    
    for i in range(1, len(coords)):
        p1 = Point(coords[i-1])
        p2 = Point(coords[i])
        angle_diff = p1.angle(p2)
        if abs(angle_diff) > math.radians(5):  # threshold for significant turn
            # Adjust energy for turns if necessary
            phi = math.atan(speeds[i]**2 / (9.81 * 0.5))
            power *= 1.0 / math.cos(phi)
        print(f"Energy: {power}")

    return Energy
