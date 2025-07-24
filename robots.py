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
    if not stc_path:
        return []

    line = LineString(stc_path)  # Keep open — rounded end caps extend properly
    offset_poly = line.buffer(cell_size / 4.0, join_style=1, cap_style=1)

    return list(offset_poly.exterior.coords)
