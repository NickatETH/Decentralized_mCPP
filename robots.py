# robots.py

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import split
from typing import Any
import numpy as np
from collections import defaultdict



def compute_power_cell(sim_state: Any, agent_id: int) -> Polygon:
    """
    Compute the weighted Voronoi (power) cell for a single agent.
    Returns:
      A Shapely Polygon representing the agent’s power cell 
      clipped to sim_state.bounding_polygon.
    """
    boundary: Polygon = sim_state.bounding_polygon
    p_i, w_i = sim_state.get_agent_seed_and_weight(agent_id)

    # start from the whole workspace
    cell = boundary

    for j in sim_state.get_neighbours(agent_id):
        p_j, w_j = sim_state.get_agent_seed_and_weight(j)

        # compute bisector coefficients A·q = B
        Ax = p_j[0] - p_i[0]
        Ay = p_j[1] - p_i[1]
        B  = ((p_j[0]**2 + p_j[1]**2)
            - (p_i[0]**2 + p_i[1]**2)
            +  w_i
            -  w_j) * 0.5

        # direction perpendicular to A
        dx, dy = -Ay, Ax
        # scale large enough to cross the boundary
        length = max(boundary.bounds[2] - boundary.bounds[0],
                     boundary.bounds[3] - boundary.bounds[1]) * 2
        # a point on the bisector line
        denom = Ax*Ax + Ay*Ay
        px = Ax * B / denom
        py = Ay * B / denom

        p1 = (px + dx * length, py + dy * length)
        p2 = (px - dx * length, py - dy * length)
        bisector = LineString([p1, p2])

        # split returns a GeometryCollection
        pieces = split(cell, bisector)

        # iterate over its .geoms, not over pieces directly
        new_cell = None
        for poly in pieces.geoms:
            q = poly.representative_point()
            if Ax * q.x + Ay * q.y <= B:
                new_cell = poly
                break

        if new_cell is not None:
            cell = new_cell
        # else: degenerate—keep existing cell

    return cell.intersection(boundary)


def polygon_to_grid(cell_poly: Polygon, cell_size: float) -> list[tuple[float, float]]:
    """
    Rasterise *cell_poly* onto a square grid whose **cell centres** are located at
    ((k + 0.5)·cell_size, (ℓ + 0.5)·cell_size) with k, ℓ ∈ ℤ.

    Returns
    -------
    inside_coords : list[tuple[float, float]]
        Coordinates of every grid cell‑centre that lies strictly inside the polygon,
        sorted by increasing y, then increasing x.
    """
    half = 0.5 * cell_size
    minx, miny, maxx, maxy = cell_poly.bounds

    # Determine the integer index ranges so that the shifted coordinates stay inside the bbox
    ix_min = int(np.ceil((minx - half) / cell_size))
    iy_min = int(np.ceil((miny - half) / cell_size))
    ix_max = int(np.floor((maxx - half) / cell_size))
    iy_max = int(np.floor((maxy - half) / cell_size))

    inside_coords: list[tuple[float, float]] = []
    for i in range(ix_min, ix_max + 1):          # x‑index
        x = (i + 0.5) * cell_size
        for j in range(iy_min, iy_max + 1):      # y‑index
            y = (j + 0.5) * cell_size
            if cell_poly.contains(Point(x, y)):  # strictly inside
                inside_coords.append((x, y))

    # ── Sort for reproducibility: lowest y first, then x ───────────────────────
    inside_coords.sort(key=lambda p: (p[1], p[0]))
    return inside_coords


def _build_nbrs(cells):
    cell_set           = set(cells)
    directions         = [(1, 0), (-1, 0), (0, 1), (0, -1)]          # E, W, N, S
    nbrs = defaultdict(list)

    for x, y in cell_set:
        for dx, dy in directions:
            n = (x + dx, y + dy)
            if n in cell_set:
                nbrs[(x, y)].append(n)
    return nbrs


def compute_stc(cells):
    """
    Returns
    -------
    list[(float, float)]
        STC tour: the robot visits every cell; edges of the tree
        are traversed twice (one forward, one back‑track).
    """
    if not cells:
        return []

    # Choose a deterministic root if caller doesn't.
    start =  min(cells, key=lambda p: (p[1], p[0]))
    nbrs  = _build_nbrs(cells)

    visited = {start}
    path    = []

    def dfs(u):
        path.append(u)                              # Enter / cover cell
        for v in sorted(nbrs[u], key=lambda p: (p[1], p[0])):  # Stable order
            if v not in visited:                    # Tree edge
                visited.add(v)
                dfs(v)                              # Recurse
                path.append(u)                      # Back‑track

    dfs(start)
    return path




def offset_stc_path(stc_path: list, cell_size: float) -> list:
    """
    Given an STC path of cell-center waypoints, create a robot path
    running parallel (offset by half cell_size) *and* closed around it.

    Returns:
        offset_coords: list of (x, y) world-coordinate points tracing
        a closed offset loop around the STC path.
    """
    if not stc_path:
        return []
    
    # 1) Don't close the loop - keep it as an open LineString
    # This allows the buffer to extend properly at the ends
    loop = LineString(stc_path)
    
    # 2) Buffer the open path by half a cell to get a uniform offset polygon
    dist = cell_size / 4.0
    # join_style=1 (round) for rounded corners as you mentioned
    # cap_style=1 (round) extends the ends properly
    offset_poly: Polygon = loop.buffer(dist,
                                     join_style=1,
                                     cap_style=1)
    
    offset_coords = list(offset_poly.exterior.coords)
    
    return offset_coords