import numpy as np
from collections import defaultdict
from typing import Iterable, Dict, List, Tuple
from shapely.geometry import Polygon, Point, LineString

Coordinate = Tuple[float, float]
Grid = List[Coordinate]

_DIRS: Tuple[Coordinate, ...] = (
    (2, 0),  # East
    (-2, 0),  # West
    (0, 2),  # North
    (0, -2),  # South
)


class STCMixin:
    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Path Generation
    # ------------------------------------------------------------------

    def polygon_to_grid(self, cell_poly: Polygon, cell_size: float) -> Grid:
        """Rasterise *cell_poly* to the centres of a square grid."""
        assert not cell_poly.is_empty, "cell_poly must not be empty"

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
        assert inside, "No grid cells found inside the polygon"
        return inside

    def _adjacency(
        self, cells: Iterable[Coordinate]
    ) -> Dict[Coordinate, List[Coordinate]]:
        """Return the 4‑neighbour adjacency list of *cells*."""
        cell_set = set(cells)
        nbrs: Dict[Coordinate, List[Coordinate]] = defaultdict(list)

        for x, y in cell_set:
            for dx, dy in _DIRS:
                n = (x + dx, y + dy)
                if n in cell_set:
                    nbrs[(x, y)].append(n)

        return nbrs

    def compute_stc(self, cells: Grid) -> Grid:
        """Compute a deterministic **Spanning‑Tree Coverage (STC)** tour over *cells*.

        The depth‑first traversal starts from the lexicographically smallest cell and
        records every entry into a cell as well as the corresponding back‑track. The
        resulting tour therefore visits some cells multiple times but ensures full
        coverage while remaining simple to execute.
        """
        if not cells:
            return []

        root = min(cells, key=lambda p: (p[1], p[0]))
        nbrs = self._adjacency(cells)

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

    def offset_stc_path(self, stc_path: Grid, cell_size: float) -> Grid:
        """Return a smooth, closed offset path surrounding *stc_path*.

        The path is obtained by buffering the open STC polyline by a quarter of the
        grid cell size using round caps and joins.
        """
        radius_factor = 0.25  # quarter of the cell size
        resolution = 4  # number of segments per quarter circle
        if not stc_path or len(stc_path) < 2:
            self.get_logger().warn("STC path is empty or has less than 2 points")
            return []

        r = cell_size * radius_factor
        line = LineString(stc_path)  # keep path open – round caps form at both ends

        poly = line.buffer(r, join_style=1, cap_style=1, resolution=resolution)
        poly = poly.buffer(r * 0.99, join_style=1, cap_style=1, resolution=resolution)
        poly = poly.buffer(-r * 1.0, join_style=1, cap_style=1, resolution=resolution)

        # Exterior ring traces the final closed loop
        ring = poly.exterior
        return ring
