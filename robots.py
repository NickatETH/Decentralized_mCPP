# power_voronoi.py

from shapely.geometry import Point, LineString, Polygon
from shapely.ops import split
from typing import Any

def compute_power_cell(sim_state: Any, agent_id: int) -> Polygon:
    """
    Compute the weighted Voronoi (power) cell for a single agent.

    Args:
      sim_state : an instance of SimulationState
      agent_id  : the ID of the agent whose cell to compute

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