"""main.py

Entry point for the multi‑robot power‑diagram balancing demo. The script

1.  Constructs a :class:`simulation.SimulationState` from a set of initial
    agent positions and an *L‑shaped* workspace polygon.
2.  Iteratively balances agent weights until every power cell area converges to
    ``workspace_area / n_agents`` within a relative tolerance.
3.  Generates a figure showing the final power cells together with grid points,
    STC tree paths, and smoothed offset paths.

"""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from robots import (
    compute_power_cell,
    compute_stc,
    offset_stc_path,
    polygon_to_grid,
    compute_energy_profile,
)
from simulation import SimulationState, UAV  # type: ignore  # external module

# Initial agent positions (id → (x, y))
INITIAL_POSITIONS: Dict[int, Tuple[float, float]] = {
    0: (1, 1),
    1: (2, 4),
    # 2: (3, 7),
    # 3: (19, 5),
    # 4: (17, 2),
    # 5: (19, 8),
}

# Workspace: L‑shaped polygon defined counter‑clockwise
WORKSPACE_COORDS: List[Tuple[float, float]] = [
    # (0, 0),
    # (50, 0),
    # (50, 20),
    # (35, 25),
    # (30, 50),
    # (0, 50),
    (0, 0),
    (10, 0),
    (10, 10),
    (0, 10),
]

# Balancing parameters
GAMMA: float = 0.075  # gradient‑descent step size
TOLERANCE: float = 0.01  # relative area error for convergence
CELL_SIZE: float = 1.0  # grid resolution for STC planning



def build_simulation_state() -> SimulationState:
    """Return a fully initialised :class:`SimulationState`."""
    sim_state = SimulationState(list(INITIAL_POSITIONS.keys()))

    for agent_id, pos in INITIAL_POSITIONS.items():
        sim_state.set_agent_position(agent_id, pos)

    sim_state.bounding_polygon = Polygon(WORKSPACE_COORDS)
    return sim_state


def balance_power_cells(sim_state: SimulationState) -> None:
    """Adjust agent weights until power‑cell areas equalise within tolerance."""
    workspace_area = sim_state.bounding_polygon.area
    target_area = workspace_area / len(sim_state.get_agent_ids())

    while True:
        for agent_id in sim_state.get_agent_ids():
            cell = compute_power_cell(sim_state, agent_id)
            area = cell.area
            centroid = (cell.centroid.x, cell.centroid.y)
            sim_state.set_agent_area_centroid(agent_id, area, centroid)

            # Gradient descent weight update
            _, weight = sim_state.get_agent_seed_and_weight(agent_id)
            sim_state.set_agent_weight(agent_id, weight - GAMMA * (area - target_area))

        # Check convergence
        errors = [
            abs(sim_state.get_agent_area(i) - target_area) / target_area
            for i in sim_state.get_agent_ids()
        ]
        if max(errors) < TOLERANCE:
            print("Converged!")
            break


def plot_results(sim_state: SimulationState) -> None:
    """Plot power cells, grid points, and coverage paths."""
    fig, ax = plt.subplots(figsize=(20, 10))
    colors = plt.cm.tab10.colors  # cycle of 10 distinct colours

    for agent_id in sim_state.get_agent_ids():
        cell = compute_power_cell(sim_state, agent_id)
        if cell.is_empty:
            continue

        # Power‑cell polygon
        x, y = cell.exterior.xy
        ax.fill(x, y, alpha=0.25, color=colors[agent_id % len(colors)])

        # Raster grid and STC path
        grid = polygon_to_grid(cell, CELL_SIZE)
        if grid:
            gx, gy = zip(*grid)
            ax.plot(gx, gy, "o", markersize=4, color=colors[agent_id % len(colors)])

            stc_path = compute_stc(grid)
            sx, sy = zip(*stc_path)
            ax.plot(sx, sy, "k--", linewidth=1)

            offset_path = offset_stc_path(stc_path, CELL_SIZE)
            ox, oy = zip(*offset_path.coords)
            ax.plot(ox, oy, "-", linewidth=2, color=colors[agent_id % len(colors)])

        # Centroid
        cx, cy = sim_state.centroids[agent_id]
        ax.plot(cx, cy, "ro", markersize=5)
        
        #Energy profile
        uav = UAV()  # Create a UAV instance
        Energy = compute_energy_profile(uav, offset_path)
        print("Energy profile computed: ", Energy)

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"Power Cells and Coverage Paths for {len(sim_state.get_agent_ids())} Agents"
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True)
    plt.show()
    





def main() -> None:
    sim_state = build_simulation_state()
    balance_power_cells(sim_state)
    plot_results(sim_state)
    



if __name__ == "__main__":
    main()
