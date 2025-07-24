# main.py

from simulation import SimulationState
from robots import compute_power_cell, polygon_to_grid, compute_stc, offset_stc_path
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


# initial_positions = {
#     0: (1, 1),
#     1: (1, 2),
#     2: (9, 5),
#     3: (18, 5),
# }

initial_positions = {
        0: (1, 1),
        1: (2, 4),
        2: (3, 7),
        3: (19, 5),
        4: (17, 2),
        5: (19, 8),
    }

# initial_positions = {
#     0: (1, 1),
#     1: (2, 4),
# }
# 2) create the simulation state
agent_ids = list(initial_positions.keys())


sim_state = SimulationState(agent_ids)


for i, pos in initial_positions.items():
    sim_state.set_agent_position(i, pos)

# Neighbors: Make every agent a neighbor of every other agent
for i in agent_ids:
    for j in agent_ids:
        if i != j:
            sim_state.neighbours[i].append(j)

# Define the Shape
# Create an L-shaped bounding polygon
coords_l = [
    (0, 0),
    (50, 0),
    (50,20),
    (35, 25),
    (30, 50),
    (0, 50)
]
# Create a rectangular bounding polygon
coords_rect = [
    (0, 0),
    (10, 0),
    (10, 10),
    (0, 10),]

l_shape = Polygon(coords_l)
sim_state.bounding_polygon = l_shape

gamma = 0.075
f = sim_state.bounding_polygon.area /  len(sim_state.get_agent_ids())


while True:
    for agent_id in sim_state.get_agent_ids():

        cell = compute_power_cell(sim_state, agent_id)
        area     = cell.area
        centroid = (cell.centroid.x, cell.centroid.y)

        sim_state.set_agent_area_centroid(agent_id, area, centroid)
        
        # 4) update my weight
        p_i, w_i = sim_state.get_agent_seed_and_weight(agent_id)
        w_new = w_i - gamma * (area - f)
        sim_state.set_agent_weight(agent_id, w_new)

        # 5) (optional) print
        errors = [abs(sim_state.get_agent_area(agent_id) - f)/f for i in sim_state.get_agent_ids()]
    #print("Current weights:", sim_state.weights, "\n Areas:", sim_state.areas)
    # 6) check convergence
    if max(errors) < 0.01:
        print("Converged!")
        print(f"final areas: {sim_state.areas}")
        break
    
#path for stc
cell_size = 1.0
paths = []

# --- Plot final power cells ---
fig, ax = plt.subplots(figsize=(20,10))
colors = plt.cm.tab10.colors  # a palette of 10 distinct colors

for agent_id in sim_state.get_agent_ids():
    cell = compute_power_cell(sim_state, agent_id)
    if cell.is_empty:
        continue

    # extract polygon boundary
    x, y = cell.exterior.xy
    ax.fill(
        x, y,
        alpha=0.25,
        color=colors[agent_id % len(colors)],
        label=f'Agent {agent_id}'
    )

    grid = polygon_to_grid(cell, cell_size)
    
    #plot the grid points
    gx, gy = zip(*grid)
    ax.plot(gx, gy, 'o', markersize=4, color=colors[agent_id % len(colors)], label=f'Grid Points {agent_id}')

    # compute STC path
    tree = compute_stc(grid)
    xt, yt = zip(*tree)
    
    # offset the path
    path = offset_stc_path(tree, cell_size  )
    paths.append(path)
    xp, yp = zip(*path)
    
    #plot the centroid
    cx, cy = sim_state.centroids[agent_id]
    ax.plot(cx, cy, 'ro', markersize=5, label=f'Centroid {agent_id}')
    ax.plot(xt, yt, 'k--', linewidth=1, label=f'STC Path {agent_id}')
    ax.plot(xp, yp, '-', linewidth=2, color=colors[agent_id % len(colors)], label=f'Offset Path {agent_id}')


    ax.set_title(f'Power Cells and STC Paths for {len(sim_state.get_agent_ids())} Agents')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    ax.set_aspect('equal', adjustable='box')    
    ax.legend(loc='upper right')
plt.grid()
plt.show()


