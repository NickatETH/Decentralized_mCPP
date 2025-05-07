import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

# Time-varying ground truth importance function with hotspots that move
def ground_truth_function(x, y, t=0):
    # t should be between 0 and 1 (normalized time)
    centers = np.array([
        [0.2, 0.2],
        [0.2 + 0.1*t, 0.8],  # This hotspot moves from left to right
        [0.8, 0.2 + 0.3*t],  # This hotspot moves from bottom to top
        [0.8, 0.8],
        [0.5, 0.5],
        [0.3, 0.5],
        [0.5, 0.3],
        [0.7, 0.5]
    ])
    # Heights also change with time - some increase, some decrease
    heights = [
        0.4, 
        2.0 * (1 - 0.01*t),  # Decreases over time
        2.0 * (1 + 0.1*t),      # Increases over time
        0.4,
        0.3 * (1 + 0.1*t),    # Increases over time
        0.3,
        0.3, 
        0.3
    ]
    widths = [0.02, 0.04, 0.04, 0.02, 0.04, 0.03, 0.025, 0.05]
    val = np.zeros_like(x)
    for (cx, cy), h, w in zip(centers, heights, widths):
        val += h * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * w))
    return val

# RBF kernel
def rbf_kernel(a, b, lengthscale=0.2):
    d2 = distance_matrix(a, b)**2
    return np.exp(-0.5 * d2 / lengthscale**2)

# Simulation parameters
np.random.seed(2)
num_drones = 8
iterations = 150
noise_std = 0.1
lengthscale = 0.2
v = 0.01              # constant forward speed
max_turn = np.pi/10   # maximum steering angle per step (rad)

# Domain and grid
grid_size = 50
xs = np.linspace(0, 1, grid_size)
ys = np.linspace(0, 1, grid_size)
Xg, Yg = np.meshgrid(xs, ys)
points_grid = np.vstack([Xg.ravel(), Yg.ravel()]).T

# Initial drone positions and headings
drone_positions = np.random.rand(num_drones, 2)
headings = np.random.rand(num_drones) * 2 * np.pi

# Storage for measurements and history
X_data, y_data = [], []
positions_over_time = [drone_positions.copy()]
headings_over_time = [headings.copy()]

# Store Voronoi regions for each iteration
region_history = []
time_values = []

for i in range(iterations):
    # Calculate normalized time (0 to 1)
    t = i / (iterations - 1) if iterations > 1 else 0
    time_values.append(t)
    
    # Collect measurements at current positions with current ground truth
    for pos in drone_positions:
        noisy = ground_truth_function(pos[0], pos[1], t) + noise_std * np.random.randn()
        X_data.append(pos.copy())
        y_data.append(noisy)
    Xd = np.array(X_data)
    yd = np.array(y_data)

    # Gaussian Process Regression
    K = rbf_kernel(Xd, Xd, lengthscale) + noise_std**2 * np.eye(len(Xd))
    alpha = np.linalg.solve(K, yd)
    K_star = rbf_kernel(Xd, points_grid, lengthscale)
    mu_pred = K_star.T.dot(alpha)

    # Compute centroids of weighted Voronoi regions
    dists = distance_matrix(points_grid, drone_positions)
    regions = np.argmin(dists, axis=1)

    centroids = np.zeros_like(drone_positions)
    for i in range(num_drones):
        pts = points_grid[regions == i]
        if len(pts) > 0:
            w = mu_pred[regions == i]
            centroids[i] = np.average(pts, axis=0, weights=w)
        else:
            centroids[i] = drone_positions[i]

    # Vehicle dynamics: steer toward centroid, move forward
    for i in range(num_drones):
        dx, dy = centroids[i] - drone_positions[i]
        desired = np.arctan2(dy, dx)
        diff = (desired - headings[i] + np.pi) % (2*np.pi) - np.pi
        steer = np.clip(diff, -max_turn, max_turn)
        headings[i] += steer
        # move forward
        drone_positions[i] += v * np.array([np.cos(headings[i]), np.sin(headings[i])])
        # keep within [0,1]
        drone_positions[i] = np.clip(drone_positions[i], 0, 1)

    positions_over_time.append(drone_positions.copy())
    headings_over_time.append(headings.copy())
    
    # Store the regions for this iteration
    region_history.append(regions.copy())

# Create animation with Voronoi regions and arrows
fig, ax = plt.subplots(figsize=(16,16))
def update(frame):
    ax.clear()
    
    # Get time value for this frame
    t = time_values[min(frame, len(time_values)-1)]
    
    # Plot ground truth function with current time value
    contour = ax.contourf(Xg, Yg, ground_truth_function(Xg, Yg, t), levels=20, alpha=0.6)
    
    # Plot Voronoi regions if available for this frame
    if frame > 0:  # Skip the initial frame as we have no measurements yet
        regions = region_history[frame-1]
        # Different color for each drone's region
        colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightsalmon', 'lightgray']
        
        # Reshape to match the grid for visualization
        region_grid = np.reshape(regions, (grid_size, grid_size))
        
        # Plot each region with a different color
        for i in range(num_drones):
            mask = region_grid == i
            if np.any(mask):
                ax.contourf(Xg, Yg, mask.astype(float), levels=[0.5, 1.5], 
                           colors=[colors[i % len(colors)]], alpha=0.3)
    
    # Get drone positions and headings
    pos = positions_over_time[frame]
    current_headings = headings_over_time[frame]
    
    # Plot arrows for each drone
    for i in range(num_drones):
        dx = np.cos(current_headings[i]) * 0.03
        dy = np.sin(current_headings[i]) * 0.03
        
        ax.arrow(pos[i,0] - dx, pos[i,1] - dy, dx, dy,
                 head_width=0.02, head_length=0.02, 
                 fc='blue', ec='blue')
    
    ax.set_title(f'Iteration {frame} (Time: {t:.2f})')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    return []

ani = animation.FuncAnimation(fig, update, frames=len(positions_over_time), blit=False)

# Save animation
writer = PillowWriter(fps=10)
ani.save('./coverage_with_dynamics.gif', writer=writer)
plt.show()
plt.close(fig)

print("Animation saved to ./coverage_with_dynamics.gif")
