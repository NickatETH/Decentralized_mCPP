import numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from . import density as D
from scipy.ndimage import gaussian_filter
from .agents import step_swarm

class Cfg:  # tiny struct for defaults
    def __init__(self, args):
        self.grid       = args.grid
        self.num_agents = args.num
        self.steps      = args.steps
        self.comm_R     = 20
        self.rep_strength = 5.0
        self.eps        = 1e-9
        self.kill_int   = args.kill if args.kill>0 else None
        self.use_V      = args.inverseV
        self.use_circle = args.circle
        self.radius     = 30.0       # starting circle radius
        self.sigma_tail = 6.0        # tail fall‐off
        self.kappa      = 0.9        # radius adaptation gain
        self.foot_R     = 5  
        self.rep_R      = self.foot_R * 1.5

        self.cells_per_drone = (2*self.foot_R + 1)**2 



def neighbourhood_mask(grid, r, c, radius):
    idx = np.arange(grid)
    RR, CC = np.meshgrid(idx, idx, indexing="ij")
    return (np.abs(RR - r) <= radius) & (np.abs(CC - c) <= radius)

def run_simulation(args):
    cfg = Cfg(args)
    g   = cfg.grid
    centre_col = (g-1)/2
    
    
    # choose density generator
    def make_density(frame):
        apex = 2 + frame*(g-4-2)/(cfg.steps-1)
        if cfg.use_V:
            return D.inverse_v(g, apex, centre_col)
        if cfg.use_circle:
            return D.circle(
                grid        = g,
                row_c       = apex,
                col_c       = centre_col,
                radius      = cfg.radius,       # <-- live, adaptive value
                sigma_tail  = cfg.sigma_tail
            )
        return D.horizontal_band(g, apex)

    # initial positions (lower half)
    rows = np.random.randint(0, g//2, cfg.num_agents)
    cols = np.random.randint(0, g,     cfg.num_agents)
    agent_pos = np.vstack([rows, cols]).T

    # Matplotlib
    fig, ax = plt.subplots(figsize=(10,10))
    density = make_density(0)
    img  = ax.imshow(density, origin="lower")
    scat = ax.scatter(agent_pos[:,1], agent_pos[:,0], c="yellow", edgecolor="k")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-.5, g-.5); ax.set_ylim(-.5, g-.5)

    circle_patches = []
    for _ in range(cfg.num_agents):
        circ = plt.Circle((0, 0), cfg.comm_R,
                        edgecolor="cyan", facecolor="none",
                        linewidth=1.5, alpha=0.3)
        ax.add_patch(circ)
        circle_patches.append(circ)

    def update(frame):
        nonlocal agent_pos, density, circle_patches

        if cfg.kill_int and frame > 0 and frame % cfg.kill_int == 0 and len(agent_pos) > 1:
            kill = np.random.choice(len(agent_pos))
            agent_pos = np.delete(agent_pos, kill, axis=0)
            circle_patches[kill].remove()
            circle_patches.pop(kill)
            scat.set_offsets(agent_pos[:, [1, 0]])

        apex_row = 2 + frame*(g-4-2)/(cfg.steps-1)
        
        if cfg.use_circle:
            mask = neighbourhood_mask(g, apex_row, centre_col, radius=int(round(cfg.radius)))
            M_star = density[mask].sum()
            dists = np.linalg.norm(
                agent_pos - np.array([apex_row, centre_col]), axis=1)
            M = (dists <= cfg.radius).sum() * cfg.cells_per_drone   

        density = make_density(frame)
        counts = np.zeros((g, g), int)
        fR = cfg.foot_R          # e.g. 2 → 5×5 square
        for r, c in agent_pos:
            counts[max(0,r-fR): min(g,r+fR+1),
                max(0,c-fR): min(g,c+fR+1)] += 1
        agent_pos = step_swarm(agent_pos, density, counts / cfg.num_agents, cfg)

        # ---- update artists ----------------------------------------------------
        img.set_data(density)
        scat.set_offsets(agent_pos[:, [1, 0]])

        # move the circles
        for circ, (r, c) in zip(circle_patches, agent_pos):
            circ.center = (c, r)          # (x, y) = (col, row)

        ax.set_title(f"frame {frame}")
        return [img, scat, *circle_patches]

    anim = FuncAnimation(fig, update, frames=cfg.steps, interval=120, blit=True)
    plt.show()
