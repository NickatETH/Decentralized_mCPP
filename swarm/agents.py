import numpy as np

def neighbourhood_mask(grid, r, c, radius):
    idx = np.arange(grid)
    RR, CC = np.meshgrid(idx, idx, indexing="ij")
    return (np.abs(RR - r) <= radius) & (np.abs(CC - c) <= radius)

def repulsion_vec(pos, self_rc, rep_R, strength):
    r, c = self_rc
    ry = rx = 0.0
    for orow, ocol in pos:
        dy = r - orow
        dx = c - ocol
        dist2 = dx*dx + dy*dy
        if 0 < dist2 <= rep_R**2:
            ry += dy / (dist2 + 1e-3)
            rx += dx / (dist2 + 1e-3)
    return strength * rx, strength * ry

def step_swarm(agent_pos, density, counts, cfg):
    grid = density.shape[0]
    Rmask = cfg.comm_R
    new_pos = agent_pos.copy()

    rows = np.arange(grid); cols = np.arange(grid)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")

    for k, (r, c) in enumerate(agent_pos):
        mask = neighbourhood_mask(grid, r, c, Rmask)
        deficit = np.maximum(density - counts, 0.0) * mask
        tot = deficit.sum()
        if tot < cfg.eps:
            continue
        g_r = (deficit * RR).sum() / tot
        g_c = (deficit * CC).sum() / tot

        # attraction + repulsion
        rx, ry = repulsion_vec(agent_pos, (r, c), cfg.rep_R, cfg.rep_strength)
        vx = (g_c - c) + rx
        vy = (g_r - r) + ry

        dr, dc = int(np.sign(vy)), int(np.sign(vx))
        nr, nc = np.clip([r + dr, c + dc], 0, grid-1)
        new_pos[k] = [nr, nc]

    return new_pos
