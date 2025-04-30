import numpy as np

SLOPE = 0.5  # Default slope for the inverse_v function
SIGMA_V = 2.0  # Default sigma for the inverse_v function

WIDTH = 5.5  # Default width for the horizontal_band function

RADIUS = 30.0  # Default radius for the circle function
SIGMA_TAIL = 5.0  # Default sigma_tail for the circle function

def inverse_v(grid, apex_row, col_center, slope=SLOPE, sigma=SIGMA_V):
    R, C = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    row_peak = apex_row - slope * np.abs(C - col_center)
    dist = R - row_peak
    d = np.exp(-dist**2 / (2 * sigma**2))
    d[d < 1e-6] = 0
    return d / d.sum()

def horizontal_band(grid, centre, width=WIDTH):
    rows = np.arange(grid)
    band = np.exp(-(rows - centre) ** 2 / (2 * width**2))
    d = np.repeat(band[:, None], grid, axis=1)
    return d / d.sum()


def circle(grid, row_c, col_c, radius=RADIUS, sigma_tail=SIGMA_TAIL):
    rows = np.arange(grid)
    cols = np.arange(grid)
    RR, CC = np.meshgrid(rows, cols, indexing="ij")
    dist = np.sqrt((RR - row_c) ** 2 + (CC - col_c) ** 2)
    d = np.zeros_like(dist, dtype=float)
    inside = dist <= radius
    d[inside] = 1.0
    tail = np.exp(-((dist - radius) ** 2) / (2 * sigma_tail**2))
    d += tail * (~inside)
    return d
