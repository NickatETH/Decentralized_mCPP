# thompson_scheduler.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


class ThompsonScheduler:
    """
    Asynchronous 2-D Thompson sampler over (seed, start_pos) ∈ [s_min,s_max]×[0,1].
    On each observe(), fits a GP to past (X,y), draws one sample on a grid, returns argmax.
    """

    def __init__(
        self,
        seed_bounds: tuple[float, float] = (0.0, 1.0),
        sp_bounds: tuple[float, float] = (0.0, 1.0),
        λ: float = 1.0,
        grid_size: int = 50,
        max_evals: int = 30,
    ):
        self.λ = λ
        self.max_evals = max_evals
        self.evals = 0

        # 2-D grid of candidate (seed, sp) pairs
        seeds = np.linspace(seed_bounds[0], seed_bounds[1], grid_size)
        sps = np.linspace(sp_bounds[0], sp_bounds[1], grid_size)
        self.grid = np.stack(np.meshgrid(seeds, sps), -1).reshape(-1, 2)

        # GP with Matern + white noise
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=[0.2, 0.2], nu=2.5
        ) + WhiteKernel(noise_level=1e-3)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)

        # History
        self.X: list[list[float, float]] = []
        self.y: list[float] = []

    def next_point(self) -> tuple[float, float] | None:
        """Return next (seed,sp) or None if done."""
        # Seed with corners if too few data:
        if self.evals < 2:
            pt = [(self.grid[0], self.grid[-1])[self.evals]]
            self.evals += 1
            return pt

        # Fit GP on (X,y)
        X = np.array(self.X)
        y = np.array(self.y)
        self.gp.fit(X, y)

        # Thompson sample on the grid
        mu, cov = self.gp.predict(self.grid, return_cov=True)
        sample = np.random.multivariate_normal(mu, cov)
        idx = np.argmax(sample)
        self.evals += 1
        return tuple(self.grid[idx])

    def observe(self, seed: float, sp: float, r: float, E: float):
        """Feed back one observation."""
        cost = r + self.λ * E
        print(f"[BO] obs: seed={seed:.3f}, sp={sp:.3f} → cost={cost:.3f}")
        self.X.append([seed, sp])
        self.y.append(cost)
