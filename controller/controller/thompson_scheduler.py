# thompson_scheduler.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

l = [
    5.0, 5.0, 5.0, 5.0,
    5.0, 5.0, 5.0, 5.0,
    0.2,  0.2,  0.2,  0.2,
]


class ThompsonScheduler:
    """
    Asynchronous 2-D Thompson sampler over (seed, start_pos) ∈ [s_min,s_max]×[0,1].
    On each observe(), fits a GP to past (X,y), draws one sample on a grid, returns argmax.
    """

    def __init__(
        self,
        candidate_set: np.ndarray,   # shape=(M, 3*NUM_AGENTS)
        lambda_BO: float = 1.0,
        max_evals: int = 1000,
    ):
        self.cand      = candidate_set
        self.lambda_BO = lambda_BO
        self.max_evals = max_evals
        self.evals     = 0

        # GP with Matern + white noise
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(length_scale=l, length_scale_bounds=[(1e-2, 1e2)], nu=1.5) \
               + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10)

        # History of (x_vector, cost)
        self.X: list[np.ndarray] = []
        self.y: list[float]    = []

    def next_point(self) -> np.ndarray | None:
        """Return one 3n-vector x, or None when budget exhausted."""
        if self.evals >= self.max_evals:
            return None
        self.evals += 1

        # Cold start: just return the first few candidates uniquely
        if len(self.X) < len(self.cand):
            return self.cand[len(self.X)]

        # Fit GP and draw one Thompson sample over ALL M candidates
        Xarr = np.vstack(self.X)
        yarr = np.array(self.y)
        self.gp.fit(Xarr, yarr)

        mu, cov = self.gp.predict(self.cand, return_cov=True)
        sample  = np.random.multivariate_normal(mu, cov)
        best_i  = int(np.argmin(sample))
        return self.cand[best_i]

    def observe(self, x: np.ndarray, cost: float):
        """Record one joint observation."""
        self.X.append(x)
        self.y.append(cost)
