from interface.srv import ComputeEnergy  # Custom service for energy computation
import math
from typing import Tuple


class EnergyMixin:
    def __init__(self):
        super().__init__()
        self._energy_srv = None
        self.path = []

    def _distance(self, p: Tuple[float, float], q: Tuple[float, float]) -> float:
        """Euclidean distance between p and q."""
        return math.dist(p, q)

    def _angle_between(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Return the unsigned angle (rad) between two 2‑D vectors v1 and v2.
        Z is ignored (assumes fairly level flight)."""
        x1, y1, _ = v1
        x2, y2, _ = v2
        dot = x1 * x2 + y1 * y2
        norm_prod = math.hypot(x1, y1) * math.hypot(x2, y2)
        if norm_prod == 0.0:
            return 0.0
        # Clamp to protect against numerical error
        cos_theta = max(-1.0, min(1.0, dot / norm_prod))
        return math.acos(cos_theta)

    def compute_energy_cb(self, request, response) -> float:
        """Return the total energy (J) required to fly one lap of *ring* at a
        constant *cruise_speed* using the simple A/B quad‑rotor model.
        """
        if request.cruise_speed <= 0.0:
            raise ValueError("cruise_speed must be > 0")
        if self.path == [] or self.path is None:
            response.energy = -1.0
            return response
        if len(self.path.coords) < 2:
            raise ValueError("ring must contain at least two vertices")

        coords = list(self.path.coords)

        # Build a constant speed list matching segment count
        speeds = [request.cruise_speed] * (len(coords) - 1)

        energy = 0.0
        distance = 0.0

        # --- Straight‑segment contribution ----------------------------------------
        for i, speed in enumerate(speeds):
            drag = request.b * speed**2
            thrust = request.a / (speed**2)
            power = 300
            turnfactor = 1.0

            dist = self._distance(coords[i], coords[i + 1])
            time = dist / speed

            # Turns
            v_in = (
                coords[i][0] - coords[i - 1][0],
                coords[i][1] - coords[i - 1][1],
                0.0,
            )
            v_out = (
                coords[i + 1][0] - coords[i][0],
                coords[i + 1][1] - coords[i][1],
                0.0,
            )
            angle = self._angle_between(v_in, v_out)
            if angle > math.radians(5):  # significant turn threshold
                turnfactor = 1.5

            energy += power * time * turnfactor
            distance += dist

        response.energy = energy
        response.path_length = distance

        return response
