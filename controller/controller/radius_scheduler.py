import rclpy
import numpy as np
from example_interfaces.msg import Float32MultiArray
from collections import deque


class RadiusScheduler:
    """Minimal Shubert–Piyavskii sampler for r(t) with L = 2 · v_max."""

    def __init__(
        self,
        node: rclpy.node.Node,
        start_pub,  # the existing publisher
        v_max: float,  # worst‑case UAV speed [m/s]
        eps: float = 1.0,  # desired accuracy [m]
    ):
        self.v_max = v_max
        self.node = node
        self.pub = start_pub
        self.L = None
        self.eps = eps
        self.round_id = 0
        self.samples = []
        self.max_radius = 0.0  # highest r(t) seen so far
        self.starting_point = 0.0
        self.r_buffer = deque(maxlen=3)
        self.cancelled = False

    def reset_state(self) -> None:
        """Reset the state for a new round."""
        self.round_id = 0
        self.samples = []
        self.max_radius = 0.0
        self.starting_point = 0.0
        self._pending_t = None  # t where we are waiting for a reply
        self.time_of_reset = self.node.get_clock().now()
        self.ghs_timer = None
        self.cancelled = False

    def roof(self, t: float) -> float:
        if not self.samples:
            return float("inf")
        return min(r_k + self.L * abs(t - t_k) for t_k, r_k in self.samples)

    def next_probe_time(self) -> float | None:
        """
        1. Find τ* where the Lipschitz roof U(τ) attains its global maximum.
        2. If U_max − max_radius ≤ ε  ⇒  finished (return None).
        3. Otherwise return τ* (rounded to the 0.0001 grid) **provided τ* has
        not already been sent or sampled**.
        """
        import numpy as np
        import math

        if len(self.samples) == 0:
            return 0.0  # first ever probe

        used_ts = {t for t, _ in self.samples}
        if self._pending_t is not None:
            used_ts.add(self._pending_t)

        best_U = -math.inf
        probe_t = None

        for t_k, _ in self.samples:
            U_val = self.roof(t_k)
            if U_val > best_U:
                best_U, probe_t = U_val, t_k

        for (t0, r0), (t1, r1) in zip(self.samples, self.samples[1:]):
            if self.L == 0:
                continue
            t_star = 0.5 * (t0 + t1) + (r1 - r0) / (2 * self.L)
            if not (t0 < t_star < t1):
                continue
            U_star = self.roof(t_star)
            if U_star > best_U:
                best_U, probe_t = U_star, t_star

        gap = best_U - self.max_radius
        if gap <= self.eps:
            self.node.get_logger().info(
                f"ε‑optimal reached: max radius = {self.max_radius:.2f} m, "
                f"roof gap = {gap:.3g} m ≤ ε = {self.eps}"
            )
            return None  # done!

        GRID = 0.0001
        TOL = 1e-7
        probe_t = float(np.round(probe_t / GRID) * GRID)  # snap to grid
        if any(abs(probe_t - t) < TOL for t in used_ts):
            # try +0.01 or -0.01
            for delta in (0.01, -0.01):
                candidate = float(np.round((probe_t + delta) / GRID) * GRID)
                if all(abs(candidate - t) > TOL for t in used_ts):
                    probe_t = candidate
                    break

        if probe_t is not None:
            probe_t = float(np.float32(np.round(probe_t, 4)))  # exactly 4 decimals
        return probe_t

    def radius_callback(self, msg: Float32MultiArray) -> None:
        """Callback when `/radius` arrives."""
        r, t_probe = msg.data
        if self._pending_t is not None and abs(t_probe - self._pending_t) < 1e-3:
            # record
            self.samples.append((t_probe, r))
            self.samples.sort(key=lambda x: x[0])
            if r > self.max_radius:
                self.max_radius = r
            self.node.get_logger().warn(
                f"[RS] got r({t_probe:.3f})={r:.3f}; max={self.max_radius:.3f}"
            )
        # check if we already have it
        elif any(abs(t_probe - t) < 1e-9 for t, _ in self.samples):
            self.node.get_logger().warn(
                f"[RS] Duplicate sample received for t={t_probe:.3f}, ignoring."
            )
            return

        self._pending_t = None

    def calculate_connectivity(self, starting_point, longest_path: float) -> float:
        """Run all rounds for this sp, blocking until each /radius arrives."""
        self.reset_state()
        print(f"longest path: {longest_path:.2f} m")
        T_total = longest_path / (self.v_max)
        self.L = 2.0 * self.v_max * T_total  # compensate for longer paths
        self.node.get_logger().error(
            f"[RS] Starting radius calculation with L={self.L:.2f} m, T_total={T_total:.2f} s"
        )

        while True:
            if self.cancelled:
                self.node.get_logger().info("Radius calculation cancelled.")
                break
            # pick next t
            if self.round_id < 2:
                t_probe = float(self.round_id)
            else:
                t_probe = self.next_probe_time()
                if t_probe is None:
                    if len(self.samples) > 10:
                        self.node.get_logger().error(
                            f"!!!Final max radius: {self.max_radius:.2f} m"
                        )
                    else:
                        self.node.get_logger().error("Not enough samples, returning")
                    break

            if self.round_id > 50:
                return self.max_radius

            self.round_id += 1
            self._pending_t = t_probe
            rclpy.spin_once(self.node, timeout_sec=0.1)
            rclpy.spin_once(self.node, timeout_sec=0.0)
            rclpy.spin_once(self.node, timeout_sec=0.0)

            payload = np.concatenate(([t_probe, t_probe], starting_point))
            self.pub.publish(Float32MultiArray(data=payload))
            self.node.get_logger().warn(
                f"[RS] round {self.round_id}: probe t={t_probe:.3f}"
            )
            self.time_of_reset = self.node.get_clock().now()

            # wait for that one reply
            while (
                rclpy.ok()
                and self._pending_t is not None
                and self.time_of_reset + rclpy.duration.Duration(seconds=4.0)
                > self.node.get_clock().now()
            ):
                if self.cancelled:
                    self.node.get_logger().info("Radius calculation cancelled.")
                    break
                rclpy.spin_once(self.node, timeout_sec=0.0)

        return self.max_radius

    def visualize_samples(self) -> None:
        """Visualize the current samples and, optionally, the Lipschitz geometry."""
        import numpy as np
        import matplotlib.pyplot as plt

        if not self.samples:
            self.node.get_logger().warn("No samples to plot yet.")
            return

        t_vals, r_vals = zip(*self.samples)
        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, r_vals, "o-", label="Samples")
        plt.axhline(self.max_radius, ls="--", color="r", label="Max Radius")

        plt.title("Communication Radius r(τ)")
        plt.xlabel("Normalised Time τ")
        plt.ylabel("Radius (m)")
        plt.xlim(0, 1)
        plt.ylim(0, max(r_vals) + 5)
        plt.grid(True, linestyle=":")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("radius_samples.png", dpi=300)
