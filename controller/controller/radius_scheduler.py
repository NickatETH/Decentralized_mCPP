import rclpy
from std_msgs.msg import Float64MultiArray


class RadiusScheduler:
    """Minimal Shubert–Piyavskii sampler for r(t) with L = 2 · v_max."""

    def __init__(
        self,
        node: rclpy.node.Node,
        start_pub,  # the existing publisher
        v_max: float,  # worst‑case UAV speed [m/s]
        eps: float = 0.2,  # desired accuracy [m]
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

    def reset_state(self) -> None:
        """Reset the state for a new round."""
        self.round_id = 0
        self.samples = []
        self.max_radius = 0.0
        self.starting_point = 0.0
        self._pending_t = None  # t where we are waiting for a reply
        self.time_of_reset = self.node.get_clock().now()
        self.ghs_timer = None

    def roof(self, t: float) -> float:
        """Upper envelope U(t) = min_k ( r_k + L * |t - t_k| ).

        `self.L` must already be expressed for the normalised domain [0,1].
        """
        return min(r_k + self.L * abs(t - t_k) for t_k, r_k in self.samples)

    def next_probe_time(self) -> float | None:
        """Return the parameter t where the gap between roof and chord is largest.

        Returns `None` when either:
        * there are fewer than two samples, or
        * the maximum gap is ≤ self.eps (estimation complete).
        """
        if len(self.samples) < 2:
            return None

        best_gap = 0.0
        best_t: float | None = None

        for (t0, r0), (t1, r1) in zip(self.samples, self.samples[1:]):
            t_mid = 0.5 * (t0 + t1)
            chord = 0.5 * (r0 + r1)
            gap = self.roof(t_mid) - chord
            if gap > best_gap:
                best_gap, best_t = gap, t_mid

        if best_gap <= self.eps:
            self.node.get_logger().info(
                f"Final max radius: {self.max_radius:.2f} m, gap = {best_gap:.3g} m"
            )
            return None

        return best_t

    def radius_callback(self, msg: Float64MultiArray) -> None:
        """Callback when `/radius` arrives."""
        r, t_probe = msg.data

        if self._pending_t is not None and abs(t_probe - self._pending_t) < 1e-9:
            # record
            self.samples.append((t_probe, r))
            self.samples.sort(key=lambda x: x[0])
            if r > self.max_radius:
                self.max_radius = r
            self.node.get_logger().info(
                f"[RS] got r({t_probe:.3f})={r:.3f}; max={self.max_radius:.3f}"
            )
        self._pending_t = None

    def calculate_connectivity(
        self, starting_point: float, longest_path: float
    ) -> float:
        """Run all rounds for this sp, blocking until each /radius arrives."""
        self.reset_state()
        self.L = 2.0 * self.v_max * longest_path  # compensate for longer pahts
        self.starting_point = starting_point

        while True:
            # pick next t
            if self.round_id < 2:
                t_probe = float(self.round_id)
            else:
                t_probe = self.next_probe_time()
                if t_probe is None:
                    self.node.get_logger().error(
                        f"!!!Final max radius: {self.max_radius:.2f} m"
                    )
                    self.visualize_samples()
                    break

            self.round_id += 1
            self._pending_t = t_probe
            rclpy.spin_once(self.node, timeout_sec=0.1)
            rclpy.spin_once(self.node, timeout_sec=0.1)

            payload = [t_probe, t_probe] + [0.25, 0.4, 0.5, 0.6]
            self.pub.publish(Float64MultiArray(data=payload))
            self.node.get_logger().warn(
                f"[RS] round {self.round_id}: probe t={t_probe:.3f}, sp={starting_point:.3f}"
            )
            self.time_of_reset = self.node.get_clock().now()

            # wait for that one reply
            while (
                rclpy.ok()
                and self._pending_t is not None
                and self.time_of_reset + rclpy.duration.Duration(seconds=2.0)
                > self.node.get_clock().now()
            ):
                rclpy.spin_once(self.node, timeout_sec=0.0)

        return self.max_radius

    def visualize_samples(self) -> None:
        """Visualize the current samples as a plot."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        t_values = [t for t, r in self.samples]
        r_values = [r for t, r in self.samples]
        plt.figure(figsize=(10, 6))
        plt.plot(t_values, r_values, marker="o", label="Samples")
        plt.title("Radius Samples Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Radius (m)")
        plt.grid()
        plt.legend()
        plt.xlim(0, 1)
        plt.ylim(0, max(r_values) + 5)
        plt.axhline(y=self.max_radius, color="r", linestyle="--", label="Max Radius")
        plt.legend()
        plt.show()
