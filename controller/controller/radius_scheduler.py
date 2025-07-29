import rclpy
from std_msgs.msg import Float64MultiArray


class RadiusScheduler:
    """Minimal Shubert–Piyavskii sampler for r(t) with L = 2 · v_max."""

    def __init__(
        self,
        node: rclpy.node.Node,
        start_pub,  # the existing publisher
        v_max: float,  # worst‑case UAV speed [m/s]
        eps: float = 2.0,  # desired accuracy [m]
    ):

        self.node = node
        self.pub = start_pub
        self.L = 2.0 * v_max
        self.eps = eps
        self.round_id = 0
        self.samples = [
            (0.0, 0.0),
            (1.0, 0.0),
        ]  # (t,r) pairs already measured; endpoints hold +∞ until sampled
        self.max_radius = 0.0  # highest r(t) seen so far
        self.starting_point = 0.0

    def reset_state(self) -> None:
        """Reset the state for a new round."""
        self.round_id = 0
        self.samples = [(0.0, 0.0), (1.0, 0.0)]
        self.max_radius = 0.0
        self.starting_point = 0.0
        self._pending_t = None  # t where we are waiting for a reply

    def roof(self, t: float) -> float:
        """Upper envelope U(t) = min_k ( r_k + L |t - t_k| )."""
        return min(r + self.L * abs(t - tk) for tk, r in self.samples)

    def request_probe(self) -> None:
        """Starts a new round if needed."""
        if self.round_id < 2:  # first two rounds are special
            t_probe = float(self.round_id)

        else:
            t_probe = self.next_probe_time()
            if t_probe is None:  # envelope tight enough
                return

        self.round_id += 1

        # Broadcast the start message to all agents
        payload = [t_probe, t_probe] + [self.starting_point]
        self.pub.publish(Float64MultiArray(data=payload))

        self.node.get_logger().info(
            f"GHS round {self.round_id} @ t={t_probe:.2} and sp={self.starting_point} rreq"
        )

    def radius_callback(self, msg: Float64MultiArray) -> None:
        """Callback when `/radius` arrives."""
        print(f"Received radius callback: {msg.data}")
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
            # clear the flag
            self._pending_t = None

    def calculate_connectivity(self, starting_point: float) -> float:
        """Run all rounds for this sp, blocking until each /radius arrives."""
        self.reset_state()
        self.starting_point = starting_point

        while True:
            # pick next t
            if self.round_id < 2:
                t_probe = float(self.round_id)
            else:
                t_probe = self.next_probe_time()
                if t_probe is None:
                    break

            self.round_id += 1
            self._pending_t = t_probe

            payload = [t_probe, t_probe] + [starting_point]
            self.pub.publish(Float64MultiArray(data=payload))
            self.node.get_logger().info(
                f"[RS] round {self.round_id}: probe t={t_probe:.3f}, sp={starting_point:.3f}"
            )

            # wait for that one reply
            while rclpy.ok() and self._pending_t is not None:
                rclpy.spin_once(self.node)

        return self.max_radius
