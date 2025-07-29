import os
import subprocess
from typing import Dict

import rclpy
import numpy as np
from rclpy.node import Node
from interface.srv import ComputeEnergy
from std_msgs.msg import Float64MultiArray, Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from visualization_msgs.msg import Marker  # Add this import
from radius_scheduler import RadiusScheduler
from thompson_scheduler import ThompsonScheduler


AGENT_IDS = [1, 2, 3, 4]

AGENT_POSITIONS = {
    1: (0.0, 0.0),
    2: (50.0, 0.0),
    3: (50.0, 50.0),
    4: (5.0, 40.0),
}

BOUNDARY_ARGS = [
    "--xmin",
    "0.0",
    "--ymin",
    "0.0",
    "--xmax",
    "50.0",
    "--ymax",
    "50.0",
]

# QoS shortcuts --------------------------------------------------------------
qos_best_effort = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
qos_reliable_vol = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
qos_reliable_tx = QoSProfile(
    depth=2,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


def launch_agent(aid: int, x: float, y: float) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "drone",
        "uav_agent",
        "--agent_id",
        str(aid),
        "--num_agents",
        str(len(AGENT_IDS)),
        "--posx",
        str(x),
        "--posy",
        str(y),
        *BOUNDARY_ARGS,
    ]
    return subprocess.Popen(cmd, preexec_fn=os.setsid)  # POSIX


class Controller(Node):
    def __init__(self) -> None:
        super().__init__("controller")
        self.start_pub = self.create_publisher(
            Float64MultiArray, "/start_ghs", qos_reliable_tx
        )
        self.radius_marker_pub = self.create_publisher(
            Marker, "/comm_radius_marker", qos_reliable_tx
        )

        self.bo_result_pub = self.create_publisher(
            Float64MultiArray, "/bo_results", qos_reliable_vol
        )
        self.bo_result_sub = self.create_subscription(
            Float64MultiArray, "/bo_results", self._on_bo_result, qos_reliable_vol
        )

        self.kill_agent_pub = self.create_publisher(
            Float64MultiArray, "/kill_agents", qos_reliable_vol
        )

        self.reset_agent_pub = self.create_publisher(
            Float64MultiArray, "/reset_agents", qos_reliable_vol
        )

        self.radius_scheduler = RadiusScheduler(
            self, self.start_pub, v_max=10.0, eps=0.01
        )
        self.create_subscription(
            Float64MultiArray,
            "/radius",
            self.radius_scheduler.radius_callback,
            qos_reliable_vol,
        )

        self.thompson_scheduler = ThompsonScheduler()

        self.energy_clients_dict: Dict[int, rclpy.client.Client] = {}

        self.cruise_speed = 10.0
        self.a = 1.0
        self.b = 1.0

        for aid in AGENT_IDS:
            x, y = AGENT_POSITIONS[aid]
            proc = launch_agent(aid, x, y)
            self.get_logger().info(f"Launched uav_agent_{aid} (pid {proc.pid})")

        for aid in AGENT_IDS:
            srv = f"/uav_agent_{aid}/compute_energy"
            cli = self.create_client(ComputeEnergy, srv)
            self.energy_clients_dict[aid] = cli
            self.get_logger().info(f"Waiting for {srv} …")
            while not cli.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn(f"{srv} not up yet, waiting…")

    def calculate_energy(self, aid: int) -> None:
        """
        Fire off all UAV energy requests in parallel and return the sum.
        """
        futures = []
        for cli in self.energy_clients_dict.values():
            req = ComputeEnergy.Request()
            req.cruise_speed = self.cruise_speed
            req.a = self.a
            req.b = self.b
            futures.append(cli.call_async(req))

        while rclpy.ok() and not all(f.done() for f in futures):
            rclpy.spin_once(self)

        total = 0.0
        for f in futures:
            try:
                total += f.result().energy
            except Exception:
                total += float("inf")
        return total

    def reset_agents(self) -> None:
        self.reset_agent_pub.publish(Empty())

    def shutdown_agents(self) -> None:
        self.kill_agent_pub.publish(Empty())


class RadiusScheduler:
    """Minimal Shubert–Piyavskii sampler for r(t) with L = 2 · v_max."""

    def __init__(
        self,
        node: rclpy.node.Node,
        start_pub,  # the existing publisher
        v_max: float,  # worst‑case UAV speed [m/s]
        eps: float = 2.0,
    ):  # desired accuracy   [m]
        self._node = node
        self._pub = start_pub
        self._L = 2.0 * v_max
        self._eps = eps
        self._round_id = 0
        # (t,r) pairs already measured; endpoints hold +∞ until sampled
        self._samples = [(0.0, 0.0), (1.0, 0.0)]
        self.max_radius = 0.0  # highest r(t) seen so far

    # ------------------------------------------------------------------ helpers
    def _roof(self, t: float) -> float:
        """Upper envelope U(t) = min_k ( r_k + L |t - t_k| )."""
        return min(r + self._L * abs(t - tk) for tk, r in self._samples)

    def _next_probe_time(self) -> float | None:
        """Return the t where the gap between roof and chord is biggest."""
        best_gap, best_t = 0.0, None
        for (t0, r0), (t1, r1) in zip(self._samples, self._samples[1:]):
            t_mid = 0.5 * (t0 + t1)
            gap = self._roof(t_mid) - 0.5 * (r0 + r1)
            if gap > best_gap:
                best_gap, best_t = gap, t_mid
        if best_gap < self._eps:
            self._node.get_logger().info(
                f"Final max radius: {self.max_radius:.2f} m, gap = {best_gap:.2f} m"
            )
            self._node.com_ok = True
            return None  # no more probes needed
        return best_t

    # ------------------------------------------------------------------ public
    def maybe_request_probe(self) -> None:
        """Call from a timer (e.g. every 0.2 s).  Starts a new round if needed."""
        if self._round_id < 2:  # first two rounds are special
            t_probe = float(self._round_id)

        else:
            t_probe = self._next_probe_time()
            if t_probe is None:  # envelope tight enough
                return

        self._round_id += 1

        # Example: starting_points indexed by agent id (1-based)
        starting_points = [0.25, 0.5, 0.75, 1.0]  # index 0 unused

        # Broadcast the start message to all agents
        payload = [t_probe, t_probe] + starting_points
        self._pub.publish(Float64MultiArray(data=payload))

        self._node.get_logger().info(
            f"GHS round {self._round_id} @ t={t_probe:.2} and sp={starting_points} rreq"
        )

    def store_radius(self, r: float, rid: float) -> None:
        """Callback when `/radius` arrives from the swarm root."""
        for i, (t, _) in enumerate(self._samples):
            if abs(t - rid) < 1e-8:
                self._samples[i] = (rid, r)
                break
        else:
            self._samples.append((rid, r))
        self._samples.sort(key=lambda x: x[0])
        if r > self.max_radius:
            self.max_radius = r
        self._node.get_logger().info(
            f"Response: r(t={rid:.2}) = {r:.2f} m  "
            f"(current max {self.max_radius:.2f} m)"
        )
        # print(f"Current samples: {[(round(t, 2), round(r, 2)) for t, r in self._samples]}")

    def run_bayes_opt(self):
        while True:
            rclpy.spin_once(self)
            pt = self.thompson_scheduler.next_point()
            if pt is None:

                break
            seed, starting_point = pt

            max_radius = self.radius_scheduler.calculate_connectivity(starting_point)
            total_energy = self.calculate_energy(seed)

            self.reset_agents()

            self.publish_bo_result(
                seed=seed, sp=starting_point, r=max_radius, E=total_energy
            )

        self.get_logger().info("BO complete ‐ shutting down agents")
        self.total_energy = 0.0
        self.shutdown_agents()

    def publish_bo_result(self, seed: float, sp: float, r: float, E: float) -> None:
        msg = Float64MultiArray()
        msg.data = [seed, sp, r, E]
        self.bo_result_pub.publish(msg)

    def bo_result_callback(self, msg: Float64MultiArray) -> None:
        seed, sp, r, E = msg.data
        self.thompson_scheduler.observe(seed, sp, r, E)

    def visualize_samples(self) -> None:
        """Visualize the current samples as a plot."""
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        t_values = [t for t, r in self._samples]
        r_values = [r for t, r in self._samples]
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


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Controller()
    try:
        node.run_bayes_opt()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
