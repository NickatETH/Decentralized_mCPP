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
from .radius_scheduler import RadiusScheduler
from .thompson_scheduler import ThompsonScheduler
import random

NUM_AGENTS = 5
NUM_CANDIDATES = 1000
lambda_BO = 1.0
MAX_EVALS = 1000

AGENT_IDS = list(range(1, NUM_AGENTS + 1))
AGENT_POSITIONS = {
    aid: (random.uniform(0, 50), random.uniform(0, 50)) for aid in AGENT_IDS
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


def launch_agent(aid: int) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "drone",
        "uav_agent",
        "--agent_id",
        str(aid),
        "--num_agents",
        str(len(AGENT_IDS)),
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
            Float64MultiArray, "/bo_results", self.bo_result_callback, qos_reliable_vol
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

        args = dict(zip(BOUNDARY_ARGS[::2], BOUNDARY_ARGS[1::2]))
        xmin, ymin = float(args["--xmin"]), float(args["--ymin"])
        xmax, ymax = float(args["--xmax"]), float(args["--ymax"])

        # Build random joint candidate set of shape (NUM_CANDIDATES, 3*NUM_AGENTS)
        seed_x = np.random.uniform(xmin, xmax, 
                                   size=(NUM_CANDIDATES, NUM_AGENTS))
        seed_y = np.random.uniform(ymin, ymax, 
                                   size=(NUM_CANDIDATES, NUM_AGENTS))
        sps    = np.random.uniform(0.0, 1.0, 
                                   size=(NUM_CANDIDATES, NUM_AGENTS))
        candidate_set = np.hstack([seed_x, seed_y, sps])

        # Instantiate your BO sampler
        self.thompson_scheduler = ThompsonScheduler(
            candidate_set=candidate_set,
            lambda_BO=lambda_BO,
            max_evals=MAX_EVALS,
        )

        self.energy_clients_dict: Dict[int, rclpy.client.Client] = {}

        self.cruise_speed = 10.0
        self.a = 1.0
        self.b = 1.0

        for aid in AGENT_IDS:
            proc = launch_agent(aid)
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
            rclpy.spin_once(self, timeout_sec=0.0)

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

    def run_bayes_opt(self):
        while True:
            rclpy.spin_once(self, timeout_sec=0.0)
            x = self.thompson_scheduler.next_point()
            if x is None:
                break
            
            n = NUM_AGENTS
            seed_xs = x[0:   n]
            seed_ys = x[n: 2*n]
            sps     = x[2*n:3*n]

            for aid in AGENT_IDS:
                msg = Float64MultiArray(data=[
                    float(aid),
                    float(seed_xs[aid-1]),
                    float(seed_ys[aid-1]),
                ])
            self.reset_agent_pub.publish(msg)

            max_radius = self.radius_scheduler.calculate_connectivity(sps)
            total_energy = self.calculate_energy()

            out = Float64MultiArray()
            out.data = x.tolist() + [max_radius, total_energy]
            self.bo_result_pub.publish(out)

        self.get_logger().info("BO complete ‐ shutting down agents")
        self.total_energy = 0.0
        self.shutdown_agents()

    def bo_result_callback(self, msg: Float64MultiArray) -> None:
            data = msg.data
            n = NUM_AGENTS
            x = np.array(data[0:3*n])         # the joint (seed_x, seed_y, sp) vector
            r = data[3*n]                     # max_radius
            E = data[3*n + 1]                 # total_energy
            cost = r + lambda_BO * E
            self.thompson_scheduler.observe(x, cost)
            self.get_logger().info(f"[BO] obs recorded, cost={cost:.3f}")

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
