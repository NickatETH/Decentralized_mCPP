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

        self.shutdown_sub = self.create_subscription(
            Empty, "/shutdown", self.shutdown_callback, 1
        )
        self.shutdown = False  # Initialize shutdown as a boolean

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
        self.get_logger().info("All energy clients initialized.")

    def _on_bo_result(self, msg: Float64MultiArray) -> None:
        self.get_logger().info("Received BO result")

    def calculate_energy(self, aid: int) -> None:
        """
        Fire off all UAV energy requests in parallel and return the sum.
        """

        total = 0.0
        for aid, cli in self.energy_clients_dict.items():
            while True:
                req = ComputeEnergy.Request()
                req.cruise_speed = self.cruise_speed
                req.power_straight = self.a
                req.power_turn = self.b
                future = cli.call_async(req)
                while rclpy.ok() and not future.done():
                    rclpy.spin_once(self)
                try:
                    energy = future.result().energy
                    if energy == -1.0:
                        self.get_logger().warn(f"Agent {aid} not ready, retrying...")
                        continue
                    total += energy
                    break
                except Exception:
                    total += float("inf")
                    break
        self.get_logger().info(f"Total energy for all agents: {total:.2f} J")
        return total

    def reset_agents(self) -> None:
        self.reset_agent_pub.publish(Empty())

    def run_bayes_opt(self):
        while not self.shutdown:
            rclpy.spin_once(self, timeout_sec=0.0)
            self.get_logger().info("Running Bayesian Optimization")
            pt = self.thompson_scheduler.next_point()
            if pt is None:

                break
            seed, starting_point = pt

            self.get_logger().info(
                f"Running BO with seed {seed} and starting point {starting_point}"
            )
            total_energy = self.calculate_energy(seed)
            max_radius = self.radius_scheduler.calculate_connectivity(starting_point)

            self.reset_agents()

            self.publish_bo_result(
                seed=seed, sp=starting_point, r=max_radius, E=total_energy
            )

        self.get_logger().info("BO complete ‐ shutting down agents")
        self.total_energy = 0.0

    def publish_bo_result(self, seed: float, sp: float, r: float, E: float) -> None:
        msg = Float64MultiArray()
        msg.data = [seed, sp, r, E]
        self.bo_result_pub.publish(msg)

    def bo_result_callback(self, msg: Float64MultiArray) -> None:
        print(f"Publishing BO result: see")
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

    def shutdown_callback(self, msg: Empty) -> None:
        """Handle shutdown signal."""
        self.get_logger().info("Received shutdown signal, shutting down controller.")
        self.shutdown = True
        self.destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Controller()
    try:
        print("Starting Bayesianff Optimization loop...")
        node.run_bayes_opt()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
