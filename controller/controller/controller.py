import os
import subprocess
from typing import Dict

import rclpy
import numpy as np
import matplotlib.pyplot as plt
from rclpy.node import Node
from interface.srv import ComputeEnergy
from example_interfaces.msg import Empty, Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from visualization_msgs.msg import Marker  # Add this import
from .radius_scheduler import RadiusScheduler
from .thompson_scheduler import ThompsonScheduler
import random
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import time

NUM_AGENTS = 4
NUM_CANDIDATES = 1000
RANDOM_SEED = 42
lambda_BO = 1.0
MAX_EVALS = 50
PATH_SCALE = 100.0

np.random.seed(RANDOM_SEED)

AGENT_IDS = list(range(1, NUM_AGENTS + 1))
AGENT_POSITIONS = {
    aid: (random.uniform(0, 20), random.uniform(0, 20)) for aid in AGENT_IDS
}

BOUNDARY_ARGS = [
    "--xmin",
    "0.0",
    "--ymin",
    "0.0",
    "--xmax",
    "20.0",
    "--ymax",
    "20.0",
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

        self.cost_history = []

        self.start_pub = self.create_publisher(
            Float32MultiArray, "/start_ghs", qos_reliable_vol
        )
        self.radius_marker_pub = self.create_publisher(
            Marker, "/comm_radius_marker", qos_reliable_tx
        )

        self.bo_result_pub = self.create_publisher(
            Float32MultiArray, "/bo_results", qos_reliable_vol
        )
        self.bo_result_sub = self.create_subscription(
            Float32MultiArray, "/bo_results", self.bo_result_callback, qos_reliable_vol
        )

        self.kill_agent_pub = self.create_publisher(
            Float32MultiArray, "/kill_agents", qos_reliable_vol
        )

        self.reset_agent_pub = self.create_publisher(
            Float32MultiArray, "/reset_agents", qos_reliable_vol
        )

        self.reset_agents_sub = self.create_subscription(
            Float32MultiArray,
            "/reset_agents",
            self.reset_position_callback,
            qos_reliable_vol,
        )
        self.reset_response = [False] * len(AGENT_IDS)

        self.shutdown_sub = self.create_subscription(
            Empty, "/shutdown", self.shutdown_callback, 1
        )
        self.shutdown = False  # Initialize shutdown as a boolean

        self.radius_scheduler = RadiusScheduler(
            self, self.start_pub, v_max=10.0, eps=0.05
        )
        self.create_subscription(
            Float32MultiArray,
            "/radius",
            self.radius_scheduler.radius_callback,
            qos_reliable_vol,
        )

        args = dict(zip(BOUNDARY_ARGS[::2], BOUNDARY_ARGS[1::2]))
        xmin, ymin = float(args["--xmin"]), float(args["--ymin"])
        xmax, ymax = float(args["--xmax"]), float(args["--ymax"])

        # Build random joint candidate set of shape (NUM_CANDIDATES, 3*NUM_AGENTS)
        seed_x = np.random.uniform(xmin, xmax, size=(NUM_CANDIDATES, NUM_AGENTS))
        seed_y = np.random.uniform(ymin, ymax, size=(NUM_CANDIDATES, NUM_AGENTS))
        sps = np.random.uniform(0.0, 1.0, size=(NUM_CANDIDATES, NUM_AGENTS))
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
        while not all(self.reset_response):
            print(self.reset_response)
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("All agents launched and reset.")
        time.sleep(1.0)
        for i in range(50):
            rclpy.spin_once(self, timeout_sec=0.01)
        self.reset_agents()
        while not all(self.reset_response):
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().info("All agents reset/INIT successfully.")

        for aid in AGENT_IDS:
            srv = f"/uav_agent_{aid}/compute_energy"
            cli = self.create_client(ComputeEnergy, srv)
            self.energy_clients_dict[aid] = cli
            self.get_logger().info(f"Waiting for {srv} …")
            while not cli.wait_for_service(timeout_sec=0.1):
                self.get_logger().warn(f"{srv} not up yet, waiting…")
                rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("All energy clients initialized.")

    def reset_position_callback(self, msg: Float32MultiArray) -> None:
        """Reset the UAV's position and partition state if the message targets this agent."""
        if len(msg.data) < 3:
            self.get_logger().warn("Reset message too short, expected 3 floats.")
            return
        if msg.data[0] != -(1.0):
            self.get_logger().warn(
                f"Reset message not for this agent: {msg.data[0]} != -1.0"
            )
            return
        self.reset_response[int(msg.data[1] - 1.0)] = True

    def calculate_energy(self):
        """
        Fire off all UAV energy requests in parallel and return the sum.
        """
        total = 0.0
        now = self.get_clock().now()
        longest_path = 0.0
        for aid, cli in self.energy_clients_dict.items():
            while True:
                rclpy.spin_once(self, timeout_sec=0.8)
                req = ComputeEnergy.Request()
                req.cruise_speed = self.cruise_speed
                req.a = self.a
                req.b = self.b
                future = cli.call_async(req)
                while rclpy.ok() and not future.done():
                    rclpy.spin_once(self, timeout_sec=0.8)
                try:
                    energy = future.result().energy
                    if energy == -1.0:
                        continue
                    total += energy
                    if longest_path < future.result().path_length:
                        longest_path = future.result().path_length
                    break
                except Exception:
                    total += float("inf")
                    break
        self.get_logger().info(f"Total energy for all agents: {total:.2f} J")
        return total, longest_path

    def reset_agents(self) -> None:
        self.reset_response = [False] * len(AGENT_IDS)
        for aid in AGENT_IDS:
            x, y = AGENT_POSITIONS[aid]
            self.reset_agent_pub.publish(Float32MultiArray(data=[aid, 1.0, x, y]))

    def eval_iteration(self, sps):
        total_energy, longest_path = self.calculate_energy()
        longest_path *= PATH_SCALE
        max_radius = self.radius_scheduler.calculate_connectivity(sps, longest_path)
        return max_radius, total_energy

    def run_bayes_opt(self):
        executor = ThreadPoolExecutor(max_workers=1)
        while True:
            self.get_logger().info("another BO round!")
            rclpy.spin_once(self, timeout_sec=0.0)
            x = self.thompson_scheduler.next_point()
            if x is None:
                break

            for i in x:
                i = float(np.float32(np.round(i, 4)))

            n = NUM_AGENTS
            seed_xs = x[0:n]
            seed_ys = x[n : 2 * n]
            sps = x[2 * n : 3 * n]
            self.get_logger().info(
                f"BO round with seeds: {seed_xs}, {seed_ys}, sps: {sps}"
            )

            for aid in AGENT_IDS:
                msg = Float32MultiArray(
                    data=[
                        float(aid),
                        1.0,
                        float(seed_xs[aid - 1]),
                        float(seed_ys[aid - 1]),
                    ]
                )
                self.reset_agent_pub.publish(msg)

            future = executor.submit(self.eval_iteration, sps)

            try:
                max_radius, total_energy = future.result(timeout=90.0)
            except TimeoutError:
                self.radius_scheduler.cancelled = True
                rclpy.spin_once(self, timeout_sec=0.1)
                self.get_logger().error("BO evaluation timed out, skipping this point.")
                time.sleep(1.0)
                continue

            self.cost_history.append((total_energy * lambda_BO + max_radius))

            out = Float32MultiArray()
            out.data = x.tolist() + [max_radius, total_energy]
            self.bo_result_pub.publish(out)
            self.plot_convergence()

        self.get_logger().info("BO complete ‐ shutting down agents")
        self.total_energy = 0.0
        self.shutdown_agents()

    def bo_result_callback(self, msg: Float32MultiArray) -> None:
        print(f"Publishing BO result: see")
        data = msg.data
        x = np.array(data[:-2], dtype=float)
        max_radius, total_energy = data[-2], data[-1]
        self.thompson_scheduler.observe(x, total_energy * lambda_BO + max_radius)

    def shutdown_callback(self, msg: Empty) -> None:
        """Handle shutdown signal."""
        self.get_logger().info("Received shutdown signal, shutting down controller.")
        self.shutdown = True

    def plot_convergence(self) -> None:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import MaxNLocator

        costs = np.array(self.cost_history)
        best = np.minimum.accumulate(costs)
        iterations = np.arange(1, len(costs) + 1)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(iterations, costs, marker="o", linestyle="--", label="Observed cost")
        ax.plot(iterations, best, marker="s", linestyle="-", label="Best so far")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost = r + λ·E")
        ax.set_title("Bayesian Optimization Convergence")

        # ensure only integer ticks
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.tight_layout()
        fig.savefig("bo_convergence.png", dpi=300)
        plt.close(fig)


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
