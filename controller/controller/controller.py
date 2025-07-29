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

qos_reliable_tx = QoSProfile(
    depth=2,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)
qos_reliable_vol = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

def launch_agent(aid: int, x: float, y: float) -> subprocess.Popen:
    cmd = [
        "ros2",
        "run",
        "drone",
        "uav_agent",
        "--agent_id",
        str(aid),
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

        # add kill agent publisher here
        self.kill_agent_pub = self.create_publisher(
            Float64MultiArray, "/kill_agents", qos_reliable_vol
        )

        # add reset agent publisher here
        self.reset_agent_pub = self.create_publisher(
            Float64MultiArray, "/reset_agents", qos_reliable_vol
        )

        self.radius_scheduler = RadiusScheduler(self, self.start_pub, v_max=10.0, eps=0.01)
        self.create_subscription(
            Float64MultiArray, "/radius", self.radius_scheduler.radius_callback, qos_reliable_vol
        )

        self.thompson_scheduler = ThompsonScheduler()

        self.energy_clients_dict: Dict[int, rclpy.client.Client] = {}

        # request constants
        self.cruise_speed = 10.0
        self.a = 1.0
        self.b = 1.0

        # launch agents
        for aid in AGENT_IDS:
            x, y = AGENT_POSITIONS[aid]
            proc = launch_agent(aid, x, y)
            self.get_logger().info(f"Launched uav_agent_{aid} (pid {proc.pid})")

        # create service clients and send first request
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
        # 1) launch all calls
        futures = []
        for cli in self.energy_clients_dict.values():
            req = ComputeEnergy.Request()
            req.cruise_speed    = self.cruise_speed
            req.a               = self.a
            req.b               = self.b
            futures.append(cli.call_async(req))

        # 2) wait for all to finish
        while rclpy.ok() and not all(f.done() for f in futures):
            rclpy.spin_once(self)

        # 3) collect and sum
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
            rclpy.spin_once(self)
            pt = self.thompson_scheduler.next_point()
            if pt is None:
                break
            seed, starting_point = pt

            max_radius = self.radius_scheduler.calculate_connectivity(starting_point)
            total_energy = self.calculate_energy(seed)

            self.reset_agents()

            self.publish_bo_result(
                seed=seed,
                sp=starting_point,
                r=max_radius,
                E=total_energy
            )

        self.get_logger().info("BO complete ‐ shutting down agents")
        self.total_energy = 0.0
        self.shutdown_agents()

    def publish_bo_result(self,
                           seed: float,
                           sp: float,
                           r: float,
                           E: float) -> None:
        msg = Float64MultiArray()
        msg.data = [seed, sp, r, E]
        self.bo_result_pub.publish(msg)

    def bo_result_callback(self, msg: Float64MultiArray) -> None:
        seed, sp, r, E = msg.data
        self.thompson_scheduler.observe(seed, sp, r, E)

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