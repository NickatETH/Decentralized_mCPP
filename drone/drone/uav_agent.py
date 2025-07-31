import sys
import argparse
from typing import List, Tuple


import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as RosPoint
from example_interfaces.msg import Empty, Float32MultiArray

from shapely.geometry import Polygon


from .radius_calculation import RadiusMixin
from .energy_calculation import EnergyMixin
from .stc import STCMixin
from .partition import PartitionMixin

from interface.srv import ComputeEnergy


Coordinate = Tuple[float, float]
Grid = List[Coordinate]

qos_reliable_vol = QoSProfile(
    depth=20,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


color_map = [
    (33 / 255, 92 / 255, 175 / 255),  # ETH Blau
    (0 / 255, 120 / 255, 148 / 255),  # ETH Petrol
    (98 / 255, 115 / 255, 19 / 255),  # ETH Grün
    (142 / 255, 103 / 255, 19 / 255),  # ETH Bronze
    (183 / 255, 53 / 255, 45 / 255),  # ETH Rot
    (167 / 255, 17 / 255, 122 / 255),  # ETH Purpur
    (111 / 255, 111 / 255, 111 / 255),  # ETH Grau
]


class UavAgent(Node, RadiusMixin, EnergyMixin, STCMixin, PartitionMixin):
    """ROS 2 node representing a UAV that exchanges position/weight with
    neighbours and computes its weighted Voronoi (power) cell.
    """

    def __init__(self, agent_id, num_agents, boundary):
        Node.__init__(self, f"uav_agent_{int(agent_id)}")  # <-- Only Node gets the name
        RadiusMixin.__init__(self)
        STCMixin.__init__(self)
        PartitionMixin.__init__(self)
        EnergyMixin.__init__(self)

        self.agent_id = agent_id
        self.num_agents = num_agents
        self.position = None  # (x, y)
        self.boundary = boundary  # Polygon defining the boundary of the area
        self.path = None
        self.shutdown = False  # Initialize shutdown as a boolean

        self.shutdown_sub = self.create_subscription(
            Empty, "/shutdown", self.shutdown_callback, 2
        )

        self.reset_sub = self.create_subscription(
            Float32MultiArray,
            "/reset_agents",
            self.reset_position_callback,
            qos_reliable_vol,
        )
        self.reset_pub = self.create_publisher(
            Float32MultiArray, "/reset_agents", qos_reliable_vol
        )

        self.prep_radiusmixin(agent_id)  # Initialize radius mixin
        self.pb_timer = None
        self.get_logger().info(f"UAV agent {self.agent_id} initialized ")
        for i in range(5):
            self.reset_pub.publish(
                Float32MultiArray(data=[-(1.0), self.agent_id, 0.0, 0.0])
            )
            rclpy.spin_once(self, timeout_sec=0.5)
            

    def reset_position_callback(self, msg: Float32MultiArray) -> None:
        """Reset the UAV's position and partition state if the message targets this agent."""
        if len(msg.data) < 3:
            self.get_logger().warn("Reset message too short, expected 3 floats.")
            return
        target_id = msg.data[0]
        if target_id != self.agent_id:
            return  # Ignore if not for this agent
        position = (msg.data[2], msg.data[3])
        self.position = position
        self.path = None
        # self.reset_partition(position=position, polygon=self.boundary)
        self.get_logger().info(f"UAV {self.agent_id} reset to position {position}")
        self.pb_timer = self.create_timer(0.1, self.balance_power_cells)

        # pub ok
        self.reset_pub.publish(
            Float32MultiArray(
                data=[-(msg.data[1]), self.agent_id, position[0], position[1]]
            )
        )

    def run_stc(self):
        """Compute the Spanning-Tree Coverage (STC) path and publish it."""
        print(f"Running STC for agent {self.agent_id}")
        cell_size = 2.0
        grid = self.polygon_to_grid(self.polygon, cell_size)
        stc_path = self.compute_stc(grid)
        offset_path = self.offset_stc_path(stc_path, cell_size)

        self.path = offset_path  # Store the path for later use
        self.get_logger().info(f"Computed STC path: points")

        self._energy_srv = self.create_service(
            ComputeEnergy, "~/compute_energy", self.compute_energy_cb
        )

        # Publish the path as a marker
        marker = Marker()
        marker.header.frame_id = "map"  # whatever fixed frame you’re using
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"agent_{int(self.agent_id)}"
        marker.id = int(self.agent_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.points = []
        for x, y in zip(offset_path.xy[0], offset_path.xy[1]):
            pt = RosPoint()
            pt.x = x
            pt.y = y
            pt.z = 0.0
            marker.points.append(pt)
        # close loop
        marker.points.append(marker.points[0])
        marker.scale.x = 0.1  # line width
        # Assign color based on agent_id for visual distinction
        idx = int(self.agent_id) % len(color_map)
        marker.color.r, marker.color.g, marker.color.b = color_map[idx]
        marker.color.a = 1.0
        if rclpy.ok():
            self.marker_pub.publish(marker)

    def shutdown_callback(self, msg: Empty) -> None:
        """Handle shutdown signal."""
        self.get_logger().info("Received shutdown signal, shutting down agent.")
        self.shutdown = True


# ------------------------------------------------------------------
# Command‑line entry point
# ------------------------------------------------------------------


def _parse_cli(argv):
    """Parse CLI args, returning (parsed_namespace, ros_args)."""
    parser = argparse.ArgumentParser(description="UAV agent node")
    parser.add_argument("--agent_id", type=float, required=True, help="Unique agent ID")
    parser.add_argument(
        "--num_agents", type=int, required=True, help="Total number of agents"
    )
    parser.add_argument("--xmin", type=float, default=-50.0, help="Boundary min x")
    parser.add_argument("--ymin", type=float, default=-50.0, help="Boundary min y")
    parser.add_argument("--xmax", type=float, default=50.0, help="Boundary max x")
    parser.add_argument("--ymax", type=float, default=50.0, help="Boundary max y")
    return parser.parse_known_args(argv[1:])


def main(argv=None):
    if argv is None:
        argv = sys.argv

    # Parse our custom args first; leave the rest for rclpy
    cli_args, ros_args = _parse_cli(argv)

    # Initialise ROS with remaining arguments
    rclpy.init(args=ros_args)

    # Build boundary polygon (simple rectangle)
    boundary = Polygon(
        [
            (cli_args.xmin, cli_args.ymin),
            (cli_args.xmax, cli_args.ymin),
            (cli_args.xmax, cli_args.ymax),
            (cli_args.xmin, cli_args.ymax),
        ]
    )

    # Instantiate and spin the agent
    agent = UavAgent(
        agent_id=cli_args.agent_id,
        num_agents=cli_args.num_agents,
        boundary=boundary,
    )

    try:
        while not agent.shutdown:
            rclpy.spin_once(agent, timeout_sec=0.0)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        # Normal termination triggered by Ctrl‑C or external shutdown
        pass
    finally:
        if agent.shutdown:
            print("Shutting down UAV agent due to received shutdown signal.")
        else:
            print("Shutting down UAV agent due to external interrupt or error.")
        print("Shutting down UAV agent...:", rclpy.ok(), agent.shutdown)
        agent.destroy_node()
        if rclpy.ok():  # avoid double shutdown
            rclpy.shutdown()


if __name__ == "__main__":
    main()
