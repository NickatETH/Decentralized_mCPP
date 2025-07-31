import rclpy
from shapely.ops import split
from shapely.geometry import LineString, Point, Polygon
from example_interfaces.msg import Float32MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as RosPoint
from typing import Tuple, List
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy
import time


color_map = [
    (33 / 255, 92 / 255, 175 / 255),  # ETH Blau
    (0 / 255, 120 / 255, 148 / 255),  # ETH Petrol
    (98 / 255, 115 / 255, 19 / 255),  # ETH Grün
    (142 / 255, 103 / 255, 19 / 255),  # ETH Bronze
    (183 / 255, 53 / 255, 45 / 255),  # ETH Rot
    (167 / 255, 17 / 255, 122 / 255),  # ETH Purpur
    (111 / 255, 111 / 255, 111 / 255),  # ETH Grau
]


qos = QoSProfile(depth=50)
qos.reliability = ReliabilityPolicy.RELIABLE


class PartitionMixin:
    """Mixin class for partitioning functionality."""

    def __init__(self):
        super().__init__()
        self.position = (0.0, 0.0)  # Initial position (x, y)
        self.weight = 0.0
        self.GAMMA = 0.0025
        self.converged = 0.0  # False
        self.TOLERANCE = 0.25
        self.boundary = None
        self.last_call = None

        self.polygon = None  # Polygon of current power cell
        self.neighbours = []  # list of (id, position, weight, converged) tuples

        # Neighbour exchange
        self.nb_pub_ = self.create_publisher(Float32MultiArray, "neighbours", qos)
        self.nb_sub_ = self.create_subscription(
            Float32MultiArray, "neighbours", self.nb_callback, qos
        )
        self.marker_pub = self.create_publisher(Marker, "power_cell_marker", qos)

    def reset_partition(
        self, position: Tuple[float, float], polygon: Polygon, weight: float = None
    ):
        """Reset the partitioning state."""
        self.position = position
        self.polygon = polygon
        self.weight = 0.0 if weight is None else weight
        self.converged = 0.0
        self.neighbours.clear()
        self.tell_neighbours()

    # ------------------------------------------------------------------
    # Neighbour handling
    # ------------------------------------------------------------------
    def tell_neighbours(self, agent_id=None) -> None:
        """Broadcast (id, x, y, weight) to other agents."""
        msg = Float32MultiArray()
        msg.data = [
            float(agent_id) if agent_id is not None else float(self.agent_id),
            float(self.position[0]),
            float(self.position[1]),
            float(self.weight),
            float(self.converged),
        ]
        if rclpy.ok():
            self.nb_pub_.publish(msg)

    def nb_callback(self, msg: Float32MultiArray) -> None:
        """Store neighbour information if the sender is close enough."""
        agent_id = int(msg.data[0])
        if agent_id == self.agent_id:
            return  # ignore our own broadcasts
        if agent_id == -1.0:

            self.tell_neighbours(None)
            return

        position = (msg.data[1], msg.data[2])
        weight = msg.data[3]
        converged = msg.data[4]

        # Only consider neighbours within 1000 m
        if Point(position).distance(Point(self.position)) < 1000.0:
            for i, (nb_id, *_rest) in enumerate(self.neighbours):
                if nb_id == agent_id:
                    self.neighbours[i] = (agent_id, position, weight, converged)
                    break
            else:
                self.neighbours.append((agent_id, position, weight, converged))

    # ------------------------------------------------------------------
    # Power‑cell computation
    # ------------------------------------------------------------------
    def compute_power_cell(self):
        """Compute and store the polygon of this agent's power cell."""
        try:
            if len(self.neighbours) < self.num_agents - 1:
                return None

            # Start with the full domain
            cell = self.boundary
            p_i = self.position
            w_i = self.weight

            for nb_id, p_j, w_j, _ in self.neighbours:
                if nb_id == self.agent_id:
                    continue

                # Weighted bisector: A·q = B
                Ax, Ay = p_j[0] - p_i[0], p_j[1] - p_i[1]
                denom = Ax * Ax + Ay * Ay
                if denom == 0:
                    continue  # identical positions

                B = (
                    p_j[0] ** 2 + p_j[1] ** 2 - p_i[0] ** 2 - p_i[1] ** 2 + w_i - w_j
                ) * 0.5
                px, py = Ax * B / denom, Ay * B / denom
                dx, dy = -Ay, Ax

                extent = (
                    max(
                        self.boundary.bounds[2] - self.boundary.bounds[0],
                        self.boundary.bounds[3] - self.boundary.bounds[1],
                    )
                    * 2
                )
                bisector = LineString(
                    [
                        (px + dx * extent, py + dy * extent),
                        (px - dx * extent, py - dy * extent),
                    ]
                )

                try:
                    parts = split(cell, bisector).geoms
                except ValueError:
                    continue

                for part in parts:
                    q = part.representative_point()
                    if Ax * q.x + Ay * q.y <= B + 1e-9:
                        cell = part
                        break

            # Store result
            self.polygon = cell.intersection(self.boundary)
            self.position = (self.polygon.centroid.x, self.polygon.centroid.y)
            # send neighbour information
            self.tell_neighbours()
            return self.polygon
        except Exception as e:
            self.get_logger().error(f"Error in compute_power_cell: {e}", exc_info=True)

    def balance_power_cells(self) -> None:
        """Adjust agent weights until power‑cell areas equalise within tolerance."""
        # wait for the next 30ms in ros time
        now = self.get_clock().now()
        sleep_ns = 5_000_000 - (now.nanoseconds % 5_000_000)
        if sleep_ns < 5_000_000:
            rclpy.spin_once(self, timeout_sec=sleep_ns / 1e9)
        if self.start_time + Duration(seconds=10.0) < now:
            self.pb_timer.cancel()
            rclpy.spin_once(self, timeout_sec=0.0)
            print("Unstuck agent", self.agent_id)
            return

        cell = self.compute_power_cell()
        if len(self.neighbours) == 0:
            self.get_logger().warn("No neighbours found, cannot balance power cells.")
            now = self.get_clock().now()
            sleep_ns = 50_000_000 - (now.nanoseconds % 50_000_000)
            if sleep_ns < 50_000_000:
                rclpy.spin_once(self, timeout_sec=sleep_ns / 1e9)
            self.tell_neighbours(-1.0)  # Call for re-broadcast
            return

        target_area = self.boundary.area / (len(self.neighbours) + 1.0)

        centroid = (cell.centroid.x, cell.centroid.y)
        self.position = centroid
        self.polygon = cell

        marker = Marker()
        marker.header.frame_id = "map"  # whatever fixed frame you’re using
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"agent_{int(self.agent_id)}"
        marker.id = int(self.agent_id)
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Convert polygon exterior to a list of geometry_msgs/Point
        marker.points = []
        for x, y in zip(self.polygon.exterior.xy[0], self.polygon.exterior.xy[1]):
            pt = RosPoint()
            pt.x = x
            pt.y = y
            pt.z = 0.0
            marker.points.append(pt)
        # close loop
        marker.points.append(marker.points[0])

        marker.scale.x = 0.1  # line width
        # idx = int(self.agent_id) % len(color_map)
        # marker.color.r, marker.color.g, marker.color.b = color_map[idx]
        marker.color.r = 0.5
        marker.color.g = 0.5
        marker.color.b = 0.5
        marker.color.a = 1.0
        marker.lifetime = Duration(seconds=1).to_msg()
        if rclpy.ok():
            self.marker_pub.publish(marker)

        # Check convergence
        error = abs(self.polygon.area - target_area) / target_area

        if error < self.TOLERANCE:
            self.weight = 0.0
            all_converged = all(nb[3] >= 1.0 for nb in self.neighbours)

            if all_converged:
                if self.converged > 10.0 and all(
                    nb[3] >= 10.0 for nb in self.neighbours
                ):
                    self.get_logger().info(f"Agent {self.agent_id} converged forall: ")

                    self.pb_timer.cancel()
                    self.run_stc()
                else:
                    self.converged += 1.0
                    rclpy.spin_once(self, timeout_sec=0.05)
                    # self.get_logger().info(
                    #     f"Agent {self.agent_id} converged: {self.converged:.1f} / 10.0"
                    # )
            else:
                self.converged = 1.0

        else:
            self.converged = 0.0
            self.weight = self.weight - self.GAMMA * (self.polygon.area - target_area)
            self.weight = max(-10, min(self.weight, 10.0))
        rclpy.spin_once(self, timeout_sec=0.0)

        return
