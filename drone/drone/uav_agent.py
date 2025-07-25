import sys
import argparse

import rclpy
import time
from rclpy.node import Node
from rclpy.duration import Duration

from std_msgs.msg import String, Float32MultiArray
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split
import matplotlib.pyplot as plt  # for plotting power cell polygons
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as RosPoint


class UavAgent(Node):
    """ROS 2 node representing a UAV that exchanges position/weight with
    neighbours and computes its weighted Voronoi (power) cell.

    Supply *agent_id*, *position* and *boundary* at launch time; only bug
    fixes and the requested renaming were applied – existing logic is
    untouched.
    """

    def __init__(self, agent_id, position, boundary):
        super().__init__(f"uav_agent_{int(agent_id)}")

        # An unused String publisher kept to preserve original behaviour.
        self.publisher_ = self.create_publisher(String, "topic", 10)

        # Parameters / state
        self.agent_id = agent_id
        self.num_agents = 4
        self.position = position  # (x, y)
        self.weight = 1.0
        self.GAMMA = 0.001
        self.converged = 0.0 # False
        self.TOLERANCE = 0.05
        self.boundary = boundary

        self.polygon = None  # Polygon of current power cell
        self.neighbours = []  # list of (id, position, weight, converged) tuples

        # Neighbour exchange
        self.nb_pub_ = self.create_publisher(Float32MultiArray, "neighbours", 10)
        self.nb_sub_ = self.create_subscription(
            Float32MultiArray, "neighbours", self.nb_callback, 10
        )
        self.marker_pub = self.create_publisher(Marker, 'power_cell_marker', 10)

        self.ask_for_neighbours(response=1.0)  # Initial broadcast
        self.pb_timer = self.create_timer(0.01, self.run_loop)



    def run_loop(self):
        """Run the agent's main loop, asking for neighbours and balancing power cells."""
        self.ask_for_neighbours(response=1.0)  # Ask for neighbours
        self.get_clock().sleep_for(Duration(seconds=0.01))
        self.balance_power_cells()
 

    # ------------------------------------------------------------------
    # Neighbour handling
    # ------------------------------------------------------------------
    def ask_for_neighbours(self, response:float):
        """Broadcast (id, x, y, weight) to other agents."""
        msg = Float32MultiArray()
        msg.data = [
            float(self.agent_id),
            float(self.position[0]),
            float(self.position[1]),
            float(self.weight),
            float(self.converged), 
            1.0                         # I want a response
        ]
        self.nb_pub_.publish(msg)
 

    def nb_callback(self, msg):
        """Store neighbour information if the sender is close enough."""
        agent_id = int(msg.data[0])
        if agent_id == self.agent_id:
            return  # ignore our own broadcasts

        position = (msg.data[1], msg.data[2])
        weight = msg.data[3]
        converged = msg.data[4]
        response = msg.data[5] 

        # Only consider neighbours within 1000 m
        if Point(position).distance(Point(self.position)) < 1000.0:
            if response == 1.0:
                self.ask_for_neighbours(response=0.0)  # Respond to the neighbour
            # Replace existing entry or append a new one
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
                    p_j[0]**2 + p_j[1]**2
                    - p_i[0]**2 - p_i[1]**2
                    + w_i - w_j
                ) * 0.5
                px, py = Ax * B / denom, Ay * B / denom
                dx, dy = -Ay, Ax

                extent = max(
                    self.boundary.bounds[2] - self.boundary.bounds[0],
                    self.boundary.bounds[3] - self.boundary.bounds[1],
                ) * 2
                bisector = LineString([
                    (px + dx * extent, py + dy * extent),
                    (px - dx * extent, py - dy * extent),
                ])

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
            return self.polygon
        except Exception as e:
            self.get_logger().error(
                f"Error in compute_power_cell: {e}",
                exc_info=True
            )


    def balance_power_cells(self) -> None:
        """Adjust agent weights until power‑cell areas equalise within tolerance."""
        cell = self.compute_power_cell()
        if (len(self.neighbours) == 0):
            self.get_logger().warn("No neighbours found, cannot balance power cells.")
            return
        if cell is None or cell.is_empty:
            self.get_logger().warn("Power cell is empty or not computed.")
            return

        target_area = self.boundary.area / (len(self.neighbours) + 1.0) 

        centroid = (cell.centroid.x, cell.centroid.y)
        self.position = centroid
        self.polygon = cell

        
        marker = Marker()
        marker.header.frame_id = 'map'          # whatever fixed frame you’re using
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'agent_{int(self.agent_id)}'
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

        marker.scale.x = 0.05     # line width
        marker.color.r = 0.1
        marker.color.g = 0.7
        marker.color.b = 0.1
        marker.color.a = 1.0
        self.marker_pub.publish(marker)
        

        # Check convergence
        error = abs(self.polygon.area - target_area) / target_area

        if error < self.TOLERANCE:
            # self.get_logger().info(
            #     f"Converged! Neighbour convergence states: "
            #     f"{[(nb[0], nb[3]) for nb in self.neighbours]}"
            # )
            self.weight = 1.0

            all_converged = all(nb[3] >= 1.0 for nb in self.neighbours)


            if all_converged:
                if self.converged > 10.0 and all(nb[3] >= 10.0 for nb in self.neighbours):
                    self.get_logger().info(f"Agent {self.agent_id} converged forall: Position: {self.position},  Area: {self.polygon.area}")
  
                    self.pb_timer.cancel()
                else:
                    self.converged += 1.0
            else:
                self.converged = 1.0

            
        else:
            self.converged = 0.0
            self.weight = self.weight - self.GAMMA * (self.polygon.area - target_area)
            self.weight = max(-50, min(self.weight, 50.0))

        return



            



# ------------------------------------------------------------------
# Command‑line entry point
# ------------------------------------------------------------------

def _parse_cli(argv):
    """Parse CLI args, returning (parsed_namespace, ros_args)."""
    parser = argparse.ArgumentParser(description="UAV agent node")
    parser.add_argument("--agent_id", type=float, required=True, help="Unique agent ID")
    parser.add_argument("--posx", type=float, required=True, help="Initial x position")
    parser.add_argument("--posy", type=float, required=True, help="Initial y position")
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
        position=(cli_args.posx, cli_args.posy),
        boundary=boundary,
    )

    try:
        rclpy.spin(agent)
    finally:
        agent.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
