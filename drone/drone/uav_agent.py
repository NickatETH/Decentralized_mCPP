import sys
import math
import argparse
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError        

from std_msgs.msg import String, Float32MultiArray, Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point as RosPoint

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import split

import matplotlib.pyplot as plt  # for plotting power cell polygons

from interface.srv import ComputeEnergy  # Custom service for energy computation

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped

Coordinate = Tuple[float, float]
Grid = List[Coordinate]

_DIRS: Tuple[Coordinate, ...] = (
    (2, 0),  # East
    (-2, 0), # West
    (0, 2),  # North
    (0, -2), # South
)

color_map = [
    (33/255, 92/255, 175/255),    # ETH Blau
    (0/255, 120/255, 148/255),    # ETH Petrol
    (98/255, 115/255, 19/255),    # ETH Grün
    (142/255, 103/255, 19/255),   # ETH Bronze
    (183/255, 53/255, 45/255),    # ETH Rot
    (167/255, 17/255, 122/255),   # ETH Purpur
    (111/255, 111/255, 111/255),  # ETH Grau
]

# QoS shortcuts --------------------------------------------------------------
qos_best_effort  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)
qos_reliable_vol = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
qos_reliable_tx  = QoSProfile(depth=20,  reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)

GHS_TEST, GHS_ACCEPT, GHS_REJECT, GHS_REPORT = 0, 1, 2, 3

class UavAgent(Node):
    """ROS 2 node representing a UAV that exchanges position/weight with
    neighbours and computes its weighted Voronoi (power) cell.

    Supply *agent_id*, *position* and *boundary* at launch time; only bug
    fixes and the requested renaming were applied – existing logic is
    untouched.
    """

    def __init__(self, agent_id, position, boundary):
        super().__init__(f"uav_agent_{int(agent_id)}")
        self.stop_all = False  # Flag to stop the agent's main loop

        # An unused String publisher kept to preserve original behaviour.
        self.publisher_ = self.create_publisher(String, "topic", 10)

        # Parameters / state
        self.agent_id = agent_id
        self.num_agents = 4
        self.position = position  # (x, y)
        self.weight = 0.0
        self.GAMMA = 0.001
        self.converged = 0.0 # False
        self.TOLERANCE = 0.05
        self.boundary = boundary

        self.polygon = None  # Polygon of current power cell
        self.neighbours = []  # list of (id, position, weight, converged) tuples

        self.path = []  # Path for STC computation

        self.radius_pos = (0.0, 0.0) 
        self.starting_point = 0.0
        self.nbr_table = {}  
        self.frag = FragmentState(self.agent_id)  # reset fragment state

        # Neighbour exchange
        self.nb_pub_ = self.create_publisher(Float32MultiArray, "neighbours", 10)
        self.nb_sub_ = self.create_subscription(
            Float32MultiArray, "neighbours", self.nb_callback, 10
        )
        self.marker_pub = self.create_publisher(Marker, 'power_cell_marker', 10)

        self.ask_for_neighbours(response=1.0)  # Initial broadcast
        self.pb_timer = self.create_timer(0.01, self.run_loop)

        self._energy_srv = self.create_service(
            ComputeEnergy,'~/compute_energyy',         
            self.compute_energy_cb        )

        self.beacon_pub = self.create_publisher(PointStamped, "/beacon", qos_best_effort)
        self.ghs_pub    = self.create_publisher(Float32MultiArray, "/ghs_packet", qos_reliable_tx)
        self.radius_pub = self.create_publisher(Float64MultiArray, "/radius", qos_reliable_vol)
        self.start_sub  = self.create_subscription(Float64MultiArray, "/start_ghs", self.start_cb, qos_reliable_tx)
        self.packet_sub = self.create_subscription(Float32MultiArray, "/ghs_packet", self.ghs_cb, qos_reliable_tx)
        self.beacon_sub = self.create_subscription(PointStamped, "/beacon", self.beacon_cb, qos_best_effort)

        self.ghs_timer = None  # Timer for GHS probe (started later)


    def run_loop(self):
        """Run the agent's main loop, asking for neighbours and balancing power cells."""
        self.ask_for_neighbours(response=1.0)  # Ask for neighbours
        self.get_clock().sleep_for(Duration(seconds=0.05))
        self.balance_power_cells()

    def run_stc(self):
        """Compute the Spanning-Tree Coverage (STC) path and publish it."""
        cell_size = 2.0
        grid = self.polygon_to_grid(self.polygon, cell_size)
        stc_path = self.compute_stc(grid)
        offset_path = self.offset_stc_path(stc_path, cell_size)
        self.path = offset_path  # Store the path for later use
        self.get_logger().info(f"Computed STC path: {len(offset_path.coords)} points")
        # Publish the path as a marker
        marker = Marker()
        marker.header.frame_id = 'map'  # whatever fixed frame you’re using
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f'agent_{int(self.agent_id)}'
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
        marker.scale.x = 0.1     # line width
        # Assign color based on agent_id for visual distinction
        idx = int(self.agent_id) % len(color_map)
        marker.color.r, marker.color.g, marker.color.b = color_map[idx]
        marker.color.a = 1.0
        if rclpy.ok():
            self.marker_pub.publish(marker)
 

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
        if rclpy.ok():
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
        idx = int(self.agent_id) % len(color_map)
        marker.color.r, marker.color.g, marker.color.b = color_map[idx]
        marker.color.a = 1.0
        if rclpy.ok():
            self.marker_pub.publish(marker)
        

        # Check convergence
        error = abs(self.polygon.area - target_area) / target_area

        if error < self.TOLERANCE:
            self.weight = 1.0
            all_converged = all(nb[3] >= 1.0 for nb in self.neighbours)


            if all_converged:
                if self.converged > 10.0 and all(nb[3] >= 10.0 for nb in self.neighbours):
                    self.get_logger().info(f"Agent {self.agent_id} converged forall: ")
  
                    self.pb_timer.cancel()
                    self.run_stc()
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
# Path Generation
# ------------------------------------------------------------------

    def polygon_to_grid(self, cell_poly: Polygon, cell_size: float) -> Grid:
        """Rasterise *cell_poly* to the centres of a square grid.
        """
        assert not cell_poly.is_empty, "cell_poly must not be empty"

        half = 0.5 * cell_size
        minx, miny, maxx, maxy = cell_poly.bounds

        ix_min = int(np.ceil((minx - half) / cell_size))
        iy_min = int(np.ceil((miny - half) / cell_size))
        ix_max = int(np.floor((maxx - half) / cell_size))
        iy_max = int(np.floor((maxy - half) / cell_size))

        inside: Grid = []
        for ix in range(ix_min, ix_max + 1):
            x = (ix + 0.5) * cell_size
            for iy in range(iy_min, iy_max + 1):
                y = (iy + 0.5) * cell_size
                if cell_poly.contains(Point(x, y)):
                    inside.append((x, y))

        inside.sort(key=lambda p: (p[1], p[0]))  # reproducible ordering
        assert inside, "No grid cells found inside the polygon"
        return inside

    def _adjacency(self, cells: Iterable[Coordinate]) -> Dict[Coordinate, List[Coordinate]]:
        """Return the 4‑neighbour adjacency list of *cells*."""
        cell_set = set(cells)
        nbrs: Dict[Coordinate, List[Coordinate]] = defaultdict(list)

        for x, y in cell_set:
            for dx, dy in _DIRS:
                n = (x + dx, y + dy)
                if n in cell_set:
                    nbrs[(x, y)].append(n)

        return nbrs

    def compute_stc(self, cells: Grid) -> Grid:
        """Compute a deterministic **Spanning‑Tree Coverage (STC)** tour over *cells*.

        The depth‑first traversal starts from the lexicographically smallest cell and
        records every entry into a cell as well as the corresponding back‑track. The
        resulting tour therefore visits some cells multiple times but ensures full
        coverage while remaining simple to execute.
        """
        if not cells:
            return []

        root = min(cells, key=lambda p: (p[1], p[0]))
        nbrs = self._adjacency(cells)

        visited = {root}
        tour: Grid = []

        def dfs(u: Coordinate) -> None:
            tour.append(u)  # enter cell
            for v in sorted(nbrs[u], key=lambda p: (p[1], p[0])):
                if v not in visited:
                    visited.add(v)
                    dfs(v)
                    tour.append(u)  # back‑track

        dfs(root)
        return tour

    def offset_stc_path(self, stc_path: Grid, cell_size: float) -> Grid:
        """Return a smooth, closed offset path surrounding *stc_path*.

        The path is obtained by buffering the open STC polyline by a quarter of the
        grid cell size using round caps and joins.
        """
        radius_factor = 0.25  # quarter of the cell size
        resolution = 4  # number of segments per quarter circle
        if not stc_path or len(stc_path) < 2:
            self.get_logger().warn("STC path is empty or has less than 2 points")
            return []

        r = cell_size * radius_factor
        line = LineString(stc_path)  # keep path open – round caps form at both ends

        poly = line.buffer(r, join_style=1, cap_style=1, resolution=resolution)
        poly = poly.buffer(r*0.99, join_style=1, cap_style=1, resolution=resolution)
        poly = poly.buffer(-r*1.0, join_style=1, cap_style=1, resolution=resolution)

        # Exterior ring traces the final closed loop
        ring = poly.exterior
        return ring

# ------------------------------------------------------------------
# Energy computation
# ------------------------------------------------------------------
    def _distance(self, p: Tuple[float, float], q: Tuple[float, float]) -> float:
        """Euclidean distance between p and q."""
        return math.dist(p, q)

    def _angle_between(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """Return the unsigned angle (rad) between two 2‑D vectors v1 and v2.
        Z is ignored (assumes fairly level flight)."""
        x1, y1, _ = v1
        x2, y2, _ = v2
        dot = x1 * x2 + y1 * y2
        norm_prod = math.hypot(x1, y1) * math.hypot(x2, y2)
        if norm_prod == 0.0:
            return 0.0
        # Clamp to protect against numerical error
        cos_theta = max(-1.0, min(1.0, dot / norm_prod))
        return math.acos(cos_theta)

    def compute_energy_cb(self, request, response) -> float:
        """Return the total energy (J) required to fly one lap of *ring* at a
        constant *cruise_speed* using the simple A/B quad‑rotor model.
        """
        if request.cruise_speed <= 0.0:
            raise ValueError("cruise_speed must be > 0")
        if self.path == []: 
            response.energy = -1.0
            return response
        if len(self.path.coords) < 2:
            raise ValueError("ring must contain at least two vertices")

        # Pre‑compute numeric coordinate list ( not required for energy)
        coords = [p for p in self.path.coords]
        # Offset the coords order by starting_point index
        if isinstance(request.starting_point, int):
            start_idx = request.starting_point % len(coords)
        else:
            # Find the closest point index to the given starting_point (float, as x)
            dists = [math.hypot(p[0] - request.starting_point, p[1]) for p in coords]
            start_idx = dists.index(min(dists))
        coords = coords[start_idx:] + coords[:start_idx]

        # Build a constant speed list matching segment count
        speeds = [request.cruise_speed] * (len(coords) - 1)

        energy = 0.0

        # --- Straight‑segment contribution ----------------------------------------
        for i, speed in enumerate(speeds):
            drag = request.b * speed ** 2
            thrust = request.a / (speed ** 2)
            power = drag + thrust
            turnfactor = 1.0

            dist = self._distance(coords[i], coords[i + 1])
            time = dist / speed

            # Turns
            v_in = (coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1], 0.0)
            v_out = (coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1], 0.0)
            angle = self._angle_between(v_in, v_out)
            if angle > math.radians(5):  # significant turn threshold
                turnfactor = 1.5

            energy += power * time * turnfactor

        response.energy = energy

        return response
    

    # ------------------------------------------------------------------
    # Beacon I/O
    # ------------------------------------------------------------------
    def send_beacon(self):
        x, y = self.radius_pos
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.agent_id)
        msg.point.x, msg.point.y = x, y
        msg.point.z = float(self.frag.frag_id)
        self.beacon_pub.publish(msg)

    def beacon_cb(self, msg: PointStamped):
        uid = float(msg.header.frame_id)
        if uid != self.agent_id:
            self.nbr_table[uid] = (msg.point.x, msg.point.y, msg.point.z)



    # ------------------------------------------------------------------
    # Start‑GHS handler
    # ------------------------------------------------------------------
    def start_cb(self, msg: Float64MultiArray):
        """
        Expects a Float64MultiArray whose length is a multiple of 4:
            [t0, aid0, rid, sp0,   t0, aid1, rid, sp1,   ...]
        Only the tuple addressed to *this* agent is processed.
        """
        # print(f"Starting GHS for agent {self.agent_id} ")
        print("\n \n \n \n \n")
        data = msg.data
        if len(data) < 2:
            self.get_logger().error("start_cb: message too short")
            return

        t_target = data[0]
        rid      = data[1]

        idx      = int(1 + self.agent_id)   
        sp       = data[idx]
        self.starting_point = sp

        path_idx = int((t_target + sp) * (len(self.path.coords) - 1) % len(self.path.coords))
        self.radius_pos = self.path.coords[path_idx]

        self.send_beacon()  # send initial beacon
        self.frag = FragmentState(self.agent_id)  # reset fragment state
        self.get_clock().sleep_for(Duration(seconds=0.1))
        self.ghs_timer = self.create_timer(0.1,lambda: self._report_best(rid))


    def _compute_best_out(self):
        # 1) collect all neighbours except your own fragment
        neighbours = []
        for nbr_uid, nbr_p in self.nbr_table.items():
            x, y, nbr_fid = nbr_p
            if nbr_fid == self.frag.frag_id or nbr_uid == self.agent_id:
                continue
            dist = math.hypot(x - self.radius_pos[0], y - self.radius_pos[1])
            neighbours.append((nbr_uid, nbr_fid, float(dist)))

        # 2) sort by distance (smallest first)
        neighbours.sort(key=lambda pair: pair[2])

        # 3) skip as many as we've already iterated
        idx = self.frag.nbr_iter
        if idx < len(neighbours):
            # 4) pick the “next best” un‐tried neighbour
            self.frag.best_out = neighbours[idx]
        else:
            print(f"Agent {self.agent_id} has no more neighbours to try: {neighbours}")
            self.frag.nbr_iter = 0  # reset neighbour iteration

    def _report_best(self, rid: float):
        """Report the best candidate to the root of the fragment."""
        self._compute_best_out()
        if self.frag.best_out is not None:
            nbr, fid, w = self.frag.best_out
            self.frag.candidate = (nbr,fid, w)  # store candidate
            pkt = Float32MultiArray(data=[3, rid, nbr, self.frag.root, fid, self.frag.level, w])
            self.ghs_pub.publish(pkt)

    # ------------------------------------------------------------------
    # GHS packet handler (Test/Accept/Reject/Report)
    # ------------------------------------------------------------------
    def ghs_cb(self, msg: Float32MultiArray):
        if self.stop_all:
            return
        typ, rid, src, dst, fid, lvl, weight, *children = msg.data
        if dst != self.agent_id:
            return # ignore packets not addressed to me
        if typ == 0:            # TEST
            ans_typ = 0      
            if self.agent_id == self.frag.frag_id:
                # I'm the root of my fragment, so I can accept the test
                if self.frag.best_out is not None and self.frag.best_out[1] == fid:
                    # build a flat list of child‐UIDs as floats
                    children_ids = list(self.frag.children)     
                    children_uids = [float(uid) for uid in children_ids]    
                    base_data = [1.0, rid, self.agent_id,
                        src, self.frag.frag_id, self.frag.level, weight]               

                    pkt_data = base_data + children_uids
                    pkt = Float32MultiArray(data=pkt_data)
                    # self.get_logger().info(f"Agent {self.agent_id} ACCEPTED TEST from {src} ")

                elif self.frag.best_out is not None:
                    # self.get_logger().info(f"Agent {self.agent_id} REJECTED TEST from {src} for {self.frag.best_out[0]}")
                    pkt = Float32MultiArray(data=[2, rid, self.agent_id, src, self.frag.frag_id, self.frag.level, weight])
                else:
                    print(f"Agent {self.agent_id} has no best_out candidate to accept/reject.")
                    return
                self.ghs_pub.publish(pkt)

            else:
                # Ask root in the name of the other agent
                pkt = Float32MultiArray(data=[0, rid, src, self.frag.root, fid, lvl, weight])
                self.ghs_pub.publish(pkt)
                # print(f"Agent {self.agent_id} forwarded TEST from {src} to root {self.frag.root}")
                

        elif typ == 1:         # ACCEPT
            self.get_logger().info(f"Agent {self.agent_id} received ACCEPT from {src} for fragment {fid} ")
            self.get_logger().warn(
                f"Frag {self.frag.root}  MERG frag {fid} at N: {self.agent_id}, lvl: {lvl} ")
            self.frag.update_max_weight(weight)
            self._merge_fragments(rid, lvl, fid, children)
            init_pkt = Float32MultiArray(
            data=[5, rid, self.agent_id, src, self.frag.frag_id, self.frag.level, weight, *self.frag.children])
            self.ghs_pub.publish(init_pkt)

            # print(f"Fragment after: {self.frag.__dict__}")
                
        elif typ == 2:         # REJECT
            self.get_logger().info(
                f"Agent {self.agent_id} received REJECT from {src}")
            self.ghs_timer = self.create_timer(0.1, lambda rid=rid: self._report_best(rid))
                    

        elif typ == 3:  # REPORT
            self.frag.reports += 1
            self.add_candidate(src, fid, weight)
            if self.frag.reports >= len(self.frag.children) + 1:     # all children have reported
                if self.frag.root == self.agent_id:
                    # ---- stage 3: root chooses ONE edge ----
                    b_src,b_fid, b_weight = self.frag.best_out

                    # Ask for a Accept
                    pkt = Float32MultiArray(data=[0, rid, self.frag.root, b_src, self.frag.frag_id, self.frag.level, b_weight])
                    self.ghs_pub.publish(pkt)
                    return
                
        elif typ == 4:  # Abort, result found
            self.get_logger().info(
                f"Agent {self.agent_id} received ABORT from {src}"
            )
            # Stop the GHS timer if it exists
            if self.ghs_timer is not None:
                self.ghs_timer.cancel()
            self.stop_all = True  # Stop the agent's main loop

        elif typ == 5:            # Ensure clean Merge
            # Check its not tryingto merge with itself
            if fid == self.frag.frag_id:
                return
            self.get_logger().warn(
                f"Secondary: Frag {self.frag.root}  MERG frag {fid} at N: {self.agent_id}, lvl: {lvl} ")
            self._merge_fragments(rid, lvl, fid, children)
            self.frag.update_max_weight(weight)
                
                
    def _report_up(self, rid):
        if self.frag.root == self.agent_id and len(self.frag.children) == self.num_agents -1:          # I'm root → decide radius
            pkt_r = Float64MultiArray(data=[self.frag.max_weight, rid])
            self.radius_pub.publish(pkt_r)
            self.get_logger().error("Final radius sent to controller: " + str(self.frag.max_weight))
           
            # send end signal to all children
            for child in self.frag.children:
                pkt = Float32MultiArray(data=[4, rid, self.agent_id, child, self.frag.frag_id, self.frag.level, self.frag.max_weight])
                self.ghs_pub.publish(pkt)
                self.get_logger().info(
                    f"Agent {self.agent_id} sent ABORT to child {child} with radius {rid}"
                )
            # stop itself
            pkt = Float32MultiArray(data=[4, rid, self.agent_id, self.agent_id, self.frag.frag_id, self.frag.level, self.frag.max_weight])
            self.ghs_pub.publish(pkt)


    def add_candidate(self, uid, fid, weight: float):
        """Add a candidate if its the best one list."""
        if self.frag.best_out is None or weight < self.frag.best_out[2]:
            self.frag.best_out = (uid, fid, weight)
            # self.get_logger().info(
            #     f"Agent {self.agent_id} added best_out candidate: {self.frag.best_out}"
            # )
        

  

    # ----------------------------------------------------------------------
    # fragment‑merge rule 
    # ----------------------------------------------------------------------
    def _merge_fragments(self, rid, other_level, other_fid, children):
        old_fid = self.frag.root
        if self.frag.level == other_level:
            self.frag.level += 1
            self.frag.frag_id = min(self.frag.frag_id, other_fid)
            self.frag.root = min(self.agent_id, other_fid)

        elif other_level > self.frag.level:
            self.frag.level  = other_level
            self.frag.frag_id = other_fid
            self.frag.root    = other_fid


        if self.frag.root == old_fid:
            self.frag.add_child(other_fid)
            for  child in children:
                if child not in self.frag.children:
                    self.frag.add_child(child)

            self.get_logger().warn(f"Fragment {self.frag.root} includes: {self.frag.children} ")

            self._report_up(rid)
  

        self.frag.best_out = None  # reset best_out
        self.frag.reports = 0  # reset reports
        self.frag.candidate = None  # reset candidate
        self.send_beacon()
        self.ghs_timer = self.create_timer(
            0.2, lambda rid=rid: self._report_best(rid)
        )
                    

class FragmentState:
    def __init__(self, uid):
        self.frag_id = uid
        self.level = 0
        self.root = uid
        self.best_out = None
        self.children = set()
        self.reports = 0
        self.candidate = None
        self.nbr_iter = 0
        self.max_weight = np.inf

    def add_child(self, uid):
        """Add a child to this fragment."""
        self.children.add(uid)

    def update_max_weight(self, weight: float):
        """Update the maximum weight of this fragment."""
        if weight < self.max_weight:
            self.max_weight = weight
 


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
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        # Normal termination triggered by Ctrl‑C or external shutdown
        pass
    finally:
        agent.destroy_node()
        if rclpy.ok():                      # avoid double shutdown
            rclpy.shutdown()


if __name__ == "__main__":
    main()
