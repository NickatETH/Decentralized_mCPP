import math
from typing import Tuple
import rclpy

import numpy as np
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, Float64MultiArray


# QoS shortcuts --------------------------------------------------------------
qos_best_effort = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
qos_reliable_vol = QoSProfile(
    depth=50,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

qos_reliable_vol_lifetime = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    lifespan=Duration(seconds=0, nanoseconds=500_000_000),
)


class RadiusMixin:

    def __init__(self):
        super().__init__()
        self.stop_all = False  # Flag to stop the agent's main loop

        self.frag = None
        self.radius_pos = (0.0, 0.0)
        self.starting_point = 0.0
        self.nbr_table = {}
        self.nbr_attempt_time = None
        self.current_rid = None

        self.beacon_pub = self.create_publisher(
            PointStamped, "/beacon", qos_reliable_vol_lifetime
        )
        self.ghs_pub = self.create_publisher(
            Float32MultiArray, "/ghs_packet", qos_reliable_vol
        )
        self.radius_pub = self.create_publisher(
            Float64MultiArray, "/radius", qos_reliable_vol
        )
        self.start_sub = self.create_subscription(
            Float64MultiArray, "/start_ghs", self.start_cb, qos_reliable_vol
        )
        self.packet_sub = self.create_subscription(
            Float32MultiArray, "/ghs_packet", self.ghs_cb, qos_reliable_vol
        )
        self.beacon_sub = self.create_subscription(
            PointStamped, "/beacon", self.beacon_cb, qos_reliable_vol_lifetime
        )

        self.ghs_timer = None  # Timer for GHS probe (started later)

    def prep_radiusmixin(self, agent_id: float):
        """Reset the radius position and ID."""
        self.frag = FragmentState(agent_id)

    def reset_radius(self):
        """Reset the radius position and fragment state."""
        self.frag = FragmentState(self.agent_id)
        self.radius_pos = (0.0, 0.0)
        self.starting_point = 0.0
        self.nbr_table.clear()
        self.nbr_attempt_time = None
        now = self.get_clock().now()  # get rid of old stuff
        while now + Duration(seconds=0.1) > self.get_clock().now():
            rclpy.spin_once(self, timeout_sec=0.01)
        self.stop_all = False

    def beacon_cb(self, msg: PointStamped):
        uid = float(msg.header.frame_id)
        if uid != self.agent_id:
            self.nbr_table[uid] = (msg.point.x, msg.point.y, msg.point.z)

    # Start‑GHS handler
    def start_cb(self, msg: Float64MultiArray):
        """
        Expects a Float64MultiArray whose length is a multiple of 4:
            [t_target, rid, sp0, sp1, sp2, ...]
        Only the tuple addressed to *this* agent is processed.
        """

        self.reset_radius()  # Reset radius state
        data = msg.data
        if len(data) < 2:
            self.get_logger().error("start_cb: message too short")
            return

        t_target = data[0]  # Offset time
        rid = data[1]  # Passthrough value (returned as is at the end with radius)
        self.current_rid = rid

        idx = int(1 + self.agent_id)
        sp = data[idx]
        self.starting_point = sp

        path_idx = int(
            (t_target + sp) * (len(self.path.coords) - 1) % len(self.path.coords)
        )
        self.radius_pos = self.path.coords[path_idx]

        x, y = self.radius_pos
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.agent_id)
        msg.point.x, msg.point.y = x, y
        msg.point.z = float(self.frag.frag_id)
        self.beacon_pub.publish(msg)

        self.frag = FragmentState(self.agent_id)  # reset fragment state
        now = self.get_clock().now()  # get rid of old stuff
        while now + Duration(seconds=0.2) > self.get_clock().now():
            rclpy.spin_once(self, timeout_sec=0.01)
        if self.ghs_timer is not None:
            self.ghs_timer.cancel()
        self.ghs_timer = self.create_timer(0.1, lambda: self._report_best(rid))

    def _compute_best_out(self):
        # 1) collect all neighbours except your own fragment
        neighbours = []
        for nbr_uid, nbr_p in self.nbr_table.items():
            x, y, nbr_fid = nbr_p
            if nbr_fid == self.frag.frag_id or nbr_uid == self.agent_id:
                continue
            dist = math.hypot(x - self.radius_pos[0], y - self.radius_pos[1])
            neighbours.append((nbr_uid, nbr_fid, float(dist)))

        neighbours.sort(key=lambda pair: pair[2])

        idx = self.frag.nbr_iter
        if idx < len(neighbours):
            self.frag.best_out = neighbours[
                idx
            ]  # pick the “next best” un‐tried neighbour
        else:
            if self.nbr_attempt_time is None:
                self.nbr_attempt_time = self.get_clock().now()
            if self.nbr_attempt_time + Duration(seconds=1.5) > self.get_clock().now():
                return
            else:
                # self.get_logger().info(
                #     f"Agent {self.agent_id} has no more neighbours to try: {neighbours}: time {self.nbr_attempt_time}, now {self.get_clock().now()}"
                # )
                # self.nbr_attempt_time = None
                # rclpy.spin_once(self, timeout_sec=0.1)
                # # pkt_r = Float64MultiArray(data=[0.0, -1.0])
                # # self.radius_pub.publish(pkt_r)
                # self.stop_all = True  # Stop the agent's main loop
                # for child in self.frag.children:
                #     pkt = Float32MultiArray(
                #         data=[
                #             4,
                #             self.current_rid,
                #             self.agent_id,
                #             child,
                #             self.frag.frag_id,
                #             self.frag.level,
                #             self.frag.max_weight,
                #         ]
                #     )
                #     self.ghs_pub.publish(pkt)
                #     print("Aborting for nbrs")
                return

            self.frag.nbr_iter = 0  # reset neighbour iteration

    def _report_best(self, rid: float):
        """Report the best candidate to the root of the fragment."""
        self.ghs_timer.cancel() if self.ghs_timer else None
        rclpy.spin_once(self, timeout_sec=0.1)
        rclpy.spin_once(self, timeout_sec=0.0)
        rclpy.spin_once(self, timeout_sec=0.0)
        if self.stop_all:
            return
        self._compute_best_out()
        if self.frag.best_out is not None:
            nbr, fid, w = self.frag.best_out
            self.frag.candidate = (nbr, fid, w)  # store candidate
            pkt = Float32MultiArray(
                data=[3, rid, nbr, self.frag.root, fid, self.frag.level, w]
            )
            self.ghs_pub.publish(pkt)

    # ------------------------------------------------------------------
    # GHS packet handler (Test/Accept/Reject/Report)
    def ghs_cb(self, msg: Float32MultiArray):
        if self.stop_all == True and msg.data[0] != 4 and msg.data[0] != 5:
            return
        if msg.data[3] != self.agent_id or msg.data[1] != self.current_rid:
            return  # ignore packets not addressed to me
        typ, rid, src, dst, fid, lvl, weight, *children = msg.data
        if typ == 0:  # TEST
            ans_typ = 0
            if self.agent_id == self.frag.frag_id:
                # I'm the root of my fragment, so I can accept the test
                if self.frag.best_out is not None and self.frag.best_out[1] == fid:
                    # build a flat list of child‐UIDs as floats
                    children_ids = list(self.frag.children)
                    children_uids = [float(uid) for uid in children_ids]
                    base_data = [
                        1.0,
                        rid,
                        self.agent_id,
                        src,
                        self.frag.frag_id,
                        self.frag.level,
                        weight,
                    ]

                    pkt_data = base_data + children_uids
                    pkt = Float32MultiArray(data=pkt_data)
                    self.ghs_pub.publish(pkt)
                    # self.get_logger().info(
                    #     f"Agent {self.agent_id} accepted TEST from {src} for fragment {fid}, root {self.frag.root}"
                    # )
                    return

                elif self.frag.best_out is not None:
                    pkt = Float32MultiArray(
                        data=[
                            2,
                            rid,
                            self.agent_id,
                            src,
                            self.frag.frag_id,
                            self.frag.level,
                            weight,
                        ]
                    )
                    self.ghs_pub.publish(pkt)
                    return
                else:
                    # reject
                    pkt = Float32MultiArray(
                        data=[
                            2,
                            rid,
                            self.agent_id,
                            src,
                            self.frag.frag_id,
                            self.frag.level,
                            weight,
                        ]
                    )

                    self.get_logger().info(
                        f"Agent {self.agent_id} REJECT: NONE FOUND from {src} for fragment {fid}, root {self.frag.root}"
                    )
                    self.ghs_pub.publish(pkt)
                    return

            else:
                # Ask root in the name of the other agent
                pkt = Float32MultiArray(
                    data=[0, rid, src, self.frag.root, fid, lvl, weight]
                )
                self.ghs_pub.publish(pkt)
                # print(f"Agent {self.agent_id} sent TEST to root {self.frag.root}")

        elif typ == 1:  # ACCEPT
            self.get_logger().info(
                f"Frag {self.frag.root}  MERG frag {fid} at N: {self.agent_id}, lvl: {lvl} "
            )
            init_pkt = Float32MultiArray(
                data=[
                    5,
                    rid,
                    self.agent_id,
                    src,
                    self.frag.frag_id,
                    self.frag.level,
                    weight,
                    *self.frag.children,
                ]
            )

            self.frag.update_max_weight(weight)
            self._merge_fragments(rid, lvl, fid, children)
            self.ghs_pub.publish(init_pkt)

            # print(f"Fragment after: {self.frag.__dict__}")

        elif typ == 2:  # REJECT
            # self.get_logger().info(f"Agent {self.agent_id} received REJECT from {src}")
            if self.stop_all != True:
                if self.ghs_timer is not None:
                    self.ghs_timer.cancel()
                self.ghs_timer = self.create_timer(
                    0.1, lambda rid=rid: self._report_best(rid)
                )

        elif typ == 3:  # REPORT
            # self.get_logger().info(f"Agent {self.agent_id} received REPORT of {src}")
            self.frag.reports += 1
            if self.frag.best_out is None or weight < self.frag.best_out[2]:
                self.frag.best_out = (src, fid, weight)

            # all children have reported
            if self.frag.reports >= len(self.frag.children) + 1:
                if self.frag.root == self.agent_id:
                    b_src, b_fid, b_weight = self.frag.best_out

                    # Ask for CONSENSUS
                    pkt = Float32MultiArray(
                        data=[
                            0,
                            rid,
                            self.frag.root,
                            b_src,
                            self.frag.frag_id,
                            self.frag.level,
                            b_weight,
                        ]
                    )
                    self.ghs_pub.publish(pkt)
                    # self.get_logger().info(
                    #     f"Agent {self.agent_id} sent CONSENSUS request for {b_src}"
                    # )

                    # send reject to other children
                    for child in self.frag.children:
                        if child != b_src:
                            pkt = Float32MultiArray(
                                data=[
                                    2,
                                    rid,
                                    self.agent_id,
                                    child,
                                    self.frag.frag_id,
                                    self.frag.level,
                                    b_weight,
                                ]
                            )
                            self.ghs_pub.publish(pkt)
                            # self.get_logger().info(
                            #     f"XXXXAgent {self.agent_id} sent REJECT to {child}"
                            # )
                    return
            # else:
            # self.get_logger().info(
            #     f"Agent {self.agent_id} children {len(self.frag.children)}, reports: {self.frag.reports}"
            # )

        elif typ == 4:  # Abort, result found --> Reset function?
            # self.get_logger().info(f"Agent {self.agent_id} received ABORT from {src}")
            if self.ghs_timer is not None:
                self.ghs_timer.cancel()
            self.stop_all = True  # Stop the agent's main loop
            # self.get_logger().info(
            #     f"Agent {self.agent_id} received ABORT from {src} for fragment {fid}, root {self.frag.root}"
            # )
            return

        elif typ == 5:  # Clean Merge
            # self.get_logger().warn(
            #     f"Agent {self.agent_id} received CLEAN MERGE from N: {src} for fragment {fid}, root {self.frag.root}"
            # )
            # Check its not trying to merge with itself
            if fid == self.frag.frag_id:
                return
            self.get_logger().info(
                f"Secondary: Frag {self.frag.root}  MERG frag {fid} at N: {self.agent_id}, lvl: {lvl} "
            )
            self._merge_fragments(rid, lvl, fid, children)
            self.frag.update_max_weight(weight)

    def _report_up(self, rid):
        if (
            self.frag.root == self.agent_id
            and len(self.frag.children) == self.num_agents - 1
            and not self.stop_all
        ):  # Root decides radius

            pkt_r = Float64MultiArray(data=[self.frag.max_weight, rid])
            if not self.stop_all:
                self.stop_all = True  # Stop the agent's main loop
                # send end signal to all children
                for child in self.frag.children:
                    pkt = Float32MultiArray(
                        data=[
                            4,
                            rid,
                            self.agent_id,
                            child,
                            self.frag.frag_id,
                            self.frag.level,
                            self.frag.max_weight,
                        ]
                    )
                    self.ghs_pub.publish(pkt)
                    rclpy.spin_once(self, timeout_sec=0.05)
                    self.ghs_pub.publish(pkt)

                pkt = Float32MultiArray(
                    data=[
                        4,
                        rid,
                        self.agent_id,
                        self.agent_id,
                        self.frag.frag_id,
                        self.frag.level,
                        self.frag.max_weight,
                    ]
                )
                self.ghs_pub.publish(pkt)

                self.radius_pub.publish(pkt_r)
                self.get_logger().error(
                    "Final radius sent to controller: " + str(self.frag.max_weight)
                )

                return True
        return False

    # fragment‑merge rule
    def _merge_fragments(self, rid, other_level, other_fid, children):
        old_fid = self.frag.root
        if self.frag.level == other_level:
            self.frag.level += 1
            self.frag.frag_id = min(self.frag.frag_id, other_fid)
            self.frag.root = min(self.agent_id, other_fid)

        elif other_level > self.frag.level:
            self.frag.level = other_level
            self.frag.frag_id = other_fid
            self.frag.root = other_fid

        if self.frag.root == old_fid:
            self.frag.add_child(other_fid)
            for child in children:
                if child not in self.frag.children:
                    self.frag.add_child(child)

            # self.get_logger().warn(
            #     f"Fragment {self.frag.root} includes: {self.frag.children} "
            # )
            if self._report_up(rid):
                return

        self.frag.best_out = None
        self.frag.reports = 0
        self.frag.candidate = None

        x, y = self.radius_pos
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.agent_id)
        msg.point.x, msg.point.y = x, y
        msg.point.z = float(self.frag.frag_id)
        self.beacon_pub.publish(msg)

        if self.ghs_timer is not None:
            self.ghs_timer.cancel()
        self.ghs_timer = self.create_timer(0.1, lambda rid=rid: self._report_best(rid))


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
