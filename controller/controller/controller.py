import os
import signal
import subprocess
from typing import Dict
from rclpy.duration import Duration

import rclpy
import numpy as np
from rclpy.node import Node
from interface.srv import ComputeEnergy
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from visualization_msgs.msg import Marker  # Add this import



AGENT_IDS = [1, 2, 3, 4]

AGENT_POSITIONS = {
    1: (0.0, 0.0),
    2: (50.0, 0.0),
    3: (50.0, 50.0),
    4: (5.0, 40.0),
}

BOUNDARY_ARGS = [
    '--xmin', '0.0', '--ymin', '0.0',
    '--xmax', '50.0', '--ymax', '50.0',
]

# QoS shortcuts --------------------------------------------------------------
qos_best_effort  = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                              durability=DurabilityPolicy.VOLATILE)
qos_reliable_vol = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
qos_reliable_tx  = QoSProfile(depth=2,  reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)



def launch_agent(aid: int, x: float, y: float) -> subprocess.Popen:
    cmd = [
        'ros2', 'run', 'drone', 'uav_agent',
        '--agent_id', str(aid), '--posx', str(x), '--posy', str(y),
        *BOUNDARY_ARGS,
    ]
    return subprocess.Popen(cmd,                # POSIX
                            preexec_fn=os.setsid)


class Controller(Node):
    def __init__(self) -> None:
        super().__init__('controller')
        self.start_pub = self.create_publisher(
            Float64MultiArray, '/start_ghs', qos_reliable_tx)
        self.radius_sub = self.create_subscription(
            Float64MultiArray, '/radius', self.radius_cb, qos_reliable_vol)
        self.radius_marker_pub = self.create_publisher(Marker, '/comm_radius_marker', qos_reliable_tx)
        self.radius_timer = None



        self.scheduler = RadiusScheduler(self,
                                    self.start_pub,
                                    v_max=10.0,
                                    eps=0.01)
        


        # don’t collide with Node internals!
        self._energy_clients_dict: Dict[int, rclpy.client.Client] = {}
        self._energy_futures_dict: Dict[int, rclpy.task.Future] = {}
        self._agent_procs: Dict[int, subprocess.Popen] = {}

        # request constants
        self.starting_point = 0.25
        self.cruise_speed = 10.0
        self.a = 1.0
        self.b = 1.0
        
        self.com_estimate = False
        self.com_ok = False

        # launch agents
        for aid in AGENT_IDS:
            x, y = AGENT_POSITIONS[aid]
            proc = launch_agent(aid, x, y)
            self._agent_procs[aid] = proc
            self.get_logger().info(f'Launched uav_agent_{aid} (pid {proc.pid})')

        # create service clients and send first request
        for aid in AGENT_IDS:
            srv = f'/uav_agent_{aid}/compute_energyy'
            cli = self.create_client(ComputeEnergy, srv)
            self._energy_clients_dict[aid] = cli
            self.get_logger().info(f'Waiting for {srv} …')
            while not cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().warn(f'{srv} not up yet, waiting…')
            self._send_request(aid)




    def radius_cb(self, msg: Float64MultiArray):
        r, rid = msg.data
        self.scheduler.store_radius(r, rid)



    # ------------------------------------------------------------------ helpers
    def _send_request(self, aid: int) -> None:
        req = ComputeEnergy.Request()
        req.starting_point = self.starting_point
        req.cruise_speed = self.cruise_speed
        req.a = self.a
        req.b = self.b
        self._energy_futures_dict[aid] = self._energy_clients_dict[aid].call_async(req)

    def _shutdown_agents(self) -> None:
        """Politely stop every launched uav_agent; force‑kill if they ignore us."""
        for aid, proc in self._agent_procs.items():
            if proc.poll() is not None:
                continue                              # already gone

            self.get_logger().info(f'Shutting uav_agent_{aid}')
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # POSIX
                proc.wait(timeout=3)                  # wait up to 3 s
            except (subprocess.TimeoutExpired, ProcessLookupError):
                # Still alive?  Brutal exit.
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait()

    # ----------------------------------------------------------- main spin loop
    def spin_until_complete(self) -> None:
        while rclpy.ok() and self._energy_futures_dict or self.com_estimate:
            rclpy.spin_once(self, timeout_sec=0.1)
            done = []
            for aid, fut in self._energy_futures_dict.items():
                if fut.done():
                    try:
                        res = fut.result()
                        if res.energy == -1.0:
                            self.get_logger().warn(
                                f'Agent {aid}: path not ready – retrying')
                            self._send_request(aid)
                            continue
                        self.get_logger().info(
                            f'Agent {aid}: Energy = {res.energy:.2f} J')
                    except Exception as exc:
                        self.get_logger().error(f'Agent {aid} failed: {exc}')
                    done.append(aid)
            for aid in done:
                del self._energy_futures_dict[aid]
        self.get_logger().info('All energy requests done, waiting for comm radius estimation...')
        self.radius_timer = self.create_timer(3.0, self.scheduler.maybe_request_probe)
        while not self.com_ok and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.5)
        self._shutdown_agents()




class RadiusScheduler:
    """Minimal Shubert–Piyavskii sampler for r(t) with L = 2 · v_max."""

    def __init__(self, node: rclpy.node.Node,
                 start_pub,                 # the existing publisher
                 v_max: float,              # worst‑case UAV speed [m/s]
                 eps: float = 2.0):         # desired accuracy   [m]
        self._node      = node
        self._pub       = start_pub
        self._L         = 2.0 * v_max
        self._eps       = eps
        self._round_id  = 0
        # (t,r) pairs already measured; endpoints hold +∞ until sampled
        self._samples = [(0.0, 0.0), (1.0, 0.0)]
        self.max_radius = 0.0               # highest r(t) seen so far

    # ------------------------------------------------------------------ helpers
    def _roof(self, t: float) -> float:
        """Upper envelope U(t) = min_k ( r_k + L |t - t_k| )."""
        return min(r + self._L * abs(t - tk) for tk, r in self._samples)

    def _next_probe_time(self) -> float | None:
        """Return the t where the gap between roof and chord is biggest."""
        best_gap, best_t = 0.0, None
        for (t0, r0), (t1, r1) in zip(self._samples, self._samples[1:]):
            t_mid = 0.5 * (t0 + t1)
            gap   = self._roof(t_mid) - 0.5 * (r0 + r1)
            if gap > best_gap:
                best_gap, best_t = gap, t_mid
        if best_gap < self._eps: 
            self._node.get_logger().info(f"Final max radius: {self.max_radius:.2f} m")
            self._node.com_ok = True
            return None  # no more probes needed
        return best_t 

    # ------------------------------------------------------------------ public
    def maybe_request_probe(self) -> None:
        """Call from a timer (e.g. every 0.2 s).  Starts a new round if needed."""
        if self._round_id < 2: # first two rounds are special
            t_probe = float(self._round_id) 
            
        else: 
            t_probe = self._next_probe_time()
            if t_probe is None:                         # envelope tight enough
                return

        self._round_id += 1

        # Example: starting_points indexed by agent id (1-based)
        starting_points = [0.25, 0.5, 0.75, 1.0]  # index 0 unused

        # Broadcast the start message to all agents
        payload = [t_probe, t_probe] + starting_points
        self._pub.publish(Float64MultiArray(data=payload))      

        
        self._node.get_logger().info(
            f"GHS round {self._round_id} @ t={t_probe:.2} and sp={starting_points} rreq")

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
            f"(current max {self.max_radius:.2f} m)")
        print(f"Current samples: {[(round(t, 2), round(r, 2)) for t, r in self._samples]}")



def main(args=None) -> None:
    rclpy.init(args=args)
    node = Controller()
    try:
        node.spin_until_complete()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()