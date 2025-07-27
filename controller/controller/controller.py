import os
import signal
import subprocess
from typing import Dict

import rclpy
from rclpy.node import Node
from interface.srv import ComputeEnergy


AGENT_IDS = [1, 2, 3, 4]

AGENT_POSITIONS = {
    1: (0.0, 0.0),
    2: (10.0, 0.0),
    3: (10.0, 10.0),
    4: (5.0, 3.0),
}

BOUNDARY_ARGS = [
    '--xmin', '0.0', '--ymin', '0.0',
    '--xmax', '50.0', '--ymax', '12.0',
]


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

        # don’t collide with Node internals!
        self._energy_clients_dict: Dict[int, rclpy.client.Client] = {}
        self._energy_futures_dict: Dict[int, rclpy.task.Future] = {}
        self._agent_procs: Dict[int, subprocess.Popen] = {}

        # request constants
        self.starting_point = 0.25
        self.cruise_speed = 10.0
        self.a = 1.0
        self.b = 1.0

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
        while rclpy.ok() and self._energy_futures_dict:
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
        self._shutdown_agents()


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
