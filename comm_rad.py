import math
from typing import Optional, Tuple, List

class SPSampler:
    L: float
    eps: float
    horizon: float

    def __post_init__(self):
        self.samples: List[Tuple[float, float]] = [(0.0, math.inf), (self.horizon, math.inf)]

    def _roof(self, t: float) -> float:
        return min(r + self.L * abs(t - tt) for tt, r in self.samples)

    def next_probe(self) -> Optional[float]:
        gaps = []
        for (t0, r0), (t1, r1) in zip(self.samples, self.samples[1:]):
            t_mid = 0.5 * (t0 + t1)
            h = self._roof(t_mid)
            if h - 0.5 * (r0 + r1) > self.eps:
                gaps.append((h, t_mid))
        return max(gaps)[1] if gaps else None

    def add(self, t: float, r: float):
        self.samples.append((t, r))
        self.samples.sort(key=lambda x: x[0])

# ===========================================================================
# 5.   Controller‑side orchestrator ------------------------------------------
# ===========================================================================

def controller_loop(sim_state, v_max: float = 15.0, eps: float = 5.0):
    horizon = (len(next(iter(sim_state.values())).trajectory) - 1) * sim_state.dt  # type: ignore
    sp = SPSampler(L=2 * v_max, eps=eps, horizon=horizon)
    round_id = 0

    while True:
        t_probe = sp.next_probe()
        if t_probe is None:
            break
        round_id += 1
        #GHS START
        for agent_id in sim_state.get_agent_ids():
            if round_id == 1:
                # Only for every neighbour Later ON!
                pos = sim_state.get_agent_post_at_time(agent_id, t_probe)
                nb_dist = []
                for nb_id in sim_state.get_agent_ids():
                    if nb_id != agent_id:
                        nb_pos = sim_state.get_agent_post_at_time(nb_id, t_probe)
                        dist = math.dist(pos, nb_pos)
                        nb_dist.append(dist)
            # Send GHS messages
            
                    




    print(f"Design radius: {max(r for _, r in sp.samples):.2f} m")
    
class MsgT(enum.IntEnum):
    BEACON      = 1  # uid, x, y, z             (broadcast)
    START_GHS   = 2  # controller → all, t_target, round_id
    TEST        = 3  # ghs
    ACCEPT      = 4
    REJECT      = 5
    REPORT      = 6
    CHANGE_ROOT = 7
    RADIUS      = 8  # root → controller, r_value, round_id

# helper for serialisation
pack_f = lambda *floats: b"|".join(str(f).encode() for f in floats)
unpack_f = lambda b: tuple(map(float, b.split(b"|")))