#!/usr/bin/env python3
"""
Entry-point.  Adjust defaults via CLI flags –– see --help.
"""
import argparse
from swarm.simulation import run_simulation

def get_args():
    p = argparse.ArgumentParser(description="UAV-swarm toy demo")
    p.add_argument("-g", "--grid", type=int, default=150,     help="grid size")
    p.add_argument("-n", "--num",  type=int, default=10,      help="# agents")
    p.add_argument("-s", "--steps",type=int, default=180,     help="# frames")
    p.add_argument("--inverseV",  action="store_true",        help="use inverse-V density")
    p.add_argument("--circle",    action="store_true",        help="use circle density")
    p.add_argument("--kill",      type=int, default=40,       help="kill interval (0 = off)")
    return p.parse_args()

if __name__ == "__main__":
    cfg = get_args()
    run_simulation(cfg)
