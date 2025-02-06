import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app



import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

import random
from omni.isaac.debug_draw import _debug_draw

draw = _debug_draw.acquire_debug_draw_interface()
N = 10000
point_list_1 = [
    (random.uniform(-1000, 1000), random.uniform(-1000, 1000), random.uniform(-1000, 1000)) for _ in range(N)
]
point_list_2 = [
    (random.uniform(-1000, 1000), random.uniform(1000, 3000), random.uniform(-1000, 1000)) for _ in range(N)
]
point_list_3 = [
    (random.uniform(-1000, 1000), random.uniform(-3000, -1000), random.uniform(-1000, 1000)) for _ in range(N)
]
colors = [(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), 1) for _ in range(N)]
sizes = [random.randint(1, 50) for _ in range(N)]
draw.draw_points(point_list_1, [(1, 0, 0, 1)] * N, [10] * N)
draw.draw_points(point_list_2, [(0, 1, 0, 1)] * N, [10] * N)
draw.draw_points(point_list_3, colors, sizes)

sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
# Set main camera
# sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
sim.reset()

while simulation_app.is_running():
    sim.step()

simulation_app.close()

