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
point_list_1 = [
    (random.uniform(-300, -100), random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(10)
]
draw.draw_lines_spline(point_list_1, (1, 1, 1, 1), 10, False)
point_list_2 = [
    (random.uniform(-300, -100), random.uniform(-100, 100), random.uniform(-100, 100)) for _ in range(10)
]
draw.draw_lines_spline(point_list_2, (1, 1, 1, 1), 5, True)


sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
# Set main camera
# sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])
sim.reset()

while simulation_app.is_running():
    print('dim 0',len(point_list_2))
    print(len(point_list_2[0]))
    sim.step()

simulation_app.close()

