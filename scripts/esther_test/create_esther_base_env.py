# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# start the simulator
import torch
from omni.isaac.lab.envs import ManagerBasedEnv
from esther.env import EstherEnvCfg



def main():
    """Main function."""
    # parse the arguments
    env_cfg = EstherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    actions = torch.zeros_like(env.action_manager.action, device=env.device)
    actions[:, 0:2] = torch.tensor([0.1, 0.1], device=env.device)
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, _ = env.step(actions)
            print("[Env]: joints: ", obs["policy"]['vel'][0,-2:])
            count += 1
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()