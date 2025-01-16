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
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass

from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.assets import Articulation, RigidObject


ESTHER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/core-robotics/Documents/WheeledTennisRobot-new/tennis_robot_base.usd"),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4)
    ),
    actuators={
        "Revolute_1": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_2": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_3": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_3"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_4": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_4"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_5": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_5"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_6": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_6"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_8": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_8"],
            stiffness=None,
            damping=None,
        ),
        "Revolute_9": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_9"],
            stiffness=None,
            damping=None,
        ),
        "passive1": ImplicitActuatorCfg(
            joint_names_expr=["passive1"],
            stiffness=None,
            damping=None,
        ),
        "passive2": ImplicitActuatorCfg(
            joint_names_expr=["passive2"],
            stiffness=None,
            damping=None,
        ),
        "passive3": ImplicitActuatorCfg(
            joint_names_expr=["passive3"],
            stiffness=None,
            damping=None,
        ),
        "passive4": ImplicitActuatorCfg(
            joint_names_expr=["passive4"],
            stiffness=None,
            damping=None,
        ),
        "passive5": ImplicitActuatorCfg(
            joint_names_expr=["passive5"],
            stiffness=None,
            damping=None,
        ),
        "passive6": ImplicitActuatorCfg(
            joint_names_expr=["passive6"],
            stiffness=None,
            damping=None,
        ),


        # "cart_actuator": ImplicitActuatorCfg(
        #     joint_names_expr=["slider_to_cart"],
        #     effort_limit=400.0,
        #     velocity_limit=100.0,
        #     stiffness=0.0,
        #     damping=10.0,
        # ),
        # "pole_actuator": ImplicitActuatorCfg(
        #     joint_names_expr=["cart_to_pole"], effort_limit=400.0, velocity_limit=100.0, stiffness=0.0, damping=0.0
        # ),
    },
)

@configclass
class EstherSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # cartpole
    robot: ArticulationCfg = ESTHER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )



@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    arm_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["Revolute_[1-6]"], scale=1.0)
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Revolute_[8-9]"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # observation terms (order preserved)
        # ['Revolute_8', 'Revolute_9', 'passive1', 'passive2', 'passive3', 'Revolute_1', 'passive4', 'passive5', 
        # 'passive6', 'Revolute_2', 'Revolute_3', 'Revolute_4', 'Revolute_5', 'Revolute_6']
        vel = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
                      env.scene[asset_cfg.name].data.joint_vel[:, [5, 9, 10, 11, 12, 13, 0, 1]])
        
        pos = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
                      env.scene[asset_cfg.name].data.joint_vel[:, [5, 9, 10, 11, 12, 13]])

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    pass


@configclass
class EstherEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = EstherSceneCfg(num_envs=2, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz


def main():
    """Main function."""
    # parse the arguments
    env_cfg = EstherEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedEnv(cfg=env_cfg)

    # simulate physics
    count = 0
    actions = torch.zeros_like(env.action_manager.action)
    actions[:,[-1,-2]] = (torch.rand((len(actions), 2),device=actions.device)-0.5)*10
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            
            print(f"[actions]: {env.action_manager.action}")

            # step the environment
            obs, _ = env.step(actions)
            # print current orientation of pole
            print("[Env 0]: joints: ", obs["policy"])
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()