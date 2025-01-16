# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


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

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

ESTHER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"assets/WheeledTennisRobot-new/tennis_robot_base.usd",
        # rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #     rigid_body_enabled=True,
        #     max_linear_velocity=1000.0,
        #     max_angular_velocity=1000.0,
        #     max_depenetration_velocity=100.0,
        #     enable_gyroscopic_forces=True,
        # ),
        # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #     enabled_self_collisions=False,
        #     solver_position_iteration_count=4,
        #     solver_velocity_iteration_count=0,
        #     sleep_threshold=0.005,
        #     stabilization_threshold=0.001,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0)
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
        # "Revolute_7": ImplicitActuatorCfg(
        #     joint_names_expr=["Revolute_7"],
        #     stiffness=None,
        #     damping=None,
        # ),
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

def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = [[0.0,0.0,-0.060], [10.0,0.0, -0.060]]
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0], orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1], orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]))


    # Articulation
    esther_cfg = ESTHER_CFG.copy()
    esther_cfg.prim_path = "/World/Origin.*/Robot"
    esther = Articulation(cfg=esther_cfg)

    # return the scene information
    scene_entities = {"esther": esther}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["esther"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
  
        # Apply random action
        # -- apply action to the robot
        # ['Revolute_8', 'Revolute_9', 'passive1', 'passive2', 'passive3', 'Revolute_1', 'passive4', 'passive5', 
        # 'passive6', 'Revolute_2', 'Revolute_3', 'Revolute_4', 'Revolute_5', 'Revolute_6']
        robot.set_joint_velocity_target(torch.zeros((14), device=sim.device))
        robot.set_joint_velocity_target(torch.tensor([5.0, 0.0],device=sim.device),joint_ids=[0,1])

        # robot.set_joint_velocity_target(torch.zeros((2, 14), device=sim.device))
        # robot.set_joint_velocity_target(torch.tensor([[5.0, 0.0], [-5.0,0]],device=sim.device),joint_ids=[0,1])

        robot.write_data_to_sim()

        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)
        print(robot.root_physx_view.get_dof_velocities())
1

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()