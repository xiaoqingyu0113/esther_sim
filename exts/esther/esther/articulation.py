import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

'''
For ESTHER_CFG, we have the following:
    - spawn: usd_path to the robot's base usd file
    - init_state: initial position of the robot
    - actuators: dictionary of actuator configurations
        - Revolute_1 to Revolute_6: actuator configurations for the robot's joints
            - Position control
            - stiffness: 3000
            - damping: 10
        - Revolute_8 and Revolute_9: actuator configurations for the robot's wheels
            - Velocity control
            - stiffness: 0
            - damping: 600
        - passive1 to passive6: actuator configurations for the robot's passive joints
            - stiffness: 0
            - damping: 0
'''
ESTHER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=os.path.abspath("assets/WheeledTennisRobot-new/tennis_robot_base.usd")),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4)
    ),
    actuators={
        "Revolute_1": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_1"],
            stiffness=3000,
            damping=10,
        ),
        "Revolute_2": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_2"],
            stiffness=3000,
            damping=10,
        ),
        "Revolute_3": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_3"],
            stiffness=3000,
            damping=10,
        ),
        "Revolute_4": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_4"],
            stiffness=3000,
            damping=10,
        ),
        "Revolute_5": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_5"],
            stiffness=3000,
            damping=10,
        ),
        "Revolute_6": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_6"],
            stiffness=3000,
            damping=10,
        ),
        "Revolute_8": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_8"],
            stiffness=0,
            damping=600,
        ),
        "Revolute_9": ImplicitActuatorCfg(
            joint_names_expr=["Revolute_9"],
            stiffness=0,
            damping=600,
        ),
        "passive1": ImplicitActuatorCfg(
            joint_names_expr=["passive1"],
            stiffness=0,
            damping=0,
        ),
        "passive2": ImplicitActuatorCfg(
            joint_names_expr=["passive2"],
            stiffness=0,
            damping=0,
        ),
        "passive3": ImplicitActuatorCfg(
            joint_names_expr=["passive3"],
            stiffness=0,
            damping=0,
        ),
        "passive4": ImplicitActuatorCfg(
            joint_names_expr=["passive4"],
            stiffness=0,
            damping=0,
        ),
        "passive5": ImplicitActuatorCfg(
            joint_names_expr=["passive5"],
            stiffness=0,
            damping=0,
        ),
        "passive6": ImplicitActuatorCfg(
            joint_names_expr=["passive6"],
            stiffness=0,
            damping=0,
        ),
    },
)
