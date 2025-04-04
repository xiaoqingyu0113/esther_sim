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
        usd_path=os.path.abspath(os.path.expanduser("~/varshith_lab/source/extensions/esther/assets/wheelchair_tennis/wtr/wtr.usd"))),
        init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.0)
                    ),
    actuators={
        "j1": ImplicitActuatorCfg(
            joint_names_expr=["j1"],
            effort_limit=100,
            stiffness=300,
            damping=10,
        ),
        "j2": ImplicitActuatorCfg(
            joint_names_expr=["j2"],
            effort_limit=100,
            stiffness=300,
            damping=10,
        ),
        "j3": ImplicitActuatorCfg(
            joint_names_expr=["j3"],
            effort_limit=100,
            stiffness=300,
            damping=10,
        ),
        "j4": ImplicitActuatorCfg(
            joint_names_expr=["j4"],
            effort_limit=100,
            stiffness=300,
            damping=10,
        ),
        "j5": ImplicitActuatorCfg(
            joint_names_expr=["j5"],
            effort_limit=100,
            stiffness=300,
            damping=10,
        ),
    
        "wheel_joint_1": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint_1"],
            effort_limit=1000000000,
            stiffness=0,
            damping=50000,
        ),
        "wheel_joint_2": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint_2"],
            effort_limit=1000000000,
            stiffness=0,
            damping=50000,
        ),
        "sw_joint_1": ImplicitActuatorCfg(
            joint_names_expr=["sw_joint_1"],
            stiffness=0,
            damping=0,
        ),
        "sw_joint_2": ImplicitActuatorCfg(
            joint_names_expr=["sw_joint_2"],
            stiffness=0,
            damping=0,
        ),
        "sw_joint_3": ImplicitActuatorCfg(
            joint_names_expr=["sw_joint_3"],
            stiffness=0,
            damping=0,
        ),
   
       
    },
)
