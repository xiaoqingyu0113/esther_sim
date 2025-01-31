from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as mdp


@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Revolute_[8-9]"], scale=1.0)
    arm_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["Revolute_[1-6]"], scale=1.0)
    