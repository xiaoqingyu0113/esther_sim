from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as mdp


@configclass
class ActionsCfg:
    
    '''
    wheel_vel:
        Revolute_9: left wheel
        Revolute_8: right wheel
    arm_pos:
        Revolute_[1-6]: arm joints
    
    '''
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Revolute_9", "Revolute_8"], scale=1.0,preserve_order=True)
    arm_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["Revolute_[1-6]"], scale=1.0, preserve_order=True)
    