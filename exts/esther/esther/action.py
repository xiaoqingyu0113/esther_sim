from omni.isaac.lab.utils import configclass
import omni.isaac.lab.envs.mdp as mdp


@configclass
class ActionsCfg:
    '''
    In env.action_manager.action, the shape is [num_envs, num_actions].

        - num_envs: Number of environments to spawn.
        - num_actions: wheel_vel [0-1], arm_pos [2-6]
    
    
    '''
    wheel_vel = mdp.JointVelocityActionCfg(asset_name="robot", joint_names="wheel_joint_[1-2]", scale=1.0,preserve_order=True)
    arm_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["j[1-5]"], scale=1.0, preserve_order=True)
    