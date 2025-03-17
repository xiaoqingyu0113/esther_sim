from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

# def dummy(env, asset_cfg = SceneEntityCfg("robot")):
#     print(env.scene[asset_cfg.name].joint_names)
#     return env.scene[asset_cfg.name].data.joint_vel[:, [0]]

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
        - Observation terms (order preserved)
            - Originally 
                ['j1', 'sw_joint_1', 'sw_joint_2', 'sw_joint_3', 'wheel_joint_1', 'wheel_joint_2', 'j2', 'j3', 'j4', 'j5']

            - Remap to [0,6,7,8,9,4,5]
                which are ['j1', 'j2', 'j3', 'j4', 'j5', 'wheel_joint_1', 'wheel_joint_2']
        
            - joint meanings:
                - j[1-5]: wam robot arm (https://web.barrett.com/files/B2576_RevAC-00.pdf)
                - wheel_joint_[1-2]: wheel joints [left, right]

        "vel" : joint velocities
        "pos" : joint positions
        "odom" : robot base odometry in world frame [pos, quat, lin_vel, ang_vel]
        """
        # vel = ObsTerm(func=dummy)

        vel = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
                      env.scene[asset_cfg.name].data.joint_vel[:, [0,6,7,8,9,4,5]])
        
        pos = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
                      env.scene[asset_cfg.name].data.joint_vel[:, [0,6,7,8,9]])
        
        odom = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
                       env.scene[asset_cfg.name].data.root_state_w)
        

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()