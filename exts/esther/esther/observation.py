from omni.isaac.lab.utils import configclass
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.
        
        Observation terms (order preserved)
        ['Revolute_8', 'Revolute_9', 'passive1', 'passive2', 'passive3', 'Revolute_1', 'passive4', 'passive5', 
        'passive6', 'Revolute_2', 'Revolute_3', 'Revolute_4', 'Revolute_5', 'Revolute_6']

        Remap to ['Revolute_8', 'Revolute_9', 'Revolute_1', 'Revolute_2', 'Revolute_3', 'Revolute_4', 'Revolute_5', 'Revolute_6']
        
        which represent the joint velocities of the joints

        index | joint_name  | real_joint_name
        0     | Revolute_8  | left wheel
        1     | Revolute_9  | right wheel
        2     | Revolute_1  | arm base yaw
        3     | Revolute_2  | arm base pitch
        4     | Revolute_3  | arm shoulder rotation
        5     | Revolute_4  | arm elbow pitch
        6     | Revolute_5  | arm wrist yew

        "vel" : joint velocities
        "pos" : joint positions
        "odom" : robot base odometry in world frame [pos, quat, lin_vel, ang_vel]
        """
        # vel = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
        #               env.scene[asset_cfg.name].data.joint_vel[:, [0, 1, 5, 9, 10, 11, 12, 13]])
        
        # pos = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
        #               env.scene[asset_cfg.name].data.joint_vel[:, [5, 9, 10, 11, 12, 13]])
        
        # odom = ObsTerm(func=lambda env, asset_cfg = SceneEntityCfg("robot"):
        #                env.scene[asset_cfg.name].data.root_state_w)
        

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()