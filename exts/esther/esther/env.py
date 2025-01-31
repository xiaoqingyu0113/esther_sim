from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import ManagerBasedEnvCfg
from .scene import EstherSceneCfg
from .observation import ObservationsCfg
from .action import ActionsCfg


@configclass
class EstherEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""
    @configclass
    class EventCfg:
        """Configuration for events."""
        pass

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