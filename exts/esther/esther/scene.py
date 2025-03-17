from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim.spawners import materials, physics_materials,PhysicsMaterialCfg
from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from .articulation import ESTHER_CFG


@configclass
class HighFrictionMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for rigid bodies.

    See :meth:`spawn_rigid_body_material` for more information.

    Note:
        The default values are the `default values used by PhysX 5
        <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#rigid-body-materials>`__.
    """

    func: Callable = physics_materials.spawn_rigid_body_material
    static_friction: float = 0.85
    dynamic_friction: float = 0.85
    restitution: float = 0.0
    improve_patch_friction: bool = True
    friction_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    restitution_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    compliant_contact_stiffness: float = 0.0
    compliant_contact_damping: float = 0.0


@configclass
class EstherSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0), physics_material=HighFrictionMaterialCfg()),
    )

    # esther robot
    robot: ArticulationCfg = ESTHER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
