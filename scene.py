import torch
import os

from isaaclab.assets import Articulation, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg, DCMotorCfg
import isaaclab.sim as sim_utils


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OBJECT_DESCRIPTION_DIR = os.path.join(BASE_DIR, "hoi_description", "objects")

CLOTH_STAND_CFG = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/ClothStand",
                    spawn=sim_utils.UsdFileCfg(
                        scale=(1.0, 1.0, 1.0),
                        usd_path=f"{OBJECT_DESCRIPTION_DIR}/clothesstand/clothesstand.usda",
                        activate_contact_sensors=True,
                        mass_props=sim_utils.MassPropertiesCfg(
                            mass=0.2,
                        ),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False,
                            retain_accelerations=False,
                            linear_damping=0.0,
                            angular_damping=0.0,
                            max_linear_velocity=1000.0,
                            max_angular_velocity=1000.0,
                            max_depenetration_velocity=1.0,
                        ),
                    ),
                )