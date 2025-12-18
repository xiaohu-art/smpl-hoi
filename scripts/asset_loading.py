import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Load multiple assets in different environments.")
parser.add_argument("--num_envs", type=int, default=1, help="The number of environments to load the asset into.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import joblib
from tqdm import tqdm

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_unique, quat_from_angle_axis

##
# Pre-defined configs
##
import active_adaptation as aa
aa.set_backend("isaac")

from smpl_hoi.asset import SMPL, LARGEBOX, MONITOR, TRASHCAN

object_list = [
    LARGEBOX.isaaclab(),
    MONITOR.isaaclab(),
    TRASHCAN.isaaclab(),
]

object_number = len(object_list)

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )



def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / 200
    # Increase solver iterations for better collision stability
    sim_cfg.physx.solver_type = 1  # TGS (Temporal Gauss-Seidel) usually more stable
    sim_cfg.physx.solver_position_iteration_count = 8
    sim_cfg.physx.solver_velocity_iteration_count = 4
    sim = SimulationContext(sim_cfg)

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    scene.clone_environments(copy_from_source=True)

    object_ids = torch.randint(
        low=0, high=object_number, size=(args_cli.num_envs,), device="cpu"
    ).tolist()

    for env_id, object_id in enumerate(object_ids):
        object_cfg = object_list[object_id]
        
        spawn = object_cfg.spawn.replace(
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0)
        )

        object_cfg = object_cfg.replace(
            prim_path=f"/World/envs/env_{env_id}/object",
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 1.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),)
        obj = RigidObject(object_cfg)
        scene.rigid_objects[f"env_{env_id}_object"] = obj

    sim.reset()
    print("[INFO] Scene ready. Running... (close window to exit)")

    dt = sim.get_physics_dt()
    while simulation_app.is_running():
        sim.render()
        sim.step()
        scene.update(dt)


if __name__ == "__main__":
    # run the main function
    main()      # type: ignore
    # close sim app
    simulation_app.close()