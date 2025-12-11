"""This script processes multiple motions from a pkl file and saves collected simulation data back to the original pkl file.

.. code-block:: bash

    # Usage
    python load_pkl.py --input_file motions.pkl --input_fps 30 --output_fps 50 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Process multiple motions from pkl file and save collected data back to pkl.")
parser.add_argument("--input_file", type=str, required=True, help="The path to the input motion pkl file containing multiple motions.")
parser.add_argument("--output_file", type=str, required=True, help="The path to the output motion pkl file containing multiple motions.")
parser.add_argument("--input_fps", type=int, default=30, help="The fps of the input motion.")
parser.add_argument("--output_fps", type=int, default=30, help="The fps of the output motion.")
parser.add_argument("--bps_path", type=str, default="./data/bps.pt", help="The path to the bps file.")
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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

from smpl_hoi.asset import SMPL, LARGEBOX

SMPLH_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Index1",
    "L_Index2",
    "L_Index3",
    "L_Middle1",
    "L_Middle2",
    "L_Middle3",
    "L_Pinky1",
    "L_Pinky2",
    "L_Pinky3",
    "L_Ring1",
    "L_Ring2",
    "L_Ring3",
    "L_Thumb1",
    "L_Thumb2",
    "L_Thumb3",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
]

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
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

    # articulation
    robot = SMPL.isaaclab().replace(prim_path="{ENV_REGEX_NS}/Robot")
    largebox = LARGEBOX.isaaclab().replace(prim_path="{ENV_REGEX_NS}/LargeBox")

from pxr import Usd, UsdGeom, Gf, Vt
import omni.usd

def create_points_prim(path: str, point_size: float = 0.01, color=(0.1, 0.8, 1.0)):
    stage = omni.usd.get_context().get_stage()
    pts = UsdGeom.Points.Define(stage, path)
    pts.CreateWidthsAttr().Set(Vt.FloatArray([point_size]))
    pts.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    pts.CreatePointsAttr().Set(Vt.Vec3fArray())
    return pts

def set_points(pts_prim, points):  
    """
    points: (P,3) / (N, P, 3) float32/float64
    """
    if points.ndim == 3:
        points = points[0]
    P = int(points.shape[0])
    arr = Vt.Vec3fArray(P)
    arr[:] = [Gf.Vec3f(float(x), float(y), float(z)) for x,y,z in points]
    pts_prim.GetPointsAttr().Set(arr)

bps_points_prim = create_points_prim("/World/BPSPointCloud", point_size=0.04, color=(0.2, 0.9, 0.9))
contact_body_points_prim_pos = create_points_prim("/World/ContactBodyPointsPos", point_size=0.04, color=(0.2, 0.9, 0.2))
contact_body_points_prim_neg = create_points_prim("/World/ContactBodyPointsNeg", point_size=0.08, color=(0.9, 0.2, 0.2))

def render_bps_points(bps, bps_object_geo, obj_rot, obj_trans):
    """
    bps: torch.Tensor, shape (num point, 3)
    bps_object_geo: torch.Tensor, shape (num point, 3)
    obj_rot: torch.Tensor, shape (num_envs, 4) quaternion (w, x, y, z)
    obj_trans: torch.Tensor, shape (num_envs, 3) translation (x, y, z)
    """
    N, P = obj_rot.shape[0], bps.shape[0]
    obj_rot_matrix = matrix_from_quat(obj_rot)    # (N, 3, 3)

    bps_object_cloud = (bps + bps_object_geo).unsqueeze(0).expand(N, -1, -1)         # (N, P, 3)
    bps_object_cloud = bps_object_cloud @ obj_rot_matrix.transpose(1, 2) + obj_trans[:, None, :]
    set_points(bps_points_prim, bps_object_cloud)

def render_contact_body_points(body_pos_w, contact):
    """
    body_pos_w: torch.Tensor, shape (N, 3)
    contact: torch.Tensor, shape (N)
    """
    contact_np = contact.cpu().numpy()
    
    # Positive contacts (1)
    mask_pos = (contact_np > 0.5)
    points_pos = body_pos_w[mask_pos].cpu().numpy().copy() if mask_pos.any() else np.empty((0, 3))
    set_points(contact_body_points_prim_pos, points_pos)

    # Negative contacts (-1)
    mask_neg = (contact_np < -0.5)
    points_neg = body_pos_w[mask_neg].cpu().numpy().copy() if mask_neg.any() else np.empty((0, 3))
    set_points(contact_body_points_prim_neg, points_neg)

class MotionLoader:
    def __init__(
        self,
        bps_path: str,
        human_data: dict,
        object_data: dict,
        input_fps: int,
        output_fps: int,
        device: torch.device,
    ):
        self.bps = torch.load(bps_path)
        self.human_data = human_data
        self.object_data = object_data
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the motion data dict."""

        human_motion = self.human_data          # poses, betas, trans, gender, dof_pos, keypoints, contact
        object_motion = self.object_data        # rot, trans, name, scale, bps_object_geo

        self.human_base_poss_input = torch.from_numpy(human_motion["trans"]).to(self.device)    # T X 3
        poses = human_motion["poses"].reshape(-1, 52, 3)                                        # T X 52 X 3 rotation vector
        T, J, _ = poses.shape
        root_pose_aa = poses[:, 0, :]
        root_rot = sRot.from_rotvec(root_pose_aa.reshape(T, 3)).as_quat()[:, [3, 0, 1, 2]]
        self.human_base_rots_input = torch.from_numpy(root_rot).to(self.device)
        body_pose_aa = poses[:, 1:, :]
        body_pos = sRot.from_rotvec(body_pose_aa.reshape(T*(J-1), 3)).as_euler("XYZ").reshape(T, (J-1)*3)
        self.human_dof_poss_input = torch.from_numpy(body_pos).to(self.device).reshape(T, (J-1)*3)    
        
        self.human_base_poss_input = self.human_base_poss_input.to(torch.float32)
        self.human_base_rots_input = self.human_base_rots_input.to(torch.float32)
        self.human_dof_poss_input = self.human_dof_poss_input.to(torch.float32)

        self.human_contact = torch.from_numpy(human_motion["contacts"]).to(self.device)
        self.human_contact = self.human_contact.to(torch.float32)

        self.object_base_poss_input = torch.from_numpy(object_motion["trans"]).to(self.device)    # T X 3
        self.object_base_rots_input = torch.from_numpy(object_motion["rot"]).to(self.device)
        self.object_base_rots_input = quat_unique(quat_from_matrix(self.object_base_rots_input))    # T X 4

        self.object_base_poss_input = self.object_base_poss_input.to(torch.float32)
        self.object_base_rots_input = self.object_base_rots_input.to(torch.float32)

        self.bps = self.bps[0].to(self.device)        # (N, 3)
        self.bps_object_geo = torch.from_numpy(object_motion["bps_object_geo"]).to(self.device)
        self.bps = self.bps.to(torch.float32)
        self.bps_object_geo = self.bps_object_geo.to(torch.float32)

        self.input_frames = self.human_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt
        print(f"Motion loaded, duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.human_base_poss = self._lerp(
            self.human_base_poss_input[index_0],
            self.human_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.human_base_rots = self._slerp(
            self.human_base_rots_input[index_0],
            self.human_base_rots_input[index_1],
            blend,
        )
        self.human_dof_poss = self._lerp(
            self.human_dof_poss_input[index_0],
            self.human_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.human_contact = self._lerp(
            self.human_contact[index_0],
            self.human_contact[index_1],
            blend.unsqueeze(1),
        )
        self.object_base_poss = self._lerp(
            self.object_base_poss_input[index_0],
            self.object_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.object_base_rots = self._slerp(
            self.object_base_rots_input[index_0],
            self.object_base_rots_input[index_1],
            blend,
        )

        assert self.object_base_poss.shape[0] == self.human_base_poss.shape[0] == self.output_frames
        print(
            f"Human motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.human_base_lin_vels = torch.gradient(self.human_base_poss, spacing=self.output_dt, dim=0)[0]
        self.human_dof_vels = torch.gradient(self.human_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.human_base_ang_vels = self._so3_derivative(self.human_base_rots, self.output_dt)

        self.object_base_lin_vels = torch.gradient(self.object_base_poss, spacing=self.output_dt, dim=0)[0]
        self.object_base_ang_vels = self._so3_derivative(self.object_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        bool,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.human_base_poss[self.current_idx : self.current_idx + 1],
            self.human_base_rots[self.current_idx : self.current_idx + 1],
            self.human_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.human_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.human_dof_poss[self.current_idx : self.current_idx + 1],
            self.human_dof_vels[self.current_idx : self.current_idx + 1],
            self.human_contact[self.current_idx : self.current_idx + 1],
            self.object_base_poss[self.current_idx : self.current_idx + 1],
            self.object_base_rots[self.current_idx : self.current_idx + 1],
            self.object_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.object_base_ang_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def process_single_motion(sim: sim_utils.SimulationContext, scene: InteractiveScene, 
                         joint_names: list[str], body_names: list[str],
                         human_data: dict, object_data: dict) -> dict:
    """Processes a single motion and returns collected simulation data."""
    # Load motion
    motion = MotionLoader(
        bps_path=args_cli.bps_path,
        human_data=human_data,
        object_data=object_data,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes, robot_joint_names = robot.find_joints(joint_names, preserve_order=True)
    robot_body_indexes, robot_body_names = robot.find_bodies(body_names, preserve_order=True)
    assert len(robot_body_indexes) == len(robot.body_names)

    object = scene["largebox"]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "object_pos_w": [],
        "object_quat_w": [],
        "object_lin_vel_w": [],
        "object_ang_vel_w": [],
        # ==== BPS Point Cloud ====
        "bps_object_geo": [],
        # ==== Contacts ====
        "contact": []
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                human_base_pos,
                human_base_rot,
                human_base_lin_vel,
                human_base_ang_vel,
                human_dof_pos,
                human_dof_vel,
                human_contact,
                object_base_pos,
                object_base_rot,
                object_base_lin_vel,
                object_base_ang_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = human_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = human_base_rot
        root_states[:, 7:10] = human_base_lin_vel
        root_states[:, 10:] = human_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = human_dof_pos
        joint_vel[:, robot_joint_indexes] = human_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)


        # set object state
        object_state = object.data.default_root_state.clone()
        object_state[:, :3] = object_base_pos
        object_state[:, :2] += scene.env_origins[:, :2]
        object_state[:, 3:7] = object_base_rot
        object_state[:, 7:10] = object_base_lin_vel
        object_state[:, 10:] = object_base_ang_vel
        object.write_root_state_to_sim(object_state)

        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        human_contact_reordered = torch.zeros_like(human_contact)
        human_contact_reordered[:, robot_body_indexes] = human_contact

        render_bps_points(motion.bps, motion.bps_object_geo, object_base_rot, object_base_pos)
        render_contact_body_points(robot.data.body_pos_w[0, :], human_contact_reordered[0, :])

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())
            log["object_pos_w"].append(object.data.root_pos_w[0, :].cpu().numpy().copy())
            log["object_quat_w"].append(object.data.root_quat_w[0, :].cpu().numpy().copy())
            log["object_lin_vel_w"].append(object.data.root_lin_vel_w[0, :].cpu().numpy().copy())
            log["object_ang_vel_w"].append(object.data.root_ang_vel_w[0, :].cpu().numpy().copy())
            log["contact"].append(human_contact_reordered[0, :].cpu().numpy().copy().astype(bool))

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "object_pos_w",
                "object_quat_w",
                "object_lin_vel_w",
                "object_ang_vel_w",
                "contact",
            ):
                log[k] = np.stack(log[k], axis=0)

            log["bps_object_geo"] = motion.bps_object_geo.cpu().numpy().copy()
            print(f"[INFO]: Motion processed successfully")
            break
    
    return log


def run_simulator(  sim: sim_utils.SimulationContext, scene: InteractiveScene, 
                    joint_names: list[str], 
                    body_names: list[str]
                ):
    """Runs the simulation loop for multiple motions."""
    # Load the pkl file containing multiple motions
    print(f"[INFO]: Loading motions from {args_cli.input_file}")
    all_motions = joblib.load(args_cli.input_file)

    processed_motions = {}
    for object_name in all_motions.keys():
        processed_motions[object_name] = {}
        for sub_object_name in all_motions[object_name].keys():
            human_data = all_motions[object_name][sub_object_name]["human"]
            object_data = all_motions[object_name][sub_object_name]["object"]
        
            # Process the motion and collect simulation data
            simulation_data = process_single_motion(
                                    sim, 
                                    scene, 
                                    joint_names, 
                                    body_names, 
                                    human_data, 
                                    object_data
                                )
            processed_motions[object_name][sub_object_name] = simulation_data
        
    # Save the updated pkl file
    print(f"[INFO]: Saving updated motions to {args_cli.output_file}")
    joblib.dump(processed_motions, args_cli.output_file)
    print(f"[INFO]: Successfully processed and saved {len(processed_motions)} motions")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    AXES = ["x", "y", "z"]
    JOINT_ORDER = [
        n for n in SMPLH_BONE_ORDER_NAMES if n != "Pelvis"
    ]
    joint_names = [f"{base}_{a}" for base in JOINT_ORDER for a in AXES]

    run_simulator(
        sim,
        scene,
        joint_names=joint_names,
        body_names=SMPLH_BONE_ORDER_NAMES
    )


if __name__ == "__main__":
    # run the main function
    main()      # type: ignore
    # close sim app
    simulation_app.close()