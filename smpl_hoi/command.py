import os
import joblib
import torch
from pathlib import Path
import trimesh

from active_adaptation.envs.mdp.base import Command, Reward, Observation, Termination
from isaaclab.utils.math import (
    quat_apply_inverse, quat_inv, quat_mul, 
    subtract_frame_transforms, quat_error_magnitude
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

DATA_ROOT = Path(__file__).parents[1] / "data"
OBJECT_PATH = Path(__file__).parent / "assets" / "objects"
assert OBJECT_PATH.exists(), f"Object path not found: {OBJECT_PATH}"

from smpl_hoi.utils import obj_forward, compute_sdf

class MotionLoader:
    def __init__(self, motion_file: str, object_name: str, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = joblib.load(motion_file)["largebox"]["sub12_largebox_000"]

        mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, object_name, f"{object_name}.obj"), force='mesh')
        object_points, _ = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)
        self.obj_verts = torch.tensor(object_points, dtype=torch.float32, device=device)
        
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.contacts = torch.tensor(data["contact"], dtype=torch.float32, device=device)
        self.object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
        self.object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
        self.object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
        self.object_ang_vel_w = torch.tensor(data["object_ang_vel_w"], dtype=torch.float32, device=device)
        
        # Precompute Interaction Guidance (IG)
        ref_obj_verts_w = obj_forward(self.obj_verts, self.object_pos_w, self.object_quat_w)
        self.ig_w = compute_sdf(self.body_pos_w, ref_obj_verts_w)
        self.ig_b = quat_apply_inverse(self.root_quat_w.unsqueeze(1).expand(-1, self.ig_w.shape[1], -1), self.ig_w)
        
        self.num_frames = self.joint_pos.shape[0]

    @property
    def num_joints(self) -> int:
        return self.joint_pos.shape[1]

    @property
    def num_bodies(self) -> int:
        return self.body_pos_w.shape[1]

    @property
    def root_pos_w(self) -> torch.Tensor:
        'shape (num_frames, 3)'
        return self.body_pos_w[:, 0, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        'shape (num_frames, 4)'
        return self.body_quat_w[:, 0, :4]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        'shape (num_frames, 3)'
        return self.body_lin_vel_w[:, 0, :3]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        'shape (num_frames, 3)'
        return self.body_ang_vel_w[:, 0, :3]

class SMPLHOITask(Command):
    def __init__(
        self, 
        env, 
        motion_file: str, 
        key_body: list[str],
    ):
        super().__init__(env)
        motion_file = DATA_ROOT / motion_file

        self.robot = env.scene["robot"]
        object_name = self.env.cfg.objects[0].name
        self.object = env.scene[object_name]
        self.env_origin = self.env.scene.env_origins

        self.key_body_ids, self.key_body_names = self.robot.find_bodies(key_body)

        self.motion = MotionLoader(motion_file, object_name, device=self.device)

        assert self.motion.num_joints == self.robot.num_joints
        assert self.motion.num_bodies == self.robot.num_bodies

    def sample_init(self, env_ids: torch.Tensor) -> dict:
        init_root_state = self.init_root_state[env_ids]
        init_root_state[:, :3] = self.motion.root_pos_w[0] + self.env_origin[env_ids]
        init_root_state[:, 3:7] = self.motion.root_quat_w[0]
        init_root_state[:, 7:10] = self.motion.root_lin_vel_w[0]
        init_root_state[:, 10:] = self.motion.root_ang_vel_w[0]

        joint_pos = self.motion.joint_pos[0]
        joint_vel = self.motion.joint_vel[0]
        self.robot.write_joint_state_to_sim(
            joint_pos, 
            joint_vel,
            joint_ids = slice(None),
            env_ids = env_ids,
        )

        init_object_state = self.object.data.default_root_state.clone()[env_ids]
        init_object_state[:, :3] = self.motion.object_pos_w[0] + self.env_origin[env_ids]
        init_object_state[:, 3:7] = self.motion.object_quat_w[0]
        init_object_state[:, 7:10] = self.motion.object_lin_vel_w[0]
        init_object_state[:, 10:] = self.motion.object_ang_vel_w[0]

        return {"robot": init_root_state, "largebox": init_object_state}

    # def update(self):
    #     t = self.env.episode_length_buf
    #     max_t = self.motion.num_frames - 1
    #     t = torch.clamp(t, max=max_t)

    #     root_state = self.init_root_state.clone()
    #     root_state[:, :3] = self.motion.root_pos_w[t] + self.env_origin
    #     root_state[:, 3:7] = self.motion.root_quat_w[t]
    #     root_state[:, 7:10] = self.motion.root_lin_vel_w[t]
    #     root_state[:, 10:] = self.motion.root_ang_vel_w[t]
    #     self.robot.write_root_state_to_sim(root_state)

    #     joint_pos = self.motion.joint_pos[t]
    #     joint_vel = self.motion.joint_vel[t]
    #     self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

    #     object_state = self.object.data.default_root_state.clone()
    #     object_state[:, :3] = self.motion.object_pos_w[t] + self.env_origin
    #     object_state[:, 3:7] = self.motion.object_quat_w[t]
    #     object_state[:, 7:10] = self.motion.object_lin_vel_w[t]
    #     object_state[:, 10:] = self.motion.object_ang_vel_w[t]
    #     self.object.write_root_state_to_sim(object_state)