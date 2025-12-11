import os
import joblib
import torch
from pathlib import Path

from active_adaptation.envs.mdp.base import Command, Reward, Observation, Termination
from isaaclab.utils.math import (
    quat_apply_inverse, quat_inv, quat_mul, 
    subtract_frame_transforms, quat_error_magnitude
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor

DATA_ROOT = Path(__file__).parents[1] / "data"
BPS_PATH = DATA_ROOT / "bps.pt"
assert BPS_PATH.exists(), f"BPS file not found: {BPS_PATH}"

class MotionLoader:
    def __init__(self, motion_file: str, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = joblib.load(motion_file)["largebox"]["sub12_largebox_000"]
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.contacts = torch.tensor(data["contact"], dtype=torch.float32, device=device)
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
        contact_body: list[str],
    ):
        super().__init__(env)
        motion_file = DATA_ROOT / motion_file

        self.robot = env.scene["robot"]
        self.env_origin = self.env.scene.env_origins

        self.key_body_ids, self.key_body_names = self.robot.find_bodies(key_body)
        self.contact_body_ids, self.contact_body_names = self.robot.find_bodies(contact_body)

        self.motion = MotionLoader(motion_file, device=self.device)
        self.bps = torch.load(BPS_PATH).to(torch.float32).to(self.device)

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

        return {"robot": init_root_state}

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