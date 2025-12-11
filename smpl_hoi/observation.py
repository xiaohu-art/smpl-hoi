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

from .command import SMPLHOITask

class root_height(Observation[SMPLHOITask]):
    def compute(self) -> torch.Tensor:
        robot = self.command_manager.robot
        return robot.data.root_pos_w[:, 2].reshape(self.num_envs, -1)

'''TODO: MERGE TO BODY OBSERVATION'''
class body_pos_local(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env) 
        self.robot = self.command_manager.robot

    def compute(self) -> torch.Tensor:
        body_pos_w = self.robot.data.body_pos_w
        root_pos_w = self.robot.data.root_pos_w.unsqueeze(1)
        root_quat_w = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1)
        body_pos_b = quat_apply_inverse(root_quat_w, body_pos_w - root_pos_w)[:, 1:, :]
        return body_pos_b.reshape(self.num_envs, -1)
    
class body_ori_local(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot

    def compute(self) -> torch.Tensor:
        root_quat_w = self.robot.data.root_quat_w.unsqueeze(1)
        body_quat_w = self.robot.data.body_quat_w
        body_ori_b = quat_mul(
            quat_inv(root_quat_w).expand(-1, body_quat_w.shape[1], -1),
            body_quat_w
        )[:, 1:, :4]
        return torch.cat([
            root_quat_w, body_ori_b,
        ], dim=1).reshape(self.num_envs, -1)

class body_linvel_local(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot

    def compute(self) -> torch.Tensor:
        body_linvel_w = self.robot.data.body_lin_vel_w
        root_quat_w = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, body_linvel_w.shape[1], -1)
        body_linvel_b = quat_apply_inverse(root_quat_w, body_linvel_w)
        return body_linvel_b.reshape(self.num_envs, -1)

class body_angvel_local(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot

    def compute(self) -> torch.Tensor:
        body_angvel_w = self.robot.data.body_ang_vel_w
        root_quat_w = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, body_angvel_w.shape[1], -1)
        body_angvel_b = quat_apply_inverse(root_quat_w, body_angvel_w)
        return body_angvel_b.reshape(self.num_envs, -1)

class contact(Observation[SMPLHOITask]):
    def __init__(self, env, threshold: float = 0.1) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.threshold = threshold
        self.contact_body_ids = self.command_manager.contact_body_ids
        self.contact_sensor = self.env.scene["contact_forces"]

    def compute(self) -> torch.Tensor:
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_body_ids, :]
        contact_norm = torch.norm(contact_forces, dim=-1)
        contact_flag = (contact_norm > self.threshold).float()
        return contact_flag.reshape(self.num_envs, -1)

class ref_body_gap(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf
        max_t = self.command_manager.motion.num_frames - 1
        t = torch.clamp(t, max=max_t)

        ref_kp_pos = self.command_manager.motion.body_pos_w[t][:, self.key_body_ids, :]
        ref_kp_pos.add_(self.command_manager.env_origin[:, None])
        ref_kp_quat = self.command_manager.motion.body_quat_w[t][:, self.key_body_ids, :]

        body_pos = self.robot.data.body_pos_w[:, self.key_body_ids, :]
        body_quat = self.robot.data.body_quat_w[:, self.key_body_ids, :]

        pos, quat = subtract_frame_transforms(body_pos, body_quat, ref_kp_pos, ref_kp_quat)
        return torch.cat([
            pos.reshape(self.num_envs, -1),
            quat.reshape(self.num_envs, -1),
        ], dim=-1).reshape(self.num_envs, -1)

class ref_body_linvel_gap(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf
        max_t = self.command_manager.motion.num_frames - 1
        t = torch.clamp(t, max=max_t)

        ref_kp_linvel_w = self.command_manager.motion.body_lin_vel_w[t][:, self.key_body_ids, :]
        body_linvel_w = self.robot.data.body_lin_vel_w[:, self.key_body_ids, :]
        root_quat_w = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, body_linvel_w.shape[1], -1)
        diff_body_linvel_b = quat_apply_inverse(root_quat_w, ref_kp_linvel_w - body_linvel_w)
        return diff_body_linvel_b.reshape(self.num_envs, -1)

class ref_body_angvel_gap(Observation[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf
        max_t = self.command_manager.motion.num_frames - 1
        t = torch.clamp(t, max=max_t)

        ref_kp_angvel_w = self.command_manager.motion.body_ang_vel_w[t][:, self.key_body_ids, :]
        body_angvel_w = self.robot.data.body_ang_vel_w[:, self.key_body_ids, :]
        root_quat_w = self.robot.data.root_quat_w.unsqueeze(1).expand(-1, body_angvel_w.shape[1], -1)
        diff_body_angvel_b = quat_apply_inverse(root_quat_w, ref_kp_angvel_w - body_angvel_w)
        return diff_body_angvel_b.reshape(self.num_envs, -1)