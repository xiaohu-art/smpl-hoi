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

'''TODO: REMOVE KEY_BODY_IDS'''
class track_kp_pos(Reward[SMPLHOITask]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf - 1

        ref_kp_pos = self.command_manager.motion.body_pos_w[t][:, self.key_body_ids, :]
        ref_kp_pos.add_(self.command_manager.env_origin[:, None])
        body_pos = self.robot.data.body_pos_w[:, self.key_body_ids, :]

        error = ((ref_kp_pos - body_pos)**2).sum(dim=-1).mean(-1, True)
        return torch.exp(-error)

class track_kp_ori(Reward[SMPLHOITask]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf - 1
        ref_kp_quat = self.command_manager.motion.body_quat_w[t][:, self.key_body_ids, :]
        body_quat = self.robot.data.body_quat_w[:, self.key_body_ids, :]
        error = (quat_error_magnitude(ref_kp_quat, body_quat) ** 2).mean(-1, True)
        return torch.exp(-error)