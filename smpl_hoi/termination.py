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

class max_timesteps(Termination[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.max_timesteps = self.command_manager.motion.num_frames

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        return (self.env.episode_length_buf >= self.max_timesteps).unsqueeze(1)