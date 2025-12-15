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

from smpl_hoi.command import SMPLHOITask
from smpl_hoi.utils import obj_forward, compute_sdf

class max_timesteps(Termination[SMPLHOITask]):
    def __init__(self, env) -> None:
        super().__init__(env)
        self.max_timesteps = self.command_manager.motion.num_frames

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        return (self.env.episode_length_buf >= self.max_timesteps).unsqueeze(1)

class body_pos_reset(Termination[SMPLHOITask]):
    def __init__(self, env, threshold: float = 0.5) -> None:
        super().__init__(env)
        self.threshold = threshold
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        t = self.env.episode_length_buf - 1

        ref_kp_pos = self.command_manager.motion.body_pos_w[t][:, self.key_body_ids, :]
        ref_kp_pos.add_(self.command_manager.env_origin[:, None])
        body_pos = self.robot.data.body_pos_w[:, self.key_body_ids, :]
        return (ref_kp_pos - body_pos).norm(dim=-1).mean(-1, True) > self.threshold

class obj_pos_reset(Termination[SMPLHOITask]):
    def __init__(self, env, threshold: float = 0.5) -> None:
        super().__init__(env)
        self.threshold = threshold
        self.object = self.command_manager.object
        self.object_verts = self.command_manager.motion.obj_verts

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        t = self.env.episode_length_buf - 1
        
        ref_obj_pos = self.command_manager.motion.object_pos_w[t]
        ref_obj_pos.add_(self.command_manager.env_origin)
        ref_obj_quat = self.command_manager.motion.object_quat_w[t]
        ref_obj_verts_w = obj_forward(self.object_verts, ref_obj_pos, ref_obj_quat)

        obj_pos_w = self.object.data.root_pos_w
        obj_quat_w = self.object.data.root_quat_w
        obj_verts_w = obj_forward(self.object_verts, obj_pos_w, obj_quat_w)
        return (ref_obj_verts_w - obj_verts_w).norm(dim=-1).mean(-1, True) > self.threshold

class cg_reset(Termination[SMPLHOITask]):
    def __init__(self, env, length: int = 10) -> None:
        super().__init__(env)
        self.robot = self.command_manager.robot
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
 
        self.left_hand_ids, self.left_hand_names = self.robot.find_bodies(["L_Index.*", "L_Middle.*", "L_Pinky.*", "L_Ring.*", "L_Thumb.*"])
        self.right_hand_ids, self.right_hand_names = self.robot.find_bodies(["R_Index.*", "R_Middle.*", "R_Pinky.*", "R_Ring.*", "R_Thumb.*"])

        self.length = length
        self.contact_reset = torch.zeros((self.num_envs, 2), device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.contact_reset[env_ids] = 0

    def compute(self, termination: torch.Tensor) -> torch.Tensor:
        t = self.env.episode_length_buf - 1
        ref_human_contact = self.command_manager.motion.contacts[t]
        contact_forces = self.contact_sensor.data.net_forces_w
        human_contact = (contact_forces.norm(dim=-1) > 0.1).float()

        ref_left_contact_hand = ref_human_contact[:, self.left_hand_ids]
        ref_left_contact_hand_any = ref_left_contact_hand.any(dim=-1, keepdim=True).float()
        left_hand_contact = human_contact[:, self.left_hand_ids]
        left_hand_contact_any = left_hand_contact.any(dim=-1, keepdim=True).float()

        ref_right_contact_hand = ref_human_contact[:, self.right_hand_ids]
        ref_right_contact_hand_any = ref_right_contact_hand.any(dim=-1, keepdim=True).float()
        right_hand_contact = human_contact[:, self.right_hand_ids]
        right_hand_contact_any = right_hand_contact.any(dim=-1, keepdim=True).float()

        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any - left_hand_contact_any) * ref_left_contact_hand_any, 
                                torch.abs(ref_right_contact_hand_any - right_hand_contact_any) * ref_right_contact_hand_any,
                                ], dim=-1)

        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        return (self.contact_reset > self.length).any(dim=-1, keepdim=True)
