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

class track_kp(Reward[SMPLHOITask]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        self.robot = self.command_manager.robot
        self.key_body_ids = self.command_manager.key_body_ids

        self.std_pos = 0.3 ** 2
        self.std_ori = 0.4 ** 2

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf - 1

        ref_kp_pos = self.command_manager.motion.body_pos_w[t][:, self.key_body_ids, :]
        ref_kp_pos.add_(self.command_manager.env_origin[:, None])
        ref_kp_quat = self.command_manager.motion.body_quat_w[t][:, self.key_body_ids, :]

        body_pos = self.robot.data.body_pos_w[:, self.key_body_ids, :]
        body_quat = self.robot.data.body_quat_w[:, self.key_body_ids, :]

        error_pos = (ref_kp_pos - body_pos).square().sum(dim=-1).mean(-1, True)
        error_ori = (quat_error_magnitude(ref_kp_quat, body_quat) ** 2).mean(-1, True)
        rp, rr = torch.exp(-error_pos / self.std_pos), torch.exp(-error_ori / self.std_ori)
        return rp * rr

class track_obj(Reward[SMPLHOITask]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        self.robot = self.command_manager.robot
        self.object = self.command_manager.object

        self.std_pos = 0.3 ** 2
        self.std_ori = 0.4 ** 2

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf - 1

        ref_obj_pos = self.command_manager.motion.object_pos_w[t]
        ref_obj_pos.add_(self.command_manager.env_origin)
        ref_obj_quat = self.command_manager.motion.object_quat_w[t]

        obj_pos = self.object.data.root_pos_w
        obj_quat = self.object.data.root_quat_w
        error_pos = (ref_obj_pos - obj_pos).square().sum(-1, True)
        error_ori = (quat_error_magnitude(ref_obj_quat, obj_quat) ** 2).unsqueeze(-1)
        rp, rr = torch.exp(-error_pos / self.std_pos), torch.exp(-error_ori / self.std_ori)
        return rp * rr

class track_cg(Reward[SMPLHOITask]):
    def __init__(self, env, weight: float = 1.0) -> None:
        super().__init__(env, weight)
        self.robot = self.command_manager.robot
        self.contact_sensor = self.env.scene.sensors["contact_forces"]
 
        self.contact_body_ids, self.contact_body_names = self.robot.find_bodies([
              ".*Hip", ".*Knee", ".*Ankle", ".*Toe", 
              "Torso", "Spine", "Chest", "Neck", "Head", 
              ".*Thorax", ".*Shoulder", ".*Elbow", ".*Wrist",
              ])
        self.left_hand_ids, self.left_hand_names = self.robot.find_bodies(["L_Index.*", "L_Middle.*", "L_Pinky.*", "L_Ring.*", "L_Thumb.*"])
        self.right_hand_ids, self.right_hand_names = self.robot.find_bodies(["R_Index.*", "R_Middle.*", "R_Pinky.*", "R_Ring.*", "R_Thumb.*"])

        self.lambda_hand = 5.0
        self.lambda_other = 5.0
        self.lambda_all = 3.0

    def compute(self) -> torch.Tensor:
        t = self.env.episode_length_buf - 1
        ref_human_contact = self.command_manager.motion.contacts[t]
        contact_forces = self.contact_sensor.data.net_forces_w
        human_contact = (contact_forces.norm(dim=-1) > 0.1).float()

        ref_left_contact_hand = ref_human_contact[:, self.left_hand_ids]
        ref_left_contact_hand_any = (ref_left_contact_hand>0.0).any(dim=-1, keepdim=True).float()
        left_hand_contact = human_contact[:, self.left_hand_ids]

        ecg_left = ref_left_contact_hand_any * torch.abs(left_hand_contact - ref_left_contact_hand_any).mean(dim=-1, keepdim=True)
        rcg_left = 0.5 * ( 1+torch.exp(-ecg_left * self.lambda_hand) ) * ref_left_contact_hand_any + ( 1-ref_left_contact_hand_any)

        ref_right_contact_hand = ref_human_contact[:, self.right_hand_ids]
        ref_right_contact_hand_any = (ref_right_contact_hand>0.0).any(dim=-1, keepdim=True).float()
        right_hand_contact = human_contact[:, self.right_hand_ids]

        ecg_right = ref_right_contact_hand_any * torch.abs(right_hand_contact - ref_right_contact_hand_any).mean(dim=-1, keepdim=True)
        rcg_right = 0.5 * ( 1+torch.exp(-ecg_right * self.lambda_hand) ) * ref_right_contact_hand_any + ( 1-ref_right_contact_hand_any)

        ref_other_contact = ref_human_contact[:, self.contact_body_ids]
        ref_other_contact_any = (ref_other_contact>0.0).any(dim=-1, keepdim=True).float()
        other_contact = human_contact[:, self.contact_body_ids]
        ecg_other = (torch.abs(other_contact - ref_other_contact) * ref_other_contact_any).mean(dim=-1, keepdim=True)
        rcg_other = torch.exp(-ecg_other * self.lambda_other)

        no_contact = 1.0 - human_contact
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < 0.0)).mean(dim=-1, keepdim=True)
        rcg_all = torch.exp(-ecg_all * self.lambda_all)

        rcg = rcg_left * rcg_right * rcg_other * rcg_all
        return rcg