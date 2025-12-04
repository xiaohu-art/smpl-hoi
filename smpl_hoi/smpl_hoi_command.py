import torch
import torch.distributions as D
import torch.nn.functional as F
from typing import List, TYPE_CHECKING, Union, Optional
from pathlib import Path

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
 
from active_adaptation.utils.math import quat_rotate, quat_rotate_inverse
from active_adaptation.envs.mdp.base import Command

import joblib
import os

CURRENT_MOTION = 0
DATA_ROOT = Path(__file__).parent / "data"

class HOITracking(Command):
    def __init__(
            self, 
            env,
            motion_clip_dir: str,
            data_path: str,
            keypoint_body: Optional[List[str]] = [
                                    "Pelvis", 
                                    ".*Hip.*", 
                                    "Torso", 
                                    ".*Knee.*", 
                                    "Spine", 
                                    ".*Ankle.*", 
                                    "Chest", 
                                    ".*Toe.*", 
                                    "Neck", 
                                    ".*Thorax.*",
                                    "Head", 
                                    ".*Shoulder.*", 
                                    ".*Elbow.*", 
                                    ".*Wrist.*", 
                                    ".*Index.*", ".*Middle.*", ".*Pinky.*", ".*Ring.*", ".*Thumb.*",
                                ],
            mode: str = "train",
            eval_id: Optional[Union[int, tuple[int, int]]] = None,
            teleop: bool = False,
        ):
        super().__init__(env, teleop=teleop)
        self.robot: Articulation = self.env.scene["robot"]
        self.object: RigidObject = self.env.scene["cloth_stand"]

        self.env_origin = self.env.scene.env_origins
        self.keypoint_body_index, _ = self.robot.find_bodies(keypoint_body)

        data_path = DATA_ROOT / "omomo_test.pkl"
        data = joblib.load(data_path)
        print(f"Loaded motion clips from {data_path}")

        bps_path = DATA_ROOT / "bps.pt"
        self.bps = torch.load(bps_path).to(self.device)
        self.bps = self.bps.to(torch.float32)           # [1, num_points, 3]
        
        self._per_env_fixed = False
        self._motion_for_env: Optional[torch.Tensor] = None

        if eval_id is not None:
            data_keys = list(data.keys())
            if isinstance(eval_id, int):
                # Single motion by index
                idx = int(eval_id)
                assert 0 <= idx < len(data_keys), (
                f"eval_id {idx} out of range for {len(data_keys)} motions"
                )
                data = {data_keys[idx]: data[data_keys[idx]]}
            else:
                start, end = eval_id
                assert 0 <= start < end <= len(data_keys), (
                f"Invalid eval_id slice {eval_id}; total motions: {len(data_keys)}"
                )
                keep_keys = data_keys[start:end]
                num_eval_motion = end - start
                assert self.num_envs == num_eval_motion, (
                f"num_envs ({self.num_envs}) must equal slice length ({num_eval_motion})"
                )
                data = {k: data[k] for k in keep_keys}
                # per-env fixed mapping: env i -> motion i
                self._per_env_fixed = True
                self._motion_for_env = torch.arange(num_eval_motion, device=self.device)

        self.load_data(data)
        assert len(self.robot.body_names) == self.body_pos_w.shape[1]
        assert len(self.robot.joint_names) == self.joint_pos.shape[1]

        if self._per_env_fixed:
            self.num_frames = int(self.motion_length.max())
        else:
            self.num_frames = int(self.joint_pos.shape[0])
        print(f"Loaded {len(data)} motion clips with {self.num_frames} frames.")

        BASELINE_MASS = 0.02
        self.min_weight = BASELINE_MASS / self.num_motions
        self.alpha0, self.beta0 = 1.0, 1.0
        self.trials = torch.zeros(self.num_motions, device=self.device)
        self.failures = torch.zeros(self.num_motions, device=self.device)
        self.curr_motion_id = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.mode = mode

    @torch.no_grad()
    def _sampling_probs(self) -> torch.Tensor:
        denom = (self.trials + self.alpha0 + self.beta0).clamp_min(1e-6)
        p_fail = (self.failures + self.alpha0) / denom

        w = p_fail.clamp_min(self.min_weight)
        return w / w.sum()

    def _pick_motion_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return a (len(env_ids),) tensor of motion ids for these envs, honoring mode & slicing."""
        # Deterministic mapping: env i -> motion i
        if self._per_env_fixed:
            assert self._motion_for_env is not None
            return self._motion_for_env[env_ids]

        # In eval/play without slicing, use CURRENT_MOTION for all selected envs
        if self.mode in ("play", "eval"):
            motion_id = int(CURRENT_MOTION % max(self.num_motions, 1))
            return torch.full((env_ids.shape[0],), motion_id, dtype=torch.long, device=self.device)

        # Default: train — weighted by failure-biased bandit
        probs = self._sampling_probs().to(self.device)
        return D.Categorical(probs).sample((env_ids.shape[0],))
    
    def sample_init(self, env_ids: torch.Tensor) -> torch.Tensor:
        motion_ids = self._pick_motion_ids(env_ids)
        self.curr_motion_id[env_ids] = motion_ids
        
        start_frames = self.start_frames[motion_ids]
        init_root_state = self.init_root_state[env_ids]     # (num_envs, 3 + 4 + 6) root position, root orientation, root linear velocity and root angular velocity
        init_root_state[:, :3] = self.root_pos_w[start_frames].to(self.device) + self.env_origin[env_ids]
        # init_root_state[:, :3] += torch.tensor([0, 0, 0.05], device=self.device)
        init_root_state[:, 3:7] = self.root_quat_w[start_frames].to(self.device)
        init_root_state[:, 7:10] = self.root_lin_vel_w[start_frames].to(self.device)
        init_root_state[:, 10:] = self.root_ang_vel_w[start_frames].to(self.device)

        init_object_state = self.object.data.default_root_state.clone()[env_ids]  # Select only the env_ids
        init_object_state[:, :3] = self.object_body_pos_w[start_frames] + self.env_origin[env_ids]
        init_object_state[:, 3:7] = self.object_body_quat_w[start_frames]
        init_object_state[:, 7:10] = self.object_body_lin_vel_w[start_frames]
        init_object_state[:, 10:] = self.object_body_ang_vel_w[start_frames]

        return {"robot": init_root_state, "cloth_stand": init_object_state}
    
    def reset(self, env_ids: torch.Tensor):
        pass

    def _update_stats(self, env_ids: torch.Tensor):
        mids = self.curr_motion_id[env_ids]
        valid = mids >= 0
        if valid.any():
            success = (self.env.stats["success"][env_ids].squeeze(-1) > 0.5)
            failed = (~success).to(self.trials.dtype)

            ones = torch.ones_like(failed, dtype=self.trials.dtype)

            self.trials.index_add_(0, mids[valid], ones[valid])
            self.failures.index_add_(0, mids[valid], failed[valid])

        self.curr_motion_id[env_ids] = -1
        
    def load_data(self, data):
        self.motion_length = []
        self.joint_pos = []
        self.joint_vel = []
        self.body_pos_w = []
        self.body_quat_w = []
        self.body_lin_vel_w = []
        self.body_ang_vel_w = []
        self.object_body_pos_w = []
        self.object_body_quat_w = []
        self.object_body_lin_vel_w = []
        self.object_body_ang_vel_w = []
        self.keypoints = []
        self.contacts = []
        self.bps_object_geo = []

        for object_name in data.keys():
            for sub_object_name in data[object_name].keys():
                motion = data[object_name][sub_object_name]
                joint_pos = torch.from_numpy(motion["joint_pos"])
                joint_vel = torch.from_numpy(motion["joint_vel"])
                body_pos_w = torch.from_numpy(motion["body_pos_w"])
                body_quat_w = torch.from_numpy(motion["body_quat_w"])
                body_lin_vel_w = torch.from_numpy(motion["body_lin_vel_w"])
                body_ang_vel_w = torch.from_numpy(motion["body_ang_vel_w"])
                object_body_pos_w = torch.from_numpy(motion["object_body_pos_w"])
                object_body_quat_w = torch.from_numpy(motion["object_body_quat_w"])
                object_body_lin_vel_w = torch.from_numpy(motion["object_body_lin_vel_w"])
                object_body_ang_vel_w = torch.from_numpy(motion["object_body_ang_vel_w"])
                keypoints = torch.from_numpy(motion["keypoints"])
                contacts = torch.from_numpy(motion["contact"])
                bps_object_geo = torch.from_numpy(motion["bps_object_geo"])

                self.motion_length.append(joint_pos.shape[0])
                self.bps_object_geo.append(bps_object_geo.to(self.device))
                self.joint_pos.append(joint_pos)
                self.joint_vel.append(joint_vel)
                self.body_pos_w.append(body_pos_w)
                self.body_quat_w.append(body_quat_w)
                self.body_lin_vel_w.append(body_lin_vel_w)
                self.body_ang_vel_w.append(body_ang_vel_w)
                self.object_body_pos_w.append(object_body_pos_w)
                self.object_body_quat_w.append(object_body_quat_w)
                self.object_body_lin_vel_w.append(object_body_lin_vel_w)
                self.object_body_ang_vel_w.append(object_body_ang_vel_w)
                self.keypoints.append(keypoints)
                self.contacts.append(contacts)

        self.motion_length = torch.tensor(self.motion_length)
        self.bps_object_geo = torch.stack([geo.float() for geo in self.bps_object_geo], dim=0).to(self.device)
        self.joint_pos = torch.cat(self.joint_pos, dim=0).float().to(self.device)
        self.joint_vel = torch.cat(self.joint_vel, dim=0).float().to(self.device)
        self.body_pos_w = torch.cat(self.body_pos_w, dim=0).float().to(self.device)
        self.body_quat_w = torch.cat(self.body_quat_w, dim=0).float().to(self.device)
        self.body_lin_vel_w = torch.cat(self.body_lin_vel_w, dim=0).float().to(self.device)
        self.body_ang_vel_w = torch.cat(self.body_ang_vel_w, dim=0).float().to(self.device)

        self.object_body_pos_w = torch.cat(self.object_body_pos_w, dim=0).float().to(self.device)
        self.object_body_quat_w = torch.cat(self.object_body_quat_w, dim=0).float().to(self.device)
        self.object_body_lin_vel_w = torch.cat(self.object_body_lin_vel_w, dim=0).float().to(self.device)
        self.object_body_ang_vel_w = torch.cat(self.object_body_ang_vel_w, dim=0).float().to(self.device)
        self.object_body_pos_w = self.object_body_pos_w.reshape(-1, 3)
        self.object_body_quat_w = self.object_body_quat_w.reshape(-1, 4)
        self.object_body_lin_vel_w = self.object_body_lin_vel_w.reshape(-1, 3)
        self.object_body_ang_vel_w = self.object_body_ang_vel_w.reshape(-1, 3)
        
        self.keypoints = torch.cat(self.keypoints, dim=0).to(self.device)   # [T, num_keypoints, 3]
        self.contacts = torch.cat(self.contacts, dim=0).to(self.device)   # [T, num_keypoints]
        self.root_pos_w = self.body_pos_w[:, 0]
        self.root_quat_w = self.body_quat_w[:, 0]
        self.root_lin_vel_w = self.body_lin_vel_w[:, 0]
        self.root_ang_vel_w = self.body_ang_vel_w[:, 0]

        self.num_motions = len(data)
        self.num_frames = self.joint_pos.shape[0]

        self.start_frames = torch.cat([torch.zeros(1), self.motion_length.cumsum(dim=0)[:-1]]).long().to(self.device)
        self.end_frames = self.motion_length.cumsum(dim=0).long().to(self.device)
        self.motion_length = self.motion_length.to(self.device)

    # # for sanity check
    # def update(self):
    #     mass = self.object.data.default_mass[0].sum()
    #     Kp = torch.tensor(25., device=self.device)
    #     Kd = 2 * torch.sqrt(Kp * mass)

    #     timestep = self.env.episode_length_buf
    #     max_timestep = self.env.max_episode_length - 1
    #     timestep = torch.clamp(timestep, max=max_timestep)
        
    #     object_state = self.object.data.root_state_w.clone()
    #     object_pos, object_vel = object_state[:, :3], object_state[:, 7:10]

    #     object_pos_ref = self.object_body_pos_w[timestep] + self.env_origin
    #     object_vel_ref = self.object_body_lin_vel_w[timestep]

    #     object_pos_error = object_pos_ref - object_pos
    #     object_vel_error = object_vel_ref - object_vel

    #     forces = (Kp * object_pos_error + Kd * object_vel_error).reshape(self.num_envs, -1, 3)
    #     torques = torch.zeros_like(forces)
    #     self.object.set_external_force_and_torque(forces, torques)

SMPLH_BODIES = [
    'Pelvis', 
    'L_Hip', 'R_Hip', 
    'Torso', 
    'L_Knee', 'R_Knee', 
    'Spine', 
    'L_Ankle', 'R_Ankle', 
    'Chest', 
    'L_Toe', 'R_Toe', 
    'Neck', 
    'L_Thorax', 'R_Thorax', 
    'Head', 
    'L_Shoulder', 'R_Shoulder', 
    'L_Elbow', 'R_Elbow', 
    'L_Wrist', 'R_Wrist', 
    'L_Index1', 'L_Middle1', 'L_Pinky1', 'L_Ring1', 'L_Thumb1', 
    'R_Index1', 'R_Middle1', 'R_Pinky1', 'R_Ring1', 'R_Thumb1', 
    'L_Index2', 'L_Middle2', 'L_Pinky2', 'L_Ring2', 'L_Thumb2', 
    'R_Index2', 'R_Middle2', 'R_Pinky2', 'R_Ring2', 'R_Thumb2', 
    'L_Index3', 'L_Middle3', 'L_Pinky3', 'L_Ring3', 'L_Thumb3', 
    'R_Index3', 'R_Middle3', 'R_Pinky3', 'R_Ring3', 'R_Thumb3'
]

SMPLH_JOINTS = [
    'L_Hip_x', 'L_Hip_y', 'L_Hip_z', 
    'R_Hip_x', 'R_Hip_y', 'R_Hip_z', 
    'Torso_x', 'Torso_y', 'Torso_z', 
    'L_Knee_x', 'L_Knee_y', 'L_Knee_z', 
    'R_Knee_x', 'R_Knee_y', 'R_Knee_z', 
    'Spine_x', 'Spine_y', 'Spine_z', 
    'L_Ankle_x', 'L_Ankle_y', 'L_Ankle_z', 
    'R_Ankle_x', 'R_Ankle_y', 'R_Ankle_z', 
    'Chest_x', 'Chest_y', 'Chest_z', 
    'L_Toe_x', 'L_Toe_y', 'L_Toe_z', 
    'R_Toe_x', 'R_Toe_y', 'R_Toe_z', 
    'Neck_x', 'Neck_y', 'Neck_z', 
    'L_Thorax_x', 'L_Thorax_y', 'L_Thorax_z', 
    'R_Thorax_x', 'R_Thorax_y', 'R_Thorax_z', 
    'Head_x', 'Head_y', 'Head_z', 
    'L_Shoulder_x', 'L_Shoulder_y', 'L_Shoulder_z', 
    'R_Shoulder_x', 'R_Shoulder_y', 'R_Shoulder_z', 
    'L_Elbow_x', 'L_Elbow_y', 'L_Elbow_z', 
    'R_Elbow_x', 'R_Elbow_y', 'R_Elbow_z', 
    'L_Wrist_x', 'L_Wrist_y', 'L_Wrist_z', 
    'R_Wrist_x', 'R_Wrist_y', 'R_Wrist_z', 
    'L_Index1_x', 'L_Index1_y', 'L_Index1_z', 
    'L_Middle1_x', 'L_Middle1_y', 'L_Middle1_z', 
    'L_Pinky1_x', 'L_Pinky1_y', 'L_Pinky1_z', 
    'L_Ring1_x', 'L_Ring1_y', 'L_Ring1_z', 
    'L_Thumb1_x', 'L_Thumb1_y', 'L_Thumb1_z', 
    'R_Index1_x', 'R_Index1_y', 'R_Index1_z', 
    'R_Middle1_x', 'R_Middle1_y', 'R_Middle1_z', 
    'R_Pinky1_x', 'R_Pinky1_y', 'R_Pinky1_z', 
    'R_Ring1_x', 'R_Ring1_y', 'R_Ring1_z', 
    'R_Thumb1_x', 'R_Thumb1_y', 'R_Thumb1_z', 
    'L_Index2_x', 'L_Index2_y', 'L_Index2_z', 
    'L_Middle2_x', 'L_Middle2_y', 'L_Middle2_z', 
    'L_Pinky2_x', 'L_Pinky2_y', 'L_Pinky2_z', 
    'L_Ring2_x', 'L_Ring2_y', 'L_Ring2_z', 
    'L_Thumb2_x', 'L_Thumb2_y', 'L_Thumb2_z', 
    'R_Index2_x', 'R_Index2_y', 'R_Index2_z', 
    'R_Middle2_x', 'R_Middle2_y', 'R_Middle2_z', 
    'R_Pinky2_x', 'R_Pinky2_y', 'R_Pinky2_z', 
    'R_Ring2_x', 'R_Ring2_y', 'R_Ring2_z', 
    'R_Thumb2_x', 'R_Thumb2_y', 'R_Thumb2_z', 
    'L_Index3_x', 'L_Index3_y', 'L_Index3_z', 
    'L_Middle3_x', 'L_Middle3_y', 'L_Middle3_z', 
    'L_Pinky3_x', 'L_Pinky3_y', 'L_Pinky3_z', 
    'L_Ring3_x', 'L_Ring3_y', 'L_Ring3_z', 
    'L_Thumb3_x', 'L_Thumb3_y', 'L_Thumb3_z', 
    'R_Index3_x', 'R_Index3_y', 'R_Index3_z', 
    'R_Middle3_x', 'R_Middle3_y', 'R_Middle3_z', 
    'R_Pinky3_x', 'R_Pinky3_y', 'R_Pinky3_z', 
    'R_Ring3_x', 'R_Ring3_y', 'R_Ring3_z', 
    'R_Thumb3_x', 'R_Thumb3_y', 'R_Thumb3_z'
]