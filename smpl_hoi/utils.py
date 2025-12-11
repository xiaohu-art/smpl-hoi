import torch
from isaaclab.utils.math import matrix_from_quat


@torch.jit.script
def obj_forward(obj_verts: torch.Tensor, obj_pos: torch.Tensor, obj_quat: torch.Tensor) -> torch.Tensor:
    '''
    obj_verts: (P, 3)
    obj_pos: (N, 3)
    obj_quat: (N, 4)
    return: (N, P, 3)
    '''
    obj_rotmat = matrix_from_quat(obj_quat)
    rot_verts = torch.matmul(obj_verts, obj_rotmat.transpose(1, 2))
    return rot_verts + obj_pos.unsqueeze(1)

@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat