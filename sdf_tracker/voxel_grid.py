from dataclasses import dataclass
import torch
import numpy as np
import third.general as utils

@dataclass
class Sdfvoxel:
    dist: float
    grad: torch.Tensor = torch.zeros(3)
    weight: float = 0.0
    r: float = 0.0
    g: float = 0.0
    b: float = 0.0
    
class VoxelGrid:
    def __init__(
        self,
        grid_dim,
        device,
        voxel_size=0.02,
        shift=torch.Tensor([[0.0, 0.0, 0.0]]).type(torch.float32)
        ):

        self.device = device
        self.grid_dim = grid_dim
        self.num_voxels = int(grid_dim[0]*grid_dim[1]*grid_dim[2])
        self.voxel_size = voxel_size
        self.shift = shift
        self.origin = shift - 0.5 * voxel_size * torch.Tensor(grid_dim).float()
        self.origin = self.origin.type(torch.float32).to(self.device)

        _, _, _, self.grid_points = utils.initial_grid(grid_dim)

    
    def voxel2world_origin(self, index):
        voxel_center = - self.voxel_size * torch.Tensor(self.grid_dim).float() / 2
        return (voxel_center + (self.voxel_size * index)).to(self.device)


    def voxel2world(self, index):
        return self.origin + (self.voxel_size * index).to(self.device)

    def world2voxelf(self, point):
        return (point - self.origin) / self.voxel_size

    def world2voxel(self, point):
        tmp = self.world2voxelf(point)
        return torch.round(tmp)
    
    def world_origin2voxel(self, point):
        voxel_center = - self.voxel_size * torch.Tensor(self.grid_dim).float() / 2
        return torch.round((point - voxel_center) / self.voxel_size)
    

    
    def idx2line(self, idx):
        if idx.shape[1] !=3:
            idx = torch.transpose(idx, 0, 1)
        return idx[:,0] + idx[:,1]*self.grid_dim[0] + idx[:,2]*self.grid_dim[0]*self.grid_dim[1]

    def line2idx(self, lin_idx):
        if type(lin_idx) is not torch.Tensor: 
            rest = torch.Tensor([lin_idx])
        else:
            rest = lin_idx
        k = torch.floor(rest/(self.grid_dim[0]*self.grid_dim[1]))
        rest -= k*self.grid_dim[0]*self.grid_dim[1]
        j = torch.floor(rest/self.grid_dim[0])
        rest -= j*self.grid_dim[0]
        i = rest
        return torch.vstack([i,j,k]).type(torch.int64)

    def nearest_index(self, point):

        float_indx = self.world2voxelf(point)
        i, j, k = float_indx[:,0], float_indx[:,1], float_indx[:,2]

        invalid = (i<=0) | (j<=0) | (k<=0) | (i>=self.grid_dim[0]-1) | (j>=self.grid_dim[1]-1) | (k>=self.grid_dim[2]-1)

        index = torch.round(i) + torch.round(j) * self.grid_dim[0] + torch.round(k) * self.grid_dim[0]*self.grid_dim[1]

        index = torch.where(invalid, torch.Tensor([-1]).to(self.device), index)

        return index.type(torch.int64)
