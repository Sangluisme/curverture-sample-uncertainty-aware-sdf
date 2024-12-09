
import torch
import open3d as o3d

class DiffPointCloud:
    def __init__(self, 
                 points:torch.Tensor, 
                 normals:torch.Tensor, 
                 colors:torch.Tensor, 
                 gaussian_curv:torch.Tensor=None, 
                 mean_curv:torch.Tensor=None):
    
        self.points = torch.from_numpy(points).type(torch.float32)
        self.normals = torch.from_numpy(normals).type(torch.float32)
        self.colors = torch.from_numpy(colors).type(torch.float32)
        self.gaussian_curv = torch.from_numpy(gaussian_curv).type(torch.float32)
        self.mean_curv = torch.from_numpy(mean_curv).type(torch.float32)
        self.n_points = points.shape[0]
        
        pt_cloud = o3d.geometry.PointCloud()
        pt_cloud.points = o3d.utility.Vector3dVector(self.points)
        
        center = torch.mean(self.points, dim=0)
        self.points = self.points - center
        
        self.min_bound = pt_cloud.get_min_bound()
        self.max_bound = pt_cloud.get_max_bound()