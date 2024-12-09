import torch
import abc
import third.general as utils
from sdf_tracker.volumetric_grad_sdf import VolumetricGradSdf
import numpy as np
import math
from scipy.spatial import cKDTree
from datasets.voxel_data import _get_voxel_attributes, _calc_curvature_bins, _lowMedHighCurvSegmentation, _sample_off_surf_interp, dot

class VoxelSampler(metaclass=abc.ABCMeta):
    def __init__(self,
                 conf, 
                 object_path:str,
                 batch_size: int,
                 device) -> None:
        super().__init__()
        
        self.use_curvature = False if conf['uniform_sampling'] else True
        self.proportions = conf['proportions']
        self.curvature_percentiles = conf['percentiles']
        # self.extrapolate = conf['extrapolate']
        self.device = device
        self.local_sigma = conf.get_float('local_sigma', 1.0)
        
        print(f"Loading object \"{object_path}\".")
        print("Using curvatures? ", "YES" if self.use_curvature else "NO")
        self.tSDF = utils.load_object(object_path)
        
        
        #normalize weight
        ratio = self.tSDF.w / torch.max(self.tSDF.w)
        self.tSDF.w = ratio**0.1
        
        
        self.batch_size = batch_size
        print(f"Fetching {self.batch_size} training points per iteration.")
        
        
        self.bounding_box, _ = self.tSDF.get_boundary()
        print("point cloud size: ", self.bounding_box)
        
        self.on_surf_attributes, self.off_surface_attributes, self.empty_surf_attributes = _get_voxel_attributes(self.tSDF, self.bounding_box)

        self.mesh_size = self.on_surf_attributes['surf_points'].shape[0]
        
        if self.use_curvature:
            self.curvature_bins = _calc_curvature_bins(
                self.on_surf_attributes['surf_mean'].cpu(),
                self.curvature_percentiles
            )
        
            print(self.curvature_bins)
            
            curvature_pts_list = _lowMedHighCurvSegmentation(
            on_surf_attributes=self.on_surf_attributes,
            off_surf_attributes=self.off_surface_attributes,
            bin_edges=self.curvature_bins
            )
            
            self.low_curvature_pts = curvature_pts_list['low_curvature_pts']
            self.med_curvature_pts = curvature_pts_list['med_curvature_pts']
            self.high_curvature_pts = curvature_pts_list['high_curvature_pts']
            self.low_curvature_voxels = curvature_pts_list['low_curvature_voxels']
            self.med_curvature_voxels = curvature_pts_list['med_curvature_voxels']
            self.high_curvature_voxels = curvature_pts_list['high_curvature_voxels']
            
        self.points = self.on_surf_attributes['surf_points']
        self.normals = self.on_surf_attributes['surf_normals']
        self.weights = self.off_surface_attributes['weight']
        self.voxels = self.off_surface_attributes['voxel']
        self.sdf = self.off_surface_attributes['sdf']
        print("Done preparing the dataset.")
        
    
    @abc.abstractmethod
    def get_points(self):
        pass
    
    @abc.abstractmethod
    def get_nonmnfld_points(self):
        pass
    
    @staticmethod
    def get_sampler(sampler_type):
        return utils.get_class(sampler_type)
    
    
    def get_voxel_points(self, n_samples):
        
        indices = torch.tensor(np.random.choice(self.points.shape[0], n_samples, True))
            
        points = self.points[indices]
        normals = self.normals[indices]
        weights =self.weights[indices]
        
        voxels = self.voxels[indices]
        sdf = self.sdf[indices]
        
        voxel_weights = (self.tSDF.voxel_size - sdf.abs()) / self.tSDF.voxel_size * weights
        return points, voxels, sdf, normals, weights, voxel_weights
    
    def get_curvature_voxel_points(self, n_samples):
        n_low_curvature = int(math.floor(self.proportions[0] * n_samples))
        low_curvature_idx = np.random.choice(
            range(self.low_curvature_pts.shape[0]),
            size=n_low_curvature,
            replace=True if n_low_curvature > self.low_curvature_pts.shape[0] else False
        )
        on_surface_sampled = len(low_curvature_idx)
        n_med_curvature = int(math.ceil(self.proportions[1] * n_samples))
        med_curvature_idx = np.random.choice(
            range(self.med_curvature_pts.shape[0]),
            size=n_med_curvature,
            replace=True if n_med_curvature > self.med_curvature_pts.shape[0] else False
        )
        on_surface_sampled += len(med_curvature_idx)
        n_high_curvature = n_samples - on_surface_sampled
        high_curvature_idx = np.random.choice(
            range(self.high_curvature_pts.shape[0]),
            size=n_high_curvature,
            replace=True if n_high_curvature > self.high_curvature_pts.shape[0] else False
        )
    
        sample = torch.cat((
        self.low_curvature_pts[low_curvature_idx],
        self.med_curvature_pts[med_curvature_idx],
        self.high_curvature_pts[high_curvature_idx]
        ), dim=0).float()
        
        sample_voxels = torch.cat((
            self.low_curvature_voxels[low_curvature_idx],
            self.med_curvature_voxels[med_curvature_idx],
            self.high_curvature_voxels[high_curvature_idx]
        ), dim=0)
        
        voxels = sample_voxels[:,:3]
        sdf = sample_voxels[:,-2]
        
        points = sample[:,:3]
        normals = sample[:,3:6]
        weights = sample[:,-2]
        
        voxel_weights = (self.tSDF.voxel_size - sdf.abs()).abs() / self.tSDF.voxel_size * weights

        
        return points, voxels, sdf, normals, weights, voxel_weights
    
    
    def get_extrapolate_points(self, n_samples):
        
        # local_sigma = math.sqrt(3)*self.tSDF.voxel_size
        local_sigma = self.local_sigma*self.tSDF.voxel_size
        
        samples = self.voxels + (torch.rand_like(self.voxels) * local_sigma - (local_sigma / 2))
        
        sample_sdf = self.sdf + dot(self.normals, samples-self.voxels)
        
        voxel_weights = (self.tSDF.voxel_size-sample_sdf.abs()).abs() / self.tSDF.voxel_size * self.weights
        
        indices = torch.tensor(np.random.choice(self.voxels.shape[0], n_samples, True))

        points = self.points[indices,:]
        voxels = samples[indices,:]
        sdf = sample_sdf[indices]
        weight = voxel_weights[indices]
        normals = self.normals[indices,:]
        voxel_weights = voxel_weights[indices]
        
        return points, voxels, sdf, weight, normals, voxel_weights
    
    
    def get_random_points(self, n_samples):
        min_bound, max_bound = self.bounding_box
        eps = self.tSDF.voxel_size
        x = torch.empty(n_samples).uniform_(min_bound[0]+eps, max_bound[0]-eps)
        y = torch.empty(n_samples).uniform_(min_bound[1]+eps, max_bound[1]-eps)
        z = torch.empty(n_samples).uniform_(min_bound[2]+eps, max_bound[2]-eps)
        
        # check if lays in non-zero weight voxels
        samples = torch.stack((x,y,z), dim=1) # n_points x 3
        # voxelization
        voxel_3d= self.tSDF.world_origin2voxel(samples)
        voxel = self.tSDF.idx2line(voxel_3d).long() # should always be with in the range
        weights = self.tSDF.w[voxel]
        
        samples = samples[weights==0].squeeze()
        
        n_points = samples.shape[0]

        return samples, -torch.ones(n_points), torch.zeros(n_points)
    
    
    
    def get_empty_points(self, n_samples):
        
        local_sigma = self.local_sigma*self.tSDF.voxel_size
        voxels = self.empty_surf_attributes['voxel']
        indices = torch.tensor(np.random.choice(voxels.shape[0], n_samples, 
                                                False)) 
        samples = voxels + (torch.rand_like(voxels) * local_sigma - (local_sigma /2))
        
        return samples[indices,:], -torch.ones(n_samples), torch.zeros(n_samples)


class IGRSampler(VoxelSampler):
    
    def __init__(self, conf, object_path: str, batch_size: int, device) -> None:
        super().__init__(conf, object_path, batch_size, device)
    
        
    def get_points(self):
        
        
        n_samples = self.batch_size # half from surf points, half from voxels
    
        if self.use_curvature:
            points, voxels, sdfs, normals, weights, voxel_weights = self.get_curvature_voxel_points(n_samples)
        else:
            points, voxels, sdfs, normals, weights, voxel_weights = self.get_voxel_points(n_samples)
        
        # sample random in space
        samples_rand, sdf_rand, weight_rand = self.get_nonmnfld_points()
        
        
        voxels_full = torch.cat((
            points, # zero sdf half batch size 
            samples_rand
        ))
        
        weights_full = torch.cat((
            weights,
            torch.zeros_like(sdf_rand)
        ))
        
        sdf_full = torch.cat((
            torch.zeros(points.shape[0]),
            sdf_rand
        ))
        
        normals_full = torch.cat((
            normals,
            torch.zeros_like(samples_rand)
        ))
        
        gt = {
            'points': points.to(self.device),
            'gt_w': weights_full.to(self.device),
            'gt_sdf': sdf_full.to(self.device),
            'gt_normals': normals_full.to(self.device)
        }
        
        return voxels_full.to(self.device), gt
        
    
    def get_nonmnfld_points(self):
        
        n_samples = self.batch_size // 8
        samples, sdf, weight = self.get_empty_points(n_samples)
        
        return samples, sdf, torch.zeros_like(sdf)


class NerualPullSampler(VoxelSampler):
    
    def __init__(self, conf, object_path: str, batch_size: int, device) -> None:
        super().__init__(conf, object_path, batch_size, device)

        
    def get_points(self):
        
        
        n_samples = int(self.batch_size * 0.1)# half from surf points, half from voxels
    
        if self.use_curvature:
            point, voxel, sdf, normal, weight, voxel_weight = self.get_curvature_voxel_points(n_samples)
        else:
            point, voxel, sdf, normal, weight, voxel_weight = self.get_voxel_points(n_samples)
            
        sample_points, sample_voxel, sample_sdf, sample_weight, sample_normal, sample_voxel_weight = self.get_extrapolate_points(int(self.batch_size * 0.9))
        
    
        samples = torch.cat((
            voxel,
            sample_voxel,
        ))
        
        points = torch.cat((
            point,
            sample_points,
            
        ))
        
        weights = torch.cat((
            voxel_weight,
            sample_voxel_weight,
            
        ))
            
        gt = {
            'points':points.to(self.device),
            'gt_w': weights.to(self.device)
        }
        
        return samples.to(self.device), gt
    
    def get_nonmnfld_points(self):
        return None
        
        
            
class ExtraSampler(VoxelSampler):
    
    def __init__(self, conf, object_path: str, batch_size: int, device) -> None:
        super().__init__(conf, object_path, batch_size, device)
        
        # self.extrapolate = conf.get_int('extrapolate', 0)

    def get_points(self):
        
        n_samples = self.batch_size // 2
        
        #half from voxels
        #half from sampled voxels
        if self.use_curvature:
            point, voxel, sdf, normal, weight, voxel_weight= self.get_curvature_voxel_points(int(n_samples * 0.2))
            
        else:
            point, voxel, sdf, normal, weight, voxel_weight = self.get_voxel_points(int(n_samples * 0.2))
        
        sample_points, sample_voxel, sample_sdf, sample_weight, sample_normal, smaple_voxel_weights = self.get_extrapolate_points(int(n_samples * 0.8))
        
        samples_rand, sdf_rand, weight_rand = self.get_nonmnfld_points()
        
        points_full = torch.cat((
            point, # n_sample x 3
            voxel,
            sample_points,
            sample_voxel,
            samples_rand
        ))
        
        normals_full = torch.cat((
            normal,
            torch.zeros_like(voxel),
            sample_normal,
            torch.zeros_like(sample_voxel),
            torch.zeros_like(samples_rand)
        ))
        
        
        sdf_full = torch.cat((
            torch.zeros(point.shape[0]),
            sdf,
            torch.zeros(sample_points.shape[0]),
            sample_sdf,
            sdf_rand
        ))
        
        weights_full = torch.cat((
            weight,
            voxel_weight,
            sample_weight,
            smaple_voxel_weights,
            torch.zeros_like(sdf_rand)
        ))
    
            # sample random in space
        
        gt = {
            'points': points_full.to(self.device),
            'gt_w': weights_full.to(self.device),
            'gt_sdf': sdf_full.to(self.device),
            'gt_normals': normals_full.to(self.device)
        }
        
        return points_full.to(self.device), gt
        
    
    def get_nonmnfld_points(self):
        
        # if self.extrapolate:
        n_samples = self.batch_size
        # else:
        #     n_samples = 0
        # samples, sdf, weight = self.get_random_points(n_samples)
        samples, sdf, weight = self.get_empty_points(n_samples)

        return samples, sdf, weight