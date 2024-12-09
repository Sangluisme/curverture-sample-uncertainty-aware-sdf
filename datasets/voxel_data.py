import math
import numpy as np
import plyfile
import torch
from torch.utils.data import Dataset
# # import trimeshs
import sys
from sdf_tracker.volumetric_grad_sdf import VolumetricGradSdf
import torch.nn.functional as F
from third.general import load_object

def _calc_curvature_bins(curvatures: torch.Tensor, percentiles: list) -> list:
    """Bins the curvature values according to `percentiles`.

    Parameters
    ----------
    curvatures: torch.Tensor
        Tensor with the curvature values for the vertices.

    percentiles: list
        List with the percentiles. Note that if any values larger than 1 in
        percentiles is divided by 100, since torch.quantile accepts only
        values in range [0, 1].

    Returns
    -------
    quantiles: list
        A list with len(percentiles) + 2 elements composed by the minimum, the
        len(percentiles) values and the maximum values for curvature.

    See Also
    --------
    torch.quantile
    """
    try:
        q = torch.quantile(curvatures, torch.Tensor(percentiles))
    except RuntimeError:
        percs = [None] * len(percentiles)
        for i, p in enumerate(percentiles):
            if p > 1.0:
                percs[i] = p / 100.0
                continue
            percs[i] = percentiles[i]
        q = torch.quantile(curvatures, torch.Tensor(percs).float())

    bins = [curvatures.min().item(), curvatures.max().item()]
    # Hack to insert elements of a list inside another list.
    bins[1:1] = q.data.tolist()
    return bins


def dot(tensor1, tensor2, dim=-1, keepdim=False) -> torch.Tensor:
    return (tensor1 * tensor2).sum(dim=dim, keepdim=keepdim)
    

def _get_voxel_attributes(tSDF: VolumetricGradSdf,
                          domain_bounds: tuple = ([-1, -1, -1], [1, 1, 1])):
    """get on surface points atrributes and off surface attributes
    
    described on equation (3)

    Args:
        tSDF (VolumetricGradSdf): voxel object

    Returns:
        on_surf_attributes: includes points location, normal, weight and curvatures
        off_surf_attributes: only consider the voxel contains at least one surface points
                            includes voxel center, sdf, weights
                            
    """
    voxel = tSDF.voxel2world_origin(tSDF.grid_points)
    normal = F.normalize(tSDF.grad, p=2, dim=1)
    points = voxel - tSDF.dist.view(tSDF.num_voxels, 1) * normal
    colors = tSDF.color
    k1 = tSDF.gaussian_curv
    k2 = tSDF.mean_curv
    
    valid = (tSDF.w > 0.0) & (torch.abs(tSDF.dist) < math.sqrt(3)*tSDF.voxel_size)
    count = torch.sum(valid)
    
    print('number of on surface points:', count)
    
    normal = normal[valid.nonzero(),:].squeeze()
    points = points[valid.nonzero(),:].squeeze()
    colors = colors[valid.nonzero(),:].squeeze()
    k1 = k1[valid.nonzero()].squeeze()
    k2 = k2[valid.nonzero()].squeeze()
    weight = tSDF.w[valid.nonzero()].squeeze()
    # weight = torch.ones_like(k1)
    
    on_surf_attributes = {
        'surf_points': points,
        'surf_normals': normal,
        'surf_colors': colors,
        'surf_gaussian': k1,
        'surf_mean': k2,
        'weight': weight
    }
    
    print('number of off surface points:', torch.sum(valid))
    print('mean weight is:', weight.mean())
    
    # valid = (tSDF.w > 0.0) & (torch.abs(tSDF.dist) < torch.sqrt(torch.Tensor([3.0])).to(tSDF.device)*tSDF.voxel_size)
    
    dist = tSDF.dist
    colors = tSDF.color
    
    off_surf_attributes = {
        'voxel': voxel[valid.nonzero(),:].squeeze(),
        'sdf': dist[valid.nonzero()].squeeze(),
        # 'weight': torch.ones_like(valid.nonzero().squeeze())*0.8,
        'gradient': tSDF.grad[valid.nonzero(),:].squeeze(),
        'weight': tSDF.w[valid.nonzero().squeeze()],
        'colors': colors[valid.nonzero()].squeeze()
    }
    
    invalid = (tSDF.w <= 0.0) | (torch.abs(tSDF.dist) >  math.sqrt(3)*tSDF.voxel_size)
    valid_x = (voxel[:,0] > domain_bounds[0][0]) & (voxel[:,0] < domain_bounds[1][0])
    valid_y = (voxel[:,1] > domain_bounds[0][1]) & (voxel[:,1] < domain_bounds[1][1])
    valid_z = (voxel[:,2] > domain_bounds[0][2]) & (voxel[:,2] < domain_bounds[1][2])

    invalid = invalid & valid_x & valid_y & valid_z
    
    empty_surf_attributes = {
        'voxel': voxel[invalid.nonzero(),:].squeeze(),
        'sdf': dist[invalid.nonzero()].squeeze(),
        # 'weight': torch.ones_like(valid.nonzero().squeeze())*0.8,
        'gradient': tSDF.grad[invalid.nonzero(),:].squeeze(),
        'weight': tSDF.w[invalid.nonzero().squeeze()],
        'colors': colors[invalid.nonzero()].squeeze()
        
    }
    
    return on_surf_attributes, off_surf_attributes, empty_surf_attributes


def _get_close_surf_pt(tSDF:VolumetricGradSdf,
                       off_surf_attibutes: list,
                       on_surf_attributes :list,
                       n_points: int):
    
    """sample along normal directions
    
        random walk up to 2 voxel size along normal directions and use equation (2) to compute the GT sdf
    
    Returns:
        list: new points off surface together with new sdf, normal, weight
    """
    
    disturb = np.random.uniform(-4*tSDF.voxel_size, 4*tSDF.voxel_size, size=(n_points,1))
    
    voxel_points = off_surf_attibutes['voxel']
    dist = off_surf_attibutes['sdf']
    normal = on_surf_attributes['surf_normals']
    weight = off_surf_attibutes['weight']
    
    valid = np.random.randint(0, high=dist.shape[0], size=n_points)
   
    
    new_points = voxel_points[valid,...] + normal[valid,...] * disturb
    new_d = dist[valid, None] + disturb
    
    
    return {
        'off_surf_pt': new_points,
        'off_surf_dist': new_d.squeeze(),
        'off_surf_normal': torch.zeros_like(new_points),
        'off_surf_weight': weight[valid,...],
        'off_surf_color': torch.zeros_like(new_points),
    }
        


def _sample_off_surf(tSDF: VolumetricGradSdf,
                     off_surf_attributes: list,
                     n_points: int):
    
    ''' 
    sample off surface by interatively choosing from off-surf points
    naiv process, no any interpolation
    
    '''
    voxel = off_surf_attributes['voxel']
    index = np.random.randint(0, high=voxel.shape[0], size=n_points)
    
    off_surf_pt = voxel[index, ...]
    off_surf_dist = off_surf_attributes['sdf'][index]
    weight = off_surf_attributes['weight'][index]
    normal = torch.zeros_like(off_surf_pt).to(off_surf_pt.device)
    color = off_surf_attributes['colors'][index,:]
    
    return {
        'off_surf_pt': off_surf_pt,
        'off_surf_dist': off_surf_dist,
        'off_surf_normal': normal,
        'off_surf_weight': weight,
        'off_surf_color': color,
    }
    

def _sample_off_surf_interp(tSDF:VolumetricGradSdf,
                        n_points: int,
                        min_bound: torch.Tensor,
                        max_bound: torch.Tensor):
    
    """sample points randomly in related domain and get GT sdf by Taylor expansion 

    """
    
    x = torch.empty(n_points).uniform_(min_bound[0], max_bound[0])
    y = torch.empty(n_points).uniform_(min_bound[1], max_bound[1])
    z = torch.empty(n_points).uniform_(min_bound[2], max_bound[2])


    samples = torch.stack((x,y,z), dim=1) # n_points x 3
    samples_dist = -torch.ones(n_points)
    samples_w = torch.zeros(n_points)
    samples_normal = torch.zeros_like(samples)
    samples_color = torch.zeros_like(samples)
    
    # voxelization
    voxel_3d= tSDF.world_origin2voxel(samples)
    voxel = tSDF.idx2line(voxel_3d).long()
        
    #check if voxel contain surf pt
    valid_voxel = (voxel < tSDF.num_voxels) & (voxel >=0)
    valid_index = voxel[valid_voxel.nonzero().squeeze()]
    
    valid_voxel_index = valid_voxel.nonzero().squeeze()
    if valid_voxel_index.shape != torch.Size([]):
        samples_dist[valid_voxel_index] = tSDF.dist[valid_index] + dot(F.normalize(tSDF.grad[valid_index,:], p=2, dim=1), samples[valid_voxel_index,:] - tSDF.voxel2world_origin(voxel_3d[valid_voxel_index,:]))
        samples_w[valid_voxel_index] = tSDF.w[valid_index]
        samples_color[valid_voxel_index, :] = tSDF.color[valid_index, :]
                
        # to prevent the sampled distance is too far from the current voxel
        invalid = (torch.abs(samples_dist) > torch.sqrt(torch.Tensor([3.0])).to(tSDF.device)*tSDF.voxel_size)
        
        samples_w[invalid] = 0
    
    return {
        'off_surf_pt': samples,
        'off_surf_dist': samples_dist,
        'off_surf_normal': samples_normal,
        'off_surf_weight': samples_w,
        'off_surf_color': samples_color,
    }
    
    
def _sample_on_surface(on_surf_attributes: list,
                       n_points: int,
                       exceptions: list = []):
    """Samples points from a voxel cloud surface.

    Slightly modified from `i3d.dataset._sample_on_surface`. Returns the
    indices of points on surface as well and excludes points with indices in
    `exceptions`.

    Parameters
    ----------
    list: on_surf_attributes

    n_points: int
        The number of vertices to sample.

    exceptions: list, optional
        The list of vertices to exclude from the selection. The default value
        is an empty list, meaning that any vertex might be selected. This works
        by setting the probabilities of any vertices with indices in
        `exceptions` to 0 and adjusting the probabilities of the remaining
        points.

    Returns
    -------
    samples: torch.Tensor
        The samples drawn from `on surface attributes`

    idx: list
        The index of `samples` in `on surface points`. Might be fed as input to further
        calls of `sample_on_surface`

    See Also
    --------
    numpy.random.choice
    
    """
    
    points = on_surf_attributes['surf_points']
    normals = on_surf_attributes['surf_normals']
    colors = on_surf_attributes['surf_colors']
    k1 = on_surf_attributes['surf_gaussian']
    k2 = on_surf_attributes['surf_mean']
    w = on_surf_attributes['weight']
    
    if exceptions:
        p = np.array(
            [1. / points.shape[0] - len(exceptions)] *
            points.shape[0]
        )
        p[exceptions] = 0.0
    
    idx = np.random.choice(
        np.arange(start=0, stop=points.shape[0]),
        size=n_points,
        replace=False,
        p=p if exceptions else None
    )
    
    on_points = points[idx,:]
    on_normals = normals[idx,:]
    on_colors = colors[idx, :]
    gaussian = k1[idx, None]
    mean = k2[idx, None]
    w = w[idx, None]

    return torch.cat((
        on_points, #points :3
        on_normals, #normals 3:6
        on_colors, # 6:9
        w,
        # gaussian, 
        mean
    ), dim=1).type(torch.float32), idx.tolist()
    
    

def _sample_nonmnfld_points(
    on_surf_attributes: list,
    n_samples: int,
    domain_bound: tuple = ([-1, -1, -1], [1, 1, 1]),
    local_sigma: float=0.01
    ):
    
    size = - max(domain_bound[0]) + min(domain_bound[1])
    global_sigma = 1.8 * size / 2
    local_sigma = size * 0.01
    # print('global sigma is:', global_sigma)
    
    points = on_surf_attributes['surf_points']

    sample_size, dim = points.shape
    sample_local = points + (torch.randn_like(points) * local_sigma)
    
    sample_global = (torch.rand(sample_size // 8, dim, device=points.device) * (global_sigma * 2)) - global_sigma

    sample = torch.cat([sample_local, sample_global], dim=0)
    
    idx = np.random.choice(
        np.arange(start=0, stop=points.shape[0]),
        size=n_samples,
        replace=False
    )
    
    return sample[idx,:]

def _sample_mnfold_points_interp(
    tSDF:VolumetricGradSdf,
    on_surf_attributes: list,
    n_points: int,
    domain_bound: tuple = ([-1, -1, -1], [1, 1, 1]),
    ):
    
    size = - min(domain_bound[0]) + max(domain_bound[1])
    global_sigma = 1.8 * size / 2
    local_sigma = tSDF.voxel_size
    # print('global sigma is:', global_sigma)
    
    voxels = on_surf_attributes['surf_points']
    # sdf = on_surf_attributes['sdf']
    sdf = torch.zeros(voxels.shape[0])
    gradient = on_surf_attributes['surf_normals']
    w = on_surf_attributes['weight']
    
    normal = F.normalize(gradient, p=2, dim=1)
    
    # sample moving directions and distance
    samples = voxels + (torch.randn_like(voxels) * local_sigma)
    sample_d = sdf + dot(normal, samples - voxels)
    
    idx = np.random.choice(
        np.arange(start=0, stop=samples.shape[0]),
        size=n_points,
        replace=False if n_points <= samples.shape[0] else True
    )
    
    return {
        'off_surf_pt': samples[idx,:],
        'off_surf_dist': sample_d[idx],
        'off_surf_normal': torch.zeros(n_points,3),
        'off_surf_weight': w[idx],
        'off_surf_color': torch.zeros(n_points,3),
    }
    


def _sample_nonmnfold_points_interp(
    tSDF:VolumetricGradSdf,
    on_surf_attributes: list,
    empty_surf_attributes: list,
    n_points: int,
    domain_bound: tuple = ([-1, -1, -1], [1, 1, 1]),
):
    local_sigma = tSDF.voxel_size
    
    # sample_global = (torch.rand(sample_size // 4, dim, device=voxels.device) * (global_sigma * 2)) - global_sigma
    voxels = on_surf_attributes['surf_points']
    # sdf = on_surf_attributes['sdf']
    sdf = torch.zeros(voxels.shape[0])
    gradient = on_surf_attributes['surf_normals']
    w = on_surf_attributes['weight']
    
    normal = F.normalize(gradient, p=2, dim=1)
    
    # sample moving directions and distance
    samples_local = voxels + (torch.randn_like(voxels) * local_sigma / 2)
    sample_d = sdf + dot(normal, samples_local - voxels)
    
    half_n_points = n_points // 2
    idx = np.random.choice(
        np.arange(start=0, stop=samples_local.shape[0]),
        size=half_n_points,
        replace=False if half_n_points <= samples_local.shape[0] else True
    )
    
    samples_local = samples_local[idx,:]
    sample_d = sample_d[idx]
    w = w[idx]
    
    '''
    # sample on space
    n_sample = voxels.shape[0] // 4
    x = torch.empty(n_sample).uniform_(domain_bound[0][0], domain_bound[1][0])
    y = torch.empty(n_sample).uniform_(domain_bound[0][1], domain_bound[1][1])
    z = torch.empty(n_sample).uniform_(domain_bound[0][2], domain_bound[1][2])
    sample_global = torch.stack((x,y,z), dim=1) # n_points x 3
  
    
    # determine the weights and distance of the global samples
    voxel_3d = tSDF.world_origin2voxel(sample_global)
    voxel = tSDF.idx2line(voxel_3d).long()
    #check if inside coarse voxel
    valid_voxel = (voxel < tSDF.num_voxels) & (voxel >=0)
    voxel_index = voxel[valid_voxel.nonzero().squeeze()]
    voxel_3d = voxel_3d[valid_voxel.nonzero().squeeze()]

    sample_weight = tSDF.w[voxel_index]
    zero_weight_global_points = (sample_weight <= 0.0).nonzero().squeeze()
    
    
    sample_global = sample_global[zero_weight_global_points,:].squeeze()
    '''
    
    empty_voxel = empty_surf_attributes['voxel']
    sample_global = empty_voxel + (torch.randn_like(empty_voxel) * local_sigma / 2)
    
    
    idx = np.random.choice(
        np.arange(start=0, stop=sample_global.shape[0]),
        size=half_n_points,
        replace=False if half_n_points <= sample_global.shape[0] else True
    )
    
    sample_global = sample_global[idx,:]
    
    sample_global_w = torch.zeros(sample_global.shape[0])
    sample_global_d = -torch.ones(sample_global.shape[0])
    # concatenate
    samples = torch.cat([samples_local, sample_global], dim=0)
    sample_d = torch.cat([sample_d, sample_global_d], dim=0)
    sample_weight = torch.cat([w, sample_global_w], dim=0)
    
    
    
    return {
        'off_surf_pt': samples,
        'off_surf_dist': sample_d,
        'off_surf_normal': torch.zeros(n_points,3),
        'off_surf_weight': sample_weight,
        'off_surf_color': torch.zeros(n_points,3),
    }
    

def _lowMedHighCurvSegmentation(
    on_surf_attributes: list,
    off_surf_attributes: list,
    bin_edges: np.array,
    exceptions: list = []
    ):
    """Samples `n_points` points from `on surface points` based on their curvature.

    This function is based on `i3d.dataset.lowMedHighCurvSegmentation`.

    Parameters
    ----------
    mesh: o3d.t.geometry.TriangleMesh,
        The mesh to sample points from.

    n_samples: int
        Number of samples to fetch.

    bin_edges: np.array
        The [minimum, low-medium threshold, medium-high threshold, maximum]
        curvature values in `mesh`. These values define thresholds between low
        and medium curvature values, and medium to high curvatures.

    proportions: np.array
        The percentage of points to fetch for each curvature band per batch of
        `n_samples`.

    Returns
    -------
    samples: torch.Tensor
        The vertices sampled from `gradient-SDF object`.
    """
    
    on_surface_sampled = 0
    points = on_surf_attributes['surf_points']
    normals = on_surf_attributes['surf_normals']
    colors = on_surf_attributes['surf_colors']
    # k1 = on_surf_attributes['surf_gaussian']
    k2 = on_surf_attributes['surf_mean']
    w = on_surf_attributes['weight']
    
    on_surface_pts = torch.column_stack((
        points, #:3
        normals, #4:7
        torch.zeros_like(w),
        w,
        k2
    ))
    
    voxels = off_surf_attributes['voxel']
    sdf = off_surf_attributes['sdf']
    
    off_surface_pts = torch.column_stack((
        voxels,
        torch.zeros_like(voxels),
        sdf,
        w
    ))
    
    if exceptions:
        index = torch.Tensor(
            list(set(range(on_surface_pts.shape[0])) - set(exceptions)),
        ).int()
        on_surface_pts = torch.index_select(
            on_surface_pts, dim=0, index=index
        )
        
    curvatures = on_surface_pts[..., -1]
    low_curvature_pts = on_surface_pts[(curvatures >= bin_edges[0]) & (curvatures < bin_edges[1]), ...]
    med_curvature_pts = on_surface_pts[(curvatures >= bin_edges[1]) & (curvatures < bin_edges[2]), ...]
    high_curvature_pts = on_surface_pts[(curvatures >= bin_edges[2]) & (curvatures <= bin_edges[3]), ...]
    
    low_curvature_voxels = off_surface_pts[(curvatures >= bin_edges[0]) & (curvatures < bin_edges[1]), ...]
    med_curvature_voxels = off_surface_pts[(curvatures >= bin_edges[1]) & (curvatures < bin_edges[2]), ...]
    high_curvature_voxels = off_surface_pts[(curvatures >= bin_edges[2]) & (curvatures <= bin_edges[3]), ...]
    
    
    print('low curvature pts {0}'.format(low_curvature_pts.shape[0]))
    print('med curvature pts {0}'.format(med_curvature_pts.shape[0]))
    print('high curvature pts {0}'.format(high_curvature_pts.shape[0]))


    return {
        'low_curvature_pts': low_curvature_pts, 
        'med_curvature_pts': med_curvature_pts, 
        'high_curvature_pts': high_curvature_pts,
        'low_curvature_voxels': low_curvature_voxels,
        'med_curvature_voxels': med_curvature_voxels,
        'high_curvature_voxels': high_curvature_voxels,
    }


def _curvature_guide_sample(curvature_pts_list: list,
                            n_samples: int,
                            proportions: np.array):
    
    low_curvature_pts = curvature_pts_list['low_curvature_pts']
    med_curvature_pts = curvature_pts_list['med_curvature_pts']
    high_curvature_pts = curvature_pts_list['high_curvature_pts']
    
    n_low_curvature = int(math.floor(proportions[0] * n_samples))
    low_curvature_idx = np.random.choice(
    range(low_curvature_pts.shape[0]),
    size=n_low_curvature,
    replace=True if n_low_curvature > low_curvature_pts.shape[0] else False
    )
    on_surface_sampled = len(low_curvature_idx)
    n_med_curvature = int(math.ceil(proportions[1] * n_samples))
    med_curvature_idx = np.random.choice(
        range(med_curvature_pts.shape[0]),
        size=n_med_curvature,
        replace=True if n_med_curvature > med_curvature_pts.shape[0] else False
    )
    on_surface_sampled += len(med_curvature_idx)
    n_high_curvature = n_samples - on_surface_sampled
    high_curvature_idx = np.random.choice(
    range(high_curvature_pts.shape[0]),
    size=n_high_curvature,
    replace=True if n_high_curvature > high_curvature_pts.shape[0] else False
    )
    
    return torch.cat((
        low_curvature_pts[low_curvature_idx, ...],
        med_curvature_pts[med_curvature_idx, ...],
        high_curvature_pts[high_curvature_idx, ...]
    ), dim=0)
    
    
    
def _create_training_data(
    tSDF: VolumetricGradSdf,
    on_surf_attributes: list,
    off_surf_attributes: list,
    empty_surf_attributes: list,
    n_on_surf: int,
    n_off_surf: int,
    on_surface_exceptions: list = [],
    domain_bounds: tuple = ([-1, -1, -1], [1, 1, 1]),
    scene = None,
    no_sdf: bool = False,
    use_curvature: bool = False,
    curvature_fractions: list = [],
    curvature_pts_list: list = [],
    extrapolate: bool = False
    ):
    """Creates a set of training data with coordinates, normals and SDF
    values.

    Parameters
    ----------
    tSDF: VolumetricGradSdf gradient-SDF object
    
    on_surf_attributes: list
        # on surface points attributes 

    off_surf_attributes: list
            # off surface points attributes 

    n_on_surf: int
        # of points to sample from the mesh.

    n_off_surf: int
        # of points to sample from the domain. Note that we sample points
        uniformely at random from the domain.

    on_surface_exceptions: list, optional
        Points that cannot be used for training, i.e. test set of points.

    domain_bounds: tuple[np.array, np.array]
        Bounds to use when sampling points from the domain.

    no_sdf: boolean, optional
        If using SIREN's original loss, we do not query SDF for domain
        points, instead we mark them with SDF = -1.

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points.

    curvature_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curvature_thresholds: list
        The curvature values to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    Returns
    -------
    full_pts: torch.Tensor
    full_normals: torch.Tensor
    full_sdf: torch.Tensor
    full_weight: torch.Tensor
    full_color: torch.Tensor

    See Also
    --------
    _sample_on_surface, _lowMedHighCurvSegmentation
    """
        
    if use_curvature:
        surf_pts = _curvature_guide_sample(
            curvature_pts_list,
            n_on_surf,
            curvature_fractions
        )
    else:
        surf_pts, _ = _sample_on_surface(
            on_surf_attributes,
            n_on_surf,
            on_surface_exceptions
        )
    
    if extrapolate:
        
        # taylor expansion sample       
        off_surf_sample_interp = _sample_nonmnfold_points_interp(tSDF=tSDF,
                                    on_surf_attributes= on_surf_attributes,
                                    empty_surf_attributes=empty_surf_attributes,
                                    n_points=n_off_surf//2,
                                    domain_bound=domain_bounds)
    
        # along normal sample
        # off_surf_sample = _sample_mnfold_points_interp(tSDF=tSDF,
        #                         on_surf_attributes=on_surf_attributes,
        #                         n_points=int(n_off_surf*0.7))
        
        # off surface sample
        off_surf_sample = _sample_off_surf(tSDF=tSDF,
                                        off_surf_attributes=off_surf_attributes,
                                        n_points=n_off_surf // 2 )
        
        # off_surf_sample_interp = _sample_mnfold_points_interp(tSDF=tSDF,
        #                     on_surf_attributes=on_surf_attributes,
        #                     n_points=n_off_surf // 2)
        
            
        
        full_pts = torch.row_stack((
            surf_pts[...,:3],
            off_surf_sample['off_surf_pt'],
            off_surf_sample_interp['off_surf_pt'],
        ))
        
        full_sdf = torch.cat((
            torch.zeros(len(surf_pts)).to(tSDF.device),
            off_surf_sample['off_surf_dist'],
            off_surf_sample_interp['off_surf_dist'],
        ))
        
        full_normals = torch.row_stack((
            surf_pts[...,3:6],
            off_surf_sample['off_surf_normal'],
            off_surf_sample_interp['off_surf_normal'],
        ))
        
        full_weight = torch.cat((
            surf_pts[...,-2],
            off_surf_sample['off_surf_weight'],
            off_surf_sample_interp['off_surf_weight'],
        ))
        
        full_color = torch.cat((
            surf_pts[...,6:9],
            off_surf_sample['off_surf_color'],
            off_surf_sample_interp['off_surf_color'],
        ))
        
       
    
    else:
      
        # surf_pts2, _ = _sample_on_surface(
        #     on_surf_attributes,
        #     n_on_surf,
        #     on_surface_exceptions
        # )
        nonmnfld_point = _sample_nonmnfld_points(
            on_surf_attributes,
            n_on_surf,
            domain_bounds
        )
        
        full_pts = torch.row_stack((
            surf_pts[...,:3],
            nonmnfld_point[...,:3],
        ))
        
        full_sdf = torch.cat((
            torch.zeros(len(surf_pts)).to(tSDF.device),
            -torch.ones(len(surf_pts)).to(tSDF.device),
        ))
        
        full_normals = torch.row_stack((
            surf_pts[...,3:6],
            -torch.ones_like(nonmnfld_point).to(tSDF.device),
        ))
        
        full_weight = torch.cat((
            surf_pts[...,-2],
            torch.zeros(len(surf_pts)).to(tSDF.device),
        ))
        
        full_color = torch.cat((
            surf_pts[...,6:9],
            torch.zeros_like(nonmnfld_point).to(tSDF.device),
        ))
        

    return full_pts.float(), full_normals.float(), full_sdf.float(),  full_weight.float(), full_color.float()
    

class TsdfCloud(Dataset):
    """SDF Point Cloud dataset.

    Parameters
    ----------
    object_path: str
        Path to the base gradient-SDF object.
        
    batch_size: integer, optional
        Used for fetching `batch_size` at every call of `__getitem__`. If set
        to 0 (default), fetches all on-surface points at every call.

    n_on_surface: int, optional
        Number of surface samples to fetch (i.e. {X | f(X) = 0}). Default value
        is None, meaning that all vertices will be used.

    off_surface_sdf: number, optional
        Value to replace the SDF calculated by the sampling function for points
        with SDF != 0. May be used to replicate the behavior of Sitzmann et al.
        If set to `None` (default) uses the SDF estimated by the sampling
        function.

    off_surface_normals: torch.Tensor, optional
        Value to replace the normals calculated by the sampling algorithm for
        points with SDF != 0. 

    use_curvature: boolean, optional
        Indicates if we must use the curvature to perform sampling on surface
        points. By default this is False.

    curvature_fractions: list, optional
        The fractions of points to sample per curvature band. Only used when
        `use_curvature` is True.

    curvature_percentiles: list, optional
        The curvature percentiles to use when defining low, medium and high
        curvatures. Only used when `use_curvature` is True.

    References
    ----------
    [1] Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B.,
    & Wetzstein, G. (2020). Implicit Neural Representations with Periodic
    Activation Functions. ArXiv. Retrieved from http://arxiv.org/abs/2006.09661
    """
    
    def __init__(self, object_path: str,
                 batch_size: int,
                 off_surface_sdf: float = None,
                 off_surface_normals: torch.Tensor = None,
                 use_curvature: bool = False,
                 curvature_fractions: list = [],
                 curvature_percentiles: list = [],
                 extrapolate: bool = False):
        super().__init__()
        
        self.off_surface_normals = None
        if off_surface_normals is not None:
            if isinstance(off_surface_normals, list):
                self.off_surface_normals = torch.Tensor(off_surface_normals)
        
        
        print(f"Loading object \"{object_path}\".")
        print("Using curvatures? ", "YES" if use_curvature else "NO")
        self.tSDF = load_object(object_path)
        
        #normalize weight
        self.tSDF.w = self.tSDF.w / torch.max(self.tSDF.w)
        
        self.batch_size = batch_size
        print(f"Fetching {self.batch_size // 2} on-surface points per iteration.")

        print("Creating point-cloud and acceleration structures.")
        self.off_surface_sdf = off_surface_sdf
        self.scene = None

        self.use_curvature = use_curvature
        self.curvature_fractions = curvature_fractions
        
        self.bounding_box, _ = self.tSDF.get_boundary()
        print(self.bounding_box)
        
        self.on_surf_attributes, self.off_surface_attributes, self.empty_surf_attributes = _get_voxel_attributes(self.tSDF, self.bounding_box)

        self.mesh_size = self.on_surf_attributes['surf_points'].shape[0]
        
        self.extrapolate = extrapolate
        
        # Binning the curvatures
        self.curvature_bins = None
        self.curvature_pts_list = None
        if use_curvature:
            self.curvature_bins = _calc_curvature_bins(
                self.on_surf_attributes['surf_mean'].cpu(),
                curvature_percentiles
            )
        
            print(self.curvature_bins)
            
            self.curvature_pts_list = _lowMedHighCurvSegmentation(
                self.on_surf_attributes,                                                          bin_edges=self.curvature_bins
                )
            
        
        print("Done preparing the dataset.")

    
    def __len__(self):
        # return  2 * self.mesh_size // self.batch_size
        return 1
    
    
    def __getitem__(self, index):
        pts, normals, sdf, w, color= _create_training_data(
            tSDF=self.tSDF,
            on_surf_attributes=self.on_surf_attributes,
            off_surf_attributes=self.off_surface_attributes,
            empty_surf_attributes=self.empty_surf_attributes,
            n_on_surf=self.batch_size // 2,
            n_off_surf=self.batch_size // 2,
            domain_bounds=(self.bounding_box[0], self.bounding_box[1]),
            scene=self.scene,
            no_sdf=self.off_surface_sdf is not None,
            use_curvature=self.use_curvature,
            curvature_fractions=self.curvature_fractions,
            curvature_pts_list=self.curvature_pts_list,
            extrapolate=self.extrapolate
        )
        
        # color = ( color - 0.5 ) * 2
        
        return {
            "coords": pts.float(),
        }, {
            'normals': normals.float(),
            'sdf':sdf.unsqueeze(1).float(),
            'weights': w.float(),
            'colors': color.float()
        }


