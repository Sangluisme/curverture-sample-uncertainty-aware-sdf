import numpy as np
import plyfile
from skimage.measure import marching_cubes, _marching_cubes_lewiner
import time
import torch
import weight_marching_cubes as mc
# import pybind11 as py
import third.general as util
import math
import trimesh
import plotly.graph_objs as go
import plotly.offline as offline
import matplotlib.pyplot as plt

def gen_mc_coordinate_grid(N: int, voxel_size: float, t: float = None,
                           device: str = "cpu",
                           voxel_origin: list = [-1, -1, -1]) -> torch.Tensor:
    """Creates the coordinate grid for inference and marching cubes run.

    Parameters
    ----------
    N: int
        Number of elements in each dimension. Total grid size will be N ** 3

    voxel_size: number
        Size of each voxel

    t: float, optional
        Reconstruction time. Required for space-time models. Default value is
        None, meaning that time is not a model parameter

    device: string, optional
        Device to store tensors. Default is CPU

    voxel_origin: list[number, number, number], optional
        Origin coordinates of the volume. Must be the (bottom, left, down)
        coordinates. Default is [-1, -1, -1]

    Returns
    -------
    samples: torch.Tensor
        A (N**3, 3) shaped tensor with samples' coordinates. If t is not None,
        then the return tensor is has 4 columns instead of 3, with the last
        column equalling `t`.
    """
    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

    sdf_coord = 3
    if t is not None:
        sdf_coord = 4

    # (x,y,z,sdf) if we are not considering time
    # (x,y,z,t,sdf) otherwise
    samples = torch.zeros(N ** 3, sdf_coord + 1, device=device,
                          requires_grad=False)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.long() / N) % N
    samples[:, 0] = ((overall_index.long() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2] 
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1] 
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0] 

    # adding the time
    if t is not None:
        samples[:, sdf_coord-1] = t

    # print(samples[:10,:])
    # print(samples[-10:,:])
    return samples

def gen_mc_coordinate_grid_local_uniform(N, bounding_box, device):
    eps = 0.1
    size = (bounding_box[1].max() - bounding_box[0].min()) + eps # enlarge for 10%
    # size = torch.round(bounding_box[1] - bounding_box[0])
    print(bounding_box)
    
    voxel_size = size / N
    center = (bounding_box[1] + bounding_box[0]) / 2
    _,_,_,grid_points = util.initial_grid([N,N,N])
    
    grid_points = - voxel_size * torch.Tensor([N,N,N]).float() / 2 + grid_points * voxel_size
    samples = torch.zeros(N**3, 4, device = device, requires_grad=False)
    samples[:,:3] = grid_points.to(device)
    
    
    return samples, voxel_size

def gen_mc_coordinate_grid_local(N, bounding_box, device, slice=None):
    
    eps = 0.1
    size = (bounding_box[1] - bounding_box[0]) + 2*eps # enlarge for 10%
    # size = torch.round(bounding_box[1] - bounding_box[0])
    # print(bounding_box)
    
    voxel_size = torch.min(size) / N
    
    new_size = torch.floor(size/voxel_size) + 1
    
    # center = (bounding_box[1] + bounding_box[0]) / 2
    # grid_dim = [new_size[0], new_size[1], new_size[2]]
    _,_,_,grid_points = util.initial_grid(new_size.tolist(), slice)
    
    if slice is not None:
        new_size = (new_size // slice * 2) - (new_size // slice)
    
    # grid_points = - voxel_size * torch.Tensor(new_size).float() / 2 + grid_points * voxel_size
    grid_points = bounding_box[0] + grid_points * voxel_size
    grid_points = grid_points.to(device)
    
    # num_samples = int(new_size[0])*int(new_size[1])*int(new_size[2])
    num_samples = grid_points.shape[0]
    # num_samples = grid_points.shape[0]
    samples = torch.zeros(num_samples, 4, device = device, requires_grad=False)
    samples[:,:3] = grid_points.to(device)
    
    # print(grid_points)
    return grid_points, voxel_size, [int(new_size[0]), int(new_size[1]), int(new_size[2])]

def create_mesh(
    decoder,
    filename="",
    threshold=0.3, # cut the value of weight to zeros
    t=-1, # time=-1 means we are only in the space
    N=256,
    max_batch=32 ** 3,
    bounding_box=(torch.Tensor([-1,-1,-1]), torch.Tensor([1,1,1])),
    level=0.0,
    offset=None,
    scale=None,
    device="cpu",
    silent=False
):
    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    # voxel_origin = [0, 0, 0]
    # voxel_size = 2 / (N-1)

    samples, voxel_size = gen_mc_coordinate_grid_local_uniform(N, bounding_box, device=device)
    # _, _, _, grid_points = util.initial_grid([N, N, N])
    voxel_origin = [( - N / 2) * voxel_size]*3
    
    
    sdf_coord = 3
    if (t != -1):
        sdf_coord = 4

    num_samples = N ** 3
    head = 0

    start = time.time()
    weight = torch.zeros_like(samples)
    while head < num_samples:
        # print(head)
        sample_subset = samples[None, head:min(head + max_batch, num_samples), 0:sdf_coord]
        decoder_output = decoder(sample_subset)["model_out"].squeeze().detach().cpu()
        if len(decoder_output.shape) > 1:
            samples[head:min(head + max_batch, num_samples), sdf_coord] = (decoder_output[...,0])
            weight[head:min(head + max_batch, num_samples), sdf_coord] = (decoder_output[...,1])
        else:
            samples[head:min(head + max_batch, num_samples), sdf_coord] = (decoder_output)
        head += max_batch

    sdf_values = samples[:, sdf_coord]
    
    sdf_values = sdf_values.reshape(N, N, N).permute(2,1,0)

    weight_value = weight[:,sdf_coord]
    
    
    end = time.time()
    if not silent:
        print(f"Sampling took: {end-start} s")
    
    
    voxel_origin = torch.Tensor(voxel_origin).to(device)

    start = time.time()
    verts, faces, normals, values = convert_sdf_samples_to_ply(
    sdf_values.data.cpu(),
    voxel_origin.cpu(),
    voxel_size,
    level,
    offset,
    scale,
        )
    
    save_ply(verts, faces, filename)
    # save_obj(verts, faces, None, filename)
        
    end = time.time()
    if not silent:
        print(f"write mesh ply: {end-start} s")

        if filename:
            print(f"Saving mesh to {filename}")
            print("Done")

    return verts, faces, normals, values
    
 
def create_weighted_mesh(
    decoder,
    filename="",
    threshold=0.3, # cut the value of weight to zeros
    t=-1, # time=-1 means we are only in the space
    N=256,
    max_batch=32 ** 3,
    bounding_box=(torch.Tensor([-1,-1,-1]), torch.Tensor([1,1,1])),
    level=0.0,
    offset=None,
    scale=None,
    device="cpu",
    silent=False):

    decoder.eval()
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not
    # the middle
    # voxel_size = 2 / (N-1)
    
    # if offset is None:
    #     offset = torch.zeros(3).to(device)
    
    # _, _, _, grid_points = util.initial_grid([N, N, N])
    
    # samples = (grid_points * voxel_size).to(device) + offset
    # samples = torch.cat((samples, torch.zeros(N**3,1).to(device)), dim=-1)
    
    grid_points, voxel_size, grid_dim = gen_mc_coordinate_grid_local(N, bounding_box, device=device)
    # voxel_origin = torch.FloatTensor(grid_dim) *0.5 * voxel_size
    voxel_origin = -grid_points[0,:]

    start = time.time()
  
    z = []
    for i,pnts in enumerate(torch.split(grid_points,10000,dim=0)):
        if (silent):
            print ('{0}'.format(i/(grid_points.shape[0] // 10000) * 100))

        # z.append(decoder(pnts[None])["model_out"].squeeze().detach().cpu().numpy())
        z.append(decoder(pnts).detach().cpu().numpy())
    z = np.concatenate(z,axis=0)
            
    sdf_values = torch.from_numpy(z[:,:1])
    sdf_values = sdf_values.reshape(grid_dim)
    
    weight_value = torch.from_numpy(z[:,-1:])
    
    print('mean of weight:', weight_value[weight_value.nonzero(as_tuple=True)].mean())
    
    # weight_value[invalid.nonzero()] = 0
    weight_value = torch.where(weight_value < threshold, torch.zeros_like(weight_value).to(weight_value.device), weight_value)
   
    # [0.6350 0.0780 0.1840], [0, 0.4470, 0.7410]
    n_weight = weight_value  / torch.max(weight_value)
    heatmap = torch.Tensor([0.6350, 0.0780, 0.1840]) * n_weight
    
    red = heatmap[:,0]
    green = heatmap[:,1]
    blue = heatmap[:,2]    
    
    red = red.reshape(grid_dim)
    green = green.reshape(grid_dim)
    blue = blue.reshape(grid_dim)
    
    weight_value = weight_value.reshape(grid_dim)
    

    end = time.time()
    if not silent:
        print(f"Sampling took: {end-start} s")
    
    # voxel_dim = [N, N, N]
    voxel_origin = torch.Tensor(voxel_origin).to(device)
    # min_bound, max_bound = get_boundary(sdf_values, voxel_size, grid_points.to(device))
    # min_index = torch.round((min_bound - voxel_origin) / voxel_size)
    # max_index =torch.round((max_bound - voxel_origin) / voxel_size)

    start = time.time()
   
    vertices, faces, if_save = weighted_marching_cubes(resolution=grid_dim, 
                        size=voxel_size, 
                        origin=voxel_origin, 
                        sdf=sdf_values.data.cpu(),
                        weight=weight_value.cpu(),
                        color=(red, green, blue),
                        iso_level=level,
                        min_bound=None,
                        max_bound=None,
                        filename=filename
                        )
    
    end = time.time()

    if not silent:
        print(f"write mesh ply: {end-start} s")

        if filename:
            print(f"Saving mesh to {filename}")
            print("Done")


    return vertices, faces, if_save


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    level=0.0,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """
    if isinstance(pytorch_3d_sdf_tensor, torch.Tensor):
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    else:
        numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    # Check if the cubes contains the zero-level set
   
    if level < numpy_3d_sdf_tensor.min() or level > numpy_3d_sdf_tensor.max():
        print(f"Surface level must be within volume data range.")
    else:
        verts, faces, normals, values = marching_cubes(
            numpy_3d_sdf_tensor, level, spacing=[voxel_size] * 3
        )

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    return mesh_points, faces, normals, values

def save_ply(
        verts: np.array,
        faces: np.array,
        filename: str,
        vertex_attributes: list = None
    ) -> None:
    """Converts the vertices and faces into a PLY format, saving the resulting
    file.

    Parameters
    ----------
    verts: np.array
        An NxD matrix with the vertices and its attributes (normals,
        curvatures, etc.). Note that we expect verts to have at least 3
        columns, each corresponding to a vertex coordinate.

    faces: np.array
        An Fx3 matrix with the vertex indices for each triangle.

    filename: str
        Path to the output PLY file.

    vertex_attributes: list of tuples
        A list with the dtypes of vertex attributes other than coordinates.

    Examples
    --------
    > # This creates a simple triangle and saves it to a file called
    > #"triagle.ply"
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > save_ply(verts, faces, "triangle.ply")

    > # Writting normal information as well
    > verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    > faces = np.array([[0, 1, 2]])
    > normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])
    > attrs = [("nx", "f4"), ("ny", "f4"), ("nz", "f4")]
    > save_ply(verts, faces, "triangle_normals.ply", vertex_attributes=attrs)
    """
    # try writing to the ply file
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    dtypes = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if vertex_attributes is not None:
        dtypes[3:3] = vertex_attributes

    verts_tuple = np.zeros(
        (num_verts,),
        dtype=dtypes
    )

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(verts[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(
        faces_building,
        dtype=[("vertex_indices", "i4", (3,))]
    )

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(filename)
    
    
def save_obj(verts, faces, normals=None, filename=""):
    if filename[-3:] == 'ply':
        filename = filename[:-3] + 'obj'
    
    if normals is not None:
        meshexport = trimesh.Trimesh(verts, faces, normals)
    else:
        meshexport = trimesh.Trimesh(verts, faces)
    meshexport.export(filename, file_type='obj')
    

def get_boundary(sdf, voxel_size, grid_points):
    valid = torch.abs(sdf).flatten() < math.sqrt(3.0)*voxel_size
    valid_voxels = grid_points[valid.nonzero(),:3].squeeze()

    min_index = torch.min(valid_voxels, dim=0)
    max_index = torch.max(valid_voxels, dim=0)

    return min_index[0], max_index[0]

def weighted_marching_cubes(resolution, size, origin, sdf, weight, color=None, iso_level=0.0, min_bound=None, max_bound=None, filename='weighted_mesh.ply'):
    assert len(resolution), f'grid size shoud be 3x1 numpy array'
    # num_voxel = resolution[0]*resolution[1]*resolution[2]
    # assert sdf.shape[0] == num_voxel, f'sdf should have same shape as grid resolution'
    assert sdf.shape == weight.shape, f'sdf shape should match weight shape'

    if (min_bound is None):
        min_bound = np.array([1, 1, 1])
    else:
        min_bound = min_bound.long().cpu()

    if (max_bound is None):
        max_bound = np.array([resolution[0], resolution[1], resolution[2]])
    else:
        max_bound = max_bound.long().cpu()

    grid_dim = np.asarray(max_bound-min_bound+1, dtype=np.int32)
    
    if isinstance(size, torch.Tensor):
        size = size.numpy()
    if isinstance(origin, torch.Tensor):
        origin = origin.cpu()
    grid_size = (size * grid_dim).astype(np.float32)
    origin = np.array(origin, dtype=np.float32)

    mesh_mc = mc.MarchingCubes(grid_dim, grid_size, origin)

   
    volumn = sdf.numpy()
    w = weight.numpy()

    if color is None:
        red = np.ones_like(volumn, dtype=np.float32)*255
        green = red
        blue = red
    else:
        red = color[0]*255
        green = color[1]*255
        blue = color[2]*255


    vertices, faces = mc.computeIsoSurface(mesh_mc, -volumn, w, red, green, blue, iso_level)
    if_save = mc.savePly(mesh_mc, filename)

    return vertices, faces, if_save


def get_surface_trace(decoder,
                      filename="",
                      resolution=128,
                      mc_value=0.0,
                      bounding_box=(torch.Tensor([-1,-1,-1]), torch.Tensor([1,1,1])),
                      is_uniform=False,
                      verbose=False,
                      connected=False):

    decoder.eval()
    
    trace = []
    
    if (is_uniform):
        grid = get_grid_uniform(resolution)
        grid_points = grid['grid_points']
    else:
        if not bounding_box is None:
            grid = get_grid(bounding_box,resolution)
        else:
            grid = get_grid(None, resolution)
        xx,yy,zz = grid['grid_points']
        grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

    z = []
    
    # print(grid_points)
    for i,pnts in enumerate(torch.split(grid_points,10000,dim=0)):
        if (verbose):
            print ('{0}'.format(i/(grid_points.shape[0] // 10000) * 100))

        # z.append(decoder(pnts[None])["model_out"].squeeze().detach().cpu().numpy())
        z.append(decoder(pnts).detach().cpu().numpy())
    z = np.concatenate(z,axis=0)[:,:1]
    
    if z.shape[1] > 1:
        z = z[:,0]

    verts=[]
    faces=[]
    meshexport= []
 
    if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

        z  = z.astype(np.float64)

        verts, faces, normals, values = marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
    
        # save_ply(verts, faces, filename)
        # save_obj(verts, faces, normals, filename)
        meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
        if connected:
            connected_comp = meshexport.split(only_watertight=False)
            max_area = 0
            max_comp = None
            for comp in connected_comp:
                if comp.area > max_area:
                    max_area = comp.area
                    max_comp = comp
            meshexport = max_comp
        
        meshexport.export(filename, 'ply')

        def tri_indices(simplices):
            return ([triplet[c] for triplet in simplices] for c in range(3))

        I, J, K = tri_indices(faces)

        trace.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=I, j=J, k=K, name='',
                          color='orange', opacity=0.5))

    if_save = True
    if len(verts) == 0:
        if_save = False
    
    

    return {"mesh_trace":trace,
            "mesh_export":meshexport}, if_save

    

def get_threed_scatter_trace(points,caption = None,colorscale = None,color = None):

    if (type(points) == list):
        trace = [go.Scatter3d(
            x=p[0][:, 0],
            y=p[0][:, 1],
            z=p[0][:, 2],
            mode='markers',
            name=p[1],
            marker=dict(
                size=3,
                line=dict(
                    width=2,
                ),
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption) for p in points]

    else:

        trace = [go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            name='projection',
            marker=dict(
                size=3,
                line=dict(
                    width=2,
                ),
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption)]

    return trace

def plot_cuts(points,decoder,path,epoch,near_zero,latent=None):
    onedim_cut = np.linspace(-1, 1, 200)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = points[:,-2].min(dim=0)[0].item()
    max_y = points[:,-2].max(dim=0)[0].item()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y - 0.1, max_y + 0.1, 10)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            input_=pnts
            if (not latent is None):
                input_ = torch.cat([latent.expand(pnts.shape[0],-1) ,pnts],dim=1)
            z.append(decoder(input_).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)[:,:1]

        if (near_zero):
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=-0.001,
                                     end=0.001,
                                     size=0.00001
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=True,
                                # contours=dict(
                                #      start=-0.001,
                                #      end=0.001,
                                #      size=0.00001
                                #      )
                                # ),colorbar = {'dtick': 0.05}
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='y = {0}'.format(pos[1, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}/cuts{1}_{2}.html'.format(path, epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        offline.plot(fig1, filename=filename, auto_open=False)

def get_grid(bounding_box,resolution):
    eps = 0.1
    input_min = bounding_box[0]
    input_max = bounding_box[1]
    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    # grid_points = torch.tensor(np.vstack([xx.T.ravel(), yy.T.ravel(), zz.T.ravel()]).T, dtype=torch.float).cuda()
    
    return {"grid_points":(xx,yy,zz),
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "shortest_axis_index":shortest_axis,
            }


def get_grid_uniform(resolution):
    x = np.linspace(-1.2,1.2, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = util.to_cuda(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float))

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.4,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}
    

        

def get_colormap(weights):
# Get the viridis colormap data
    viridis = plt.cm.get_cmap('viridis')

    # Create an array of 256 values evenly spaced between 0 and 1
    rgb_values = viridis(weights.squeeze())[:,:3]

    
    # Get the RGB values from the lookup table
    return rgb_values

    # Test the function

