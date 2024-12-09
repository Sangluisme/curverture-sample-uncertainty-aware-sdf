import os
from glob import glob
import torch
import numpy as np
from skimage import measure
import trimesh
import pickle
import os.path as osp
import json
import shutil
import logging
from plyfile import PlyData


def create_output_paths(checkpoint_path, experiment_name, overwrite=True):
    """Helper function to create the output folders. Returns the resulting path.
    """
    full_path = osp.join(".", checkpoint_path, experiment_name)
    if osp.exists(full_path) and overwrite:
        shutil.rmtree(full_path)
    elif osp.exists(full_path):
        logging.warning("Output path exists. Not overwritting.")
        return full_path

    os.makedirs(osp.join(full_path, "models"))
    os.makedirs(osp.join(full_path, "reconstructions"))
    return full_path


def load_experiment_parameters(parameters_path):
    try:
        with open(parameters_path, "r") as fin:
            parameter_dict = json.load(fin)
    except FileNotFoundError:
        logging.warning("File '{parameters_path}' not found.")
        return {}
    return parameter_dict


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)



def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m


class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass

class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)), 5.0e-6)


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as inp:
        object = pickle.load(inp)
    
    return object


def load_timestamp_file(filename):
    """
    specially for format like tum rgbd data set

    Output
    list -- has the format [timestamp_rgb rgb_filename timestamp_depth depth_filename]
    """

    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n")
    list = [[str(v.strip()) for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]

    return list



def load_intrinsics(filename):
    file = open(filename)
    data = file.read()
    lines = [ [float(v.strip()) for v in item.split()] for item in data.split('\n')[:-1]]
    return np.asarray(lines)



def initial_grid(grid_dim, slice=None):
    if slice is None:      
        xx, yy, zz = np.mgrid[0:grid_dim[0], 0:grid_dim[1], 0:grid_dim[2]]
    else:
        grid_dim1 = [grid_dim[0] // slice,grid_dim[1] // slice,grid_dim[2] // slice]
        
        grid_dim2 = [grid_dim1[0] * 2, grid_dim1[1] * 2,grid_dim1[2] * 2]
        
        xx, yy, zz = np.mgrid[grid_dim1[0]:grid_dim2[0], grid_dim1[1]:grid_dim2[1], grid_dim1[2]:grid_dim2[2]]
    grid_points = torch.tensor(np.vstack([xx.flatten('F'), yy.flatten('F'), zz.flatten('F')]).T, dtype=torch.float32)
    return xx, yy, zz, grid_points
    


def gen_mc_coordinate_grid(N, bouding_box, device):
    size = (bouding_box[1] - bouding_box[0]).max()
    voxel_size = size / N
    
    _,_,_,grid_points = initial_grid([N,N,N])
    grid_points = - voxel_size * torch.Tensor([N,N,N]).float() / 2 + grid_points * voxel_size
    samples = torch.zeros(N**3, 4, device = device, requires_grad=False)
    samples[:,:3] = grid_points.to(device)
    
    return samples, voxel_size


def export_mesh(sdf, filename):

    verts, faces, normals, values = measure.marching_cubes(sdf.numpy(), level=0) #, method='lewiner', gradient_direction='ascent')



    if verts.shape[0] == 0:
        return False
    
    with open(filename, 'w') as f:

        f.write( "ply \n")
        f.write( "format ascii 1.0 \n")
        f.write( "element vertex %d \n" % verts.shape[0])
        f.write( "property float x \n")
        f.write( "property float y \n")
        f.write( "property float z \n")
        # f.write( "property uchar red")
        # f.write( "property uchar green")
        # f.write( "property uchar blue")
        f.write( "element face %d \n" % faces.shape[0])
        f.write( "property list uchar int vertex_indices \n")
        f.write( "end_header \n")

        # write vertices
        for i in range(verts.shape[0]):
            f.write( "%f %f %f \n" % (verts[i][0], verts[i][1], verts[i][2]))
        
        # for i in range(color.shape[0]):
        #     f.write( "%f %f %f \n" % (color[i][0], color[i][1], color[i][2]))

        # write faces
        for i in range(faces.shape[0]):
            f.write( "3 %d %d %d \n" % (faces[i][0], faces[i][1], faces[i][2]))
        

    f.close()
    return True



def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj



def _read_ply_with_curvatures(path: str):
    """Reads a PLY file with position, normal and curvature info.

    Note that we expect the input ply to contain x,y,z vertex data, as well
    as nx,ny,nz normal data and the curvature stored in the `quality` vertex
    property.

    Parameters
    ----------
    path: str, PathLike
        Path to the ply file. We except the file to be in binary format.

    Returns
    -------
    point cloud: differentiable point cloud object include curvature informations
        
    vertices: numpy.array
        The same vertex information as stored in `mesh` returned for
        convenience only.

    """
    # Reading the PLY file with curvature info
    with open(path, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=(num_verts, 11), dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["nx"]
        vertices[:, 4] = plydata["vertex"].data["ny"]
        vertices[:, 5] = plydata["vertex"].data["nz"]
        vertices[:, 6] = plydata["vertex"].data["red"]
        vertices[:, 7] = plydata["vertex"].data["green"]
        vertices[:, 8] = plydata["vertex"].data["blue"]
        vertices[:, 9] = plydata["vertex"].data["quality1"]
        vertices[:, 10] = plydata["vertex"].data["quality2"]

    points = torch.Tensor(vertices[:,:3]).float()
    normals = torch.Tensor(vertices[:,3:6]).float()
    colors = torch.Tensor(vertices[:,6:9]).float()
    k1 = torch.Tensor(vertices[:,9])
    k2 = torch.Tensor(vertices[:,10])    
    
    
    return torch.cat((points, normals, k1.unsqueeze(-1), k2.unsqueeze(-1)),dim= -1)


def normalize_ply(point_set):
    pnts = point_set[:,:3].numpy()
    center = np.mean(pnts, axis=0)
    pnts = pnts - np.expand_dims(center, axis=0)
    
    point_set[:,:3] = torch.Tensor(pnts)

    return point_set, center


def load_point_cloud_by_file_extension(file_name):

    ext = file_name.split('.')[-1]

    if ext == "npz" or ext == "npy":
        point_set = torch.tensor(np.load(file_name)).float()
    else:
        mesh = trimesh.load(file_name, ext)
        print(mesh.bounds)
        point_set = torch.tensor(mesh.vertices).float()
        if hasattr(mesh, 'vertex_normals'):
            normal = torch.tensor(mesh.vertex_normals).float()
            point_set = torch.cat((point_set, normal), dim=-1)
        else:
            print("load curvature point cloud")
            point_set = _read_ply_with_curvatures(file_name)

    point_set, center = normalize_ply(point_set)

    return point_set, center