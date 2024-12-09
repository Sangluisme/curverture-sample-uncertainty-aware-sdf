import os
from datetime import datetime
import time
from pyhocon import ConfigFactory
import argparse
import sys
sys.path.append('../code')
sys.path.append('./code')
import torch
import numpy as np

import sdf_tracker.rigid_point_optimizer as RigidOptimizer
import sdf_tracker.volumetric_grad_sdf as VolGradSdf
import third.image_loader as Loader
import third.normal_estimator as NormalEstimator
from third.curvature_estimator import DifferentialGeometryEstimator
import third.general as utils
from third.camera_utils import load_pose, Quaternion
from third.timer import Timer as Timer

_first = 0
_last = sys.maxsize

class Tracker():
    def __init__(self, device, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])

        # create folder to save results
        self.exps_folder_name = kwargs['results_dir']
        self.expname = kwargs['expname']
        self.device = device
        
        #mc_resolution
        self.mc_resolution = [kwargs['mc_resolution']] * 3
        self.voxel_size = kwargs['voxel_size']

        utils.mkdir_ifnotexists(self.exps_folder_name)
        self.expdir = os.path.join(self.exps_folder_name, self.expname+str(kwargs['mc_resolution']))
        utils.mkdir_ifnotexists(os.path.join(self.expdir))

        # parse dataset config
        dataset_conf = self.conf.get_config('dataset')
        if (dataset_conf['mode'] == 'tum') &  (dataset_conf.get('prefix', None) is not None):
            data_dir = os.path.join(dataset_conf['data_path'], dataset_conf['prefix']+self.expname)
        else:
            data_dir = os.path.join(dataset_conf['data_path'], self.expname)
        
        self.pose_file = os.path.join(dataset_conf['data_path'], self.expname, kwargs['pose_file'])
        self.start = self.conf.get_int('dataset.first', _first)
        self.end = self.conf.get_int('dataset.last', _last)

        intrinsic = self.conf.get_string('dataset.intrinsics', "intrinsics.txt")
        assoc_file = self.conf.get_string('dataset.assoc_file', "associated.txt")
        depth_prefix = self.conf.get_string('dataset.depth_prefix', "depth/")
        rgb_prefix = self.conf.get_string('dataset.rgb_prefix', "rgb/")
        digital = self.conf.get_int('dataset.digital', 3)
        depth_factor = dataset_conf['depth_factor']
        
        

        T = Timer()
        
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, 'runconf.conf')))

        
        T.tic()
        try:
            self.poses = load_pose(self.pose_file)
            self.tracking = False
            print('{0} poses are loaded.'.format(len(self.poses)))
        except:
            self.tracking = True

        T.toc('load file')
        
        
        T.tic()
        self.loader = Loader.ImageLoader(data_dir=data_dir, 
                                    mode=dataset_conf['mode'], 
                                    intrinsics_file=intrinsic, 
                                    assoc_file=assoc_file,
                                    depth_prefix=depth_prefix,
                                    rgb_prefix=rgb_prefix,
                                    depth_factor=depth_factor,
                                    digital=digital)
        T.toc("initial image loader")
        print("...load dataset intrinsics:\n {0}".format(self.loader.K))
        
        
        # parse argument to normal estimator
        T.tic()
        self.normal_estimator = NormalEstimator.NormalEstimator(self.loader.K, **self.conf.get_config('model')['normal_estimator'])
        T.toc("initial normal estimator")
        
        self.curv_estimator = DifferentialGeometryEstimator(self.loader.K, **self.conf.get_config('model')['normal_estimator'])
        
        # parse argument for sdf
        truncate = self.conf.get_int('model.grad_sdf.T')
        self.tSDF = VolGradSdf.VolumetricGradSdf(self.normal_estimator, self.curv_estimator, device=self.device, grid_dim=self.mc_resolution, voxel_size=self.voxel_size,  **self.conf.get_config('model')['grad_sdf'])
        
        print("...initial grid size {0} ".format(self.tSDF.grid_dim))
        print("...initial voxel size is {0} ".format(self.tSDF.voxel_size))

        self.pOpt = RigidOptimizer.RigidPointOptimizer(self.tSDF, torch.eye(4), device=self.device)


    
    def load_image(self, index):
        timestamp, rgb, depth = self.loader.load_img_pair(index)
        return timestamp, rgb, depth
    
    def run(self):
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('running on {0}'.format(device))
        
        T = Timer()
    
        timestamp, rgb, depth = self.load_image(self.start)
        assert (rgb is not None), "fail to load rgb image."
        assert (depth is not None), "fail to load depth image."
        
        
        T.tic()
        self.tSDF.compute_centroid(self.loader.K, depth)
        T.toc("compute shift")

        T.tic()
        self.tSDF.setup(rgb, depth, self.loader.K)
        T.toc("initial SDF")

        if self.tracking:
            pose_f = open(self.expdir + "/tracking_pose.txt",'w')
            pose_f.write(("%s" % timestamp) + " ".join(["%f"%v  for v in self.pOpt.pose.Quaternion().tolist()]) + "\n")
            tracking_pose = []
        
        for i in range(self.start, self.end-2):
            timestamp, rgb, depth = self.load_image(i+1)
            if ((rgb is None) or (depth is None)):
                print("couldn't load depth or rgb images.")
                break
            
            self.tSDF.update_counter()
            
            pose = torch.eye(4)
            if self.tracking:
                T.tic()
                conv = self.pOpt.optimize(depth, self.loader.K)
                T.toc("point optimization")
                if conv:
                    T.tic()
                    self.tSDF.update(rgb, depth, self.loader.K, self.pOpt.pose.mat)
                    T.toc('integrate depth data to sdf')

                    pose = self.pOpt.pose.mat
                    print("current pose:\n {0}".format(self.pOpt.pose.mat))
                    pose_f.write(("%s " % timestamp) + " ".join(["%f "%v  for v in self.pOpt.pose.Quaternion().tolist()]) + "\n")
                    tracking_pose.append(self.pOpt.pose)
            else:
                if isinstance(self.poses, dict):
                    element_key, pose = list(self.poses.items())[i+1-self.start]
                    print('load pose {0}'.format(element_key))
                    pose = torch.Tensor(pose).to(self.device)
                else:
                    pose = torch.Tensor(self.poses[i+1-self.start]).to(self.device)
                    print('load pose {0}'.format(i+1-self.start))
                
                print('current pose is:', pose)
                T.tic()
                self.tSDF.update(rgb, depth, self.loader.K, pose)
                T.toc('integrate depth data to sdf')
            
            #save middle pc/mesh
            if (i+1-self.start) % 500 == 0:
                self.tSDF.export_pc(self.expdir + "/checkpoint_" + str(i+1-self.start) + ".ply")
                utils.save_object(self.tSDF.to_cpu(), self.expdir + '/checkpoint_' + str(i+1) +'_tsdf.pkl')

        # save pointcloud
        T.tic()
        self.tSDF.export_pc(self.expdir + "/init_pc.ply")
        T.toc("save pointcloud")
        
        # save point cloud with curvature
        T.tic()
        self.tSDF.export_voxelcloud(self.expdir + "/init_voxel_pc.ply")
        T.toc("save point cloud with curvature info")

        # save mesh
        # Note that the marching cube function works not as good as c++ one
        T.tic()
        # boundary = tracker.tSDF.get_boundary()
        self.tSDF.export_mesh(self.expdir + "/init_mesh.ply")
        T.toc("save mesh")

        T.tic()
        # boundary = tracker.tSDF.get_boundary()
        _, _, if_save = self.tSDF.export_weighted_mesh(self.expdir + "/init_weighted_mesh.ply")
        if not if_save:
            print('!!! warning: fail to save weighted mesh')
        T.toc("save weighted mesh")
        
        T.tic()
        utils.save_object(self.tSDF.to_cpu(), self.expdir + '/tsdf.pkl')
        T.toc("save grad-sdf object")

        return self.tSDF
        
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--results_dir', type=str, default='../results/', help="result save path.")
    parser.add_argument('--expname', type=str, help='dataset name', required=True)
    parser.add_argument('--pose_file', type=str, help='pose file name under the datasets', default='poses.txt')
    parser.add_argument('--conf', type=str, help='config file name.', required=True)
    parser.add_argument('--mc_resolution', type = int, help='marching cubes resolution', default=128)
    parser.add_argument('--voxel_size', type =float, help='marching cubes resolution', default=0.02)
    
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('running on {0}'.format(device))

    fuser = Tracker(device=device,
                        conf=args.conf,
                        expname=args.expname,
                        results_dir=args.results_dir,
                        pose_file=args.pose_file,
                        mc_resolution=args.mc_resolution,
                        voxel_size=args.voxel_size)
    
    fuser.run()
   