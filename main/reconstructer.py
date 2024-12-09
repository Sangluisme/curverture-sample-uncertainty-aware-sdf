import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
from datetime import datetime
from pyhocon import ConfigFactory
import numpy as np
import argparse
import GPUtil
import torch
import third.general as utils
from datasets.voxel_sampler import VoxelSampler
from third.diff_operators import gradient
from scipy.spatial import cKDTree
from third.meshing import create_mesh, create_weighted_mesh, get_surface_trace, get_threed_scatter_trace, plot_cuts
from models.loss import Loss


class Reconstructer:

    def run(self):

        print("running")
        
        best_loss = np.inf

        if self.eval:

            print("evaluating epoch: {0}".format(self.startepoch))
            my_path = os.path.join(self.expdir, 'evaluation', str(self.startepoch))

            utils.mkdir_ifnotexists(os.path.join(self.expdir, 'evaluation'))
            utils.mkdir_ifnotexists(my_path)
            self.plot_shapes(epoch=self.startepoch, path=my_path, with_cuts=True)
            return

        print("training")

        for epoch in range(0, self.nepochs + 1):
            
            samples, gt = self.dataset.get_points()

            if epoch % self.conf.get_int('training.checkpoint_freq') == 0:
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
            if (epoch < 300) and (epoch % 1000 == 0):
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch)
                print('saving checkpoint: ', epoch)
                self.save_checkpoints(epoch)
            elif epoch % self.conf.get_int('training.plot_freq') == 0:
                print('plot validation epoch: ', epoch)
                self.plot_shapes(epoch, points=gt['points'])
            if self.use_weight:    
                if epoch % (2*self.conf.get_int('training.plot_freq')) == 0:
                    print('uncertainty validation epoch: ', epoch)
                    self.plot_uncertainty(epoch)
            
            self.network.train()
            self.adjust_learning_rate(epoch)
 

            # forward pass
            samples_in = samples.clone().detach().requires_grad_(True)

            samples_pred = self.network(samples_in.cuda())

            # compute loss
            loss, loss_dic = self.loss.compute_loss(samples_in, samples_pred, gt)
                            

            # back propagation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                self.save_best_loss(epoch)
                

            if epoch % self.conf.get_int('training.checkpoint_freq') == 0:
                print('Train Epoch: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\t'.format(
                    epoch, self.nepochs, 100. * epoch / self.nepochs,
                    loss.item()))
                for (k,v) in loss_dic.items():
                        print(k, v.item())



    def plot_shapes(self, epoch, points=None, center=None, path=None):
        # plot network validation shapes
        with torch.no_grad():

            self.network.eval()

            if not path:
                path = self.plots_dir

            # filename = '{0}/{1}_{2}'.format(path, epoch, self.expname)

            surface, if_save = get_surface_trace(
                    decoder=self.network,
                    filename=os.path.join(self.expdir, "reconstructions", f"{epoch}.ply"),
                    resolution=128,
                    bounding_box=self.dataset.bounding_box,
                )


            
        
    def plot_uncertainty(self, epoch, threshold=0.0):
        with torch.no_grad():

            self.network.eval()

            filename = os.path.join(self.expdir, "reconstructions", f"uncertainty_{epoch}.ply")
            create_weighted_mesh(decoder=self.network, 
                             filename=filename, 
                             bounding_box=self.dataset.bounding_box,
                             threshold=threshold,
                             N=self.mc_resolution,
                             device=self.device)
    
    def plot_weighted_mesh(self, epoch, threshold=0.7):
         with torch.no_grad():

            self.network.eval()
            filename = os.path.join(self.expdir, "reconstructions", f"weighted_{epoch}.ply")
            create_weighted_mesh(decoder=self.network, 
                             filename=filename, 
                             bounding_box=self.dataset.bounding_box,
                             threshold=threshold,
                             N=128,
                             device=self.device)

           

    def __init__(self, device, **kwargs):

        self.home_dir = os.path.abspath(os.pardir)

        # config setting

        self.device = device
        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        
        self.exps_folder_name = kwargs['results_dir']
        self.expname = kwargs['expname']
        self.mode = kwargs['mode']
        self.eval = kwargs['eval']
    
        utils.mkdir_ifnotexists(os.path.join('../',self.exps_folder_name))
        self.expdir = os.path.join(self.exps_folder_name, self.expname+str(kwargs['input_level']))
        utils.mkdir_ifnotexists(os.path.join(self.expdir))
        
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        self.expdir = os.path.join(self.expdir, self.timestamp)
        
        # save config file for later reference
        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, 'runconf.conf')))

        
        sampling_config = self.conf.get_config('sampling')
        self.dataset_folder = os.path.join(self.conf['dataset_path'], self.expname+str(kwargs['input_level'])) 
      
        self.dataset = VoxelSampler.get_sampler(sampling_config.get_string('sampler'))(conf=sampling_config,
                                    object_path=self.dataset_folder + '/tsdf.pkl',
                                    batch_size=self.conf['training.batch_size'],
                                    device=self.device)

        self.use_weight = False
        if self.conf['network.inputs.d_out'] == 2:
            self.use_weight = True
            
        torch.save(self.dataset.bounding_box, os.path.join(self.expdir, "data_bounding"))
        print(self.dataset.bounding_box)


        self.plots_dir = os.path.join(self.expdir, 'reconstructions')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.checkpoints_path = os.path.join(self.expdir, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        self.network = utils.get_class(self.conf.get_string('network.networks'))(**self.conf.get_config(
                                                                                        'network.inputs'))
        self.loss = Loss(conf=self.conf.get_config('loss'))
        self.loss.get_loss()
        
        if torch.cuda.is_available():
            self.network.cuda()

        self.lr_schedules = self.get_learning_rate_schedules(self.conf.get_list('training.learning_rate_schedule'))
        self.nepochs = self.conf['training.nepochs']
        self.plot_freq = self.conf['training.plot_freq']
        self.checkpoints_freq = self.conf['training.checkpoint_freq']
        self.mc_resolution = self.conf['training.mc_resolution']
        self.mc_level = self.conf.get('training.mc_level', 0.0)
        self.weight_decay = self.conf.get_float('training.weight_decay')


        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.network.parameters(),
                    "lr": self.lr_schedules[0].get_learning_rate(0),
                    "weight_decay": self.weight_decay
                },
            ])

        # if continue load checkpoints


    def get_learning_rate_schedules(self, schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    utils.StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self, epoch):

        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

    def save_best_loss(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir,  "best_loss.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "best_loss.pth"))
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./conf/recon_bunny.conf')
    parser.add_argument('--results_dir', type=str, default='../exp/')
    parser.add_argument('--expname', type=str, default='recon_bunny')

    parser.add_argument('--input_level', type=int, help='point cloud extract from which level of marhcing cubes, among 64, 128, 256', default=128)
    parser.add_argument('--eval', default=False, action='store_true')

    opt = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device ='cpu'

    trainrunner = Reconstructer(
                    device=device,
                    conf= opt.conf,
                    expname=opt.expname,
                    results_dir=opt.results_dir,
                    mode = opt.mode,
                    input_level=opt.input_level,
                    eval=opt.eval
    )

    trainrunner.run()
