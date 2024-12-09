import torch
from torch import nn
import numpy as np
import third.general as utils
from models.model import SIREN, ImplicitNet

class WeightImplictNetwork(nn.Module):
    def __init__(self,
                 implicit_conf,
                 occupancy_conf,
                 color_conf=None,
                 with_color=False) -> None:
        super().__init__()
        
        self.sdf_conf = implicit_conf
        self.w_conf= occupancy_conf
        self.with_color = with_color
        self.color_conf = color_conf
        
        self.sdf_net = utils.get_class(implicit_conf['type'])(
            n_in_features=3,
            n_out_features=1,
            hidden_layer_config=implicit_conf['hidden_layer_nodes'],
            skip_in = implicit_conf['skip_in'],
            w0=implicit_conf['w0'],
            ww=implicit_conf.get('ww', None),
            activations=implicit_conf['activations'],
            beta = implicit_conf.get('beta', 0),
            geometric_init=implicit_conf.get('geometric_init', True),
            radius_init = implicit_conf.get('radius_init', 1),
            multires=implicit_conf.get('multires', 0)
        )
        
        self.w_net = utils.get_class(occupancy_conf['type'])(
            n_in_features=3,
            n_out_features=1,
            hidden_layer_config=occupancy_conf['hidden_layer_nodes'],
            skip_in = occupancy_conf['skip_in'],
            w0=occupancy_conf['w0'],
            ww=occupancy_conf.get('ww', None),
            activations=occupancy_conf['activations'],
            beta = occupancy_conf.get('beta', 0),
            geometric_init=occupancy_conf.get('geometric_init', True),
            radius_init = implicit_conf.get('radius_init', 1),
            multires=occupancy_conf.get('multires', 0)
        )
        print("implicit network:")
        print(self.sdf_net)
        
        print("occupancy network:")
        print(self.w_net)
        
        if with_color:
            self.color_net = utils.get_class(color_conf['type'])(
                n_in_features=3,
                n_out_features=3,
                hidden_layer_config=color_conf['hidden_layer_nodes'],
                skip_in = color_conf['skip_in'],
                w0=color_conf['w0'],
                ww=color_conf.get('ww', None),
                activations=color_conf['activations'],
                beta = color_conf.get('beta', 0),
                geometric_init=color_conf.get('geometric_init', True),
                multires=color_conf.get('multires', 0)
            )
            print("color network:")
            print(self.color_net)
        
        
        
    def forward(self, x):
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The model input containing of size Nx3

        Returns
        -------
        dict
            Dictionary of tensors with the input coordinates under 'model_in'
            and the model output under 'model_out'.
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        sdf = self.sdf_net(coords)
        w = self.w_net(coords)
        # print(w.mean())
        if not self.with_color:
            y = torch.cat((sdf, w), dim=-1)
        if self.with_color:
            c = self.color_net(coords)
            y = torch.cat((sdf, w, c), dim=-1)
        return {"model_in": coords_org, "model_out": y}
    
        
class OccupancyImplictNetwork(nn.Module):
    def __init__(self,
                 model_conf,
                 color_conf=None,
                 with_color=False) -> None:
        super().__init__()
        
        self.model_conf = model_conf
        self.with_color = with_color
        self.color_conf = color_conf
        
        self.network = utils.get_class(model_conf['type'])(
            n_in_features=3,
            n_out_features=model_conf['d_out'],
            hidden_layer_config=model_conf['hidden_layer_nodes'],
            skip_in = model_conf['skip_in'],
            w0=model_conf['w0'],
            ww=model_conf.get('ww', None),
            activations=model_conf['activations'],
            beta = model_conf.get('beta', 0),
            geometric_init=model_conf.get('geometric_init', True),
            radius_init = model_conf.get('radius_init', 1),
            multires=model_conf.get('multires', 0)
        )
        
        
        print("implicit network:")
        print(self.network)
        
        
        if with_color:
            self.color_net = utils.get_class(color_conf['type'])(
                n_in_features=3,
                n_out_features=3,
                hidden_layer_config=color_conf['hidden_layer_nodes'],
                skip_in = color_conf['skip_in'],
                w0=color_conf['w0'],
                ww=color_conf.get('ww', None),
                activations=color_conf['activations'],
                beta = color_conf.get('beta', 0),
                geometric_init=color_conf.get('geometric_init', True),
                multires=color_conf.get('multires', 0)
            )
            print("color network:")
            print(self.color_net)
        
        
        
    def forward(self, x):
        """Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            The model input containing of size Nx3

        Returns
        -------
        dict
            Dictionary of tensors with the input coordinates under 'model_in'
            and the model output under 'model_out'.
        """
        # Enables us to compute gradients w.r.t. coordinates
        coords_org = x.clone().detach().requires_grad_(True)
        coords = coords_org
        output = self.network(coords)
        # sdf = output[...,:1]

        # if output.shape[1] > 1:
        # # split into sdf and occupancy
        #     w = output[...,-1:]
        #     # apply sigmoid to weight to ensure [0,1]
        #     w = torch.sigmoid(w)
        
        # else:
        #     w = torch.ones_like(sdf)
        
        # # print(w.mean())
        # if not self.with_color:
        #     y = torch.cat((sdf, w), dim=-1)
        # if self.with_color:
        #     c = self.color_net(coords)
        #     y = torch.cat((sdf, w, c), dim=-1)
        return {"model_in": coords_org, "model_out": output}
    
        
               