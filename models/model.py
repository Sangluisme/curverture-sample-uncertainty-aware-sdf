# coding: utf-8

import torch
from torch import nn
import numpy as np
from torch.autograd import grad
import math


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


# def sine_init(m):
#     with torch.no_grad():
#         if hasattr(m, 'weight'):
#             num_input = m.weight.size(-1)
#             # See supplement Sec. 1.5 for discussion of factor 30
#             m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)

@torch.no_grad()
def sine_init(m, w0):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-np.sqrt(6 / num_input) / w0,
                          np.sqrt(6 / num_input) / w0)


@torch.no_grad()
def first_layer_sine_init(m):
    if hasattr(m, 'weight'):
        num_input = m.weight.size(-1)
        m.weight.uniform_(-1 / num_input, 1 / num_input)


class SineLayer(nn.Module):
    """A Sine non-linearity layer.
    """
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f"SineLayer(w0={self.w0})"


class SIREN(nn.Module):
    
    def __init__(self, 
                 d_in, 
                 dims, 
                 d_out, 
                 w0=30, 
                 ww=None, 
                 nonlinearity='sine',
                 weight_init = None):
        super().__init__()
        
        self.w0 = w0
        if ww is None:
            self.ww = w0
        else:
            self.ww = ww
        
        self.d_out = d_out
        
        
        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'sine':(SineLayer(), sine_init, first_layer_sine_init),
                         'relu':(nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid':(nn.Sigmoid(), init_weights_xavier, None),
                         'tanh':(nn.Tanh(), init_weights_xavier, None),
                         'selu':(nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus':(nn.Softplus(), init_weights_normal, None),
                         'elu':(nn.ELU(inplace=True), init_weights_elu, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.weight_activation = nn.Softplus(beta=1)

        net = []
        net.append(nn.Sequential(
            nn.Linear(d_in, dims[0]),
            SineLayer(self.w0)
        ))

        for i in range(1, len(dims)):
            net.append(nn.Sequential(
                nn.Linear(dims[i-1], dims[i]),
                SineLayer(self.ww)
            ))

        net.append(nn.Sequential(
            nn.Linear(dims[-1], d_out),
        ))

        self.net = nn.Sequential(*net)
        if self.weight_init is not None:
            self.net.apply(lambda module: self.weight_init(module, self.ww))

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)
    
    def apply_activations(self, x):
        
        y = x
        y[:,-1] = torch.sigmoid(x[:,-1])
        
        return y

    def forward(self, input):
            
        coords = input
        y = self.net(coords)
        
        if self.d_out > 1:
            # y = self.apply_activations(y)
            out2 = self.weight_activation(y[:,-1:])
            out1 = y[:,:1]
            
            y = torch.cat((out1, out2), dim=1)
        return y




from models.embedder import get_embedder


class ImplicitNet(nn.Module):
    def __init__(
        self,
        d_in,
        dims,
        d_out,
        skip_in=(),
        geometric_init=True,
        radius_init=1,
        multires = 0,
        beta=100
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out]

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.embed_fn = None
        self.d_out = d_out
        
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
         
            self.embed_fn = embed_fn
            dims[0] = input_ch
            
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        # vanilla relu
        else:
            # self.activation = nn.LeakyReLU()
            self.activation = nn.ReLU()
        
        self.weight_activation = nn.Softplus(beta=1)

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
            
            # if true preform preform geometric initialization
            if geometric_init:
                
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -radius_init)
                    
                elif multires > 0 and layer == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and layer in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                
            
            lin = nn.utils.weight_norm(lin)
            
            setattr(self, "lin" + str(layer), lin)
    
    def apply_activations(self, x):
        
        y = x
        
        # y[:,-1] = torch.sigmoid(x[:,-1])
        y[:,-1] = self.weight_activation(x[:,-1])
            
        return y
    
    def forward(self, input):
    
        if self.embed_fn is not None:
            input = self.embed_fn(input)
            
        x = input

        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))

            if layer in self.skip_in:
                x = torch.cat([x, input], -1) / np.sqrt(2)

            x = lin(x)

            if layer < self.num_layers - 2:
                x = self.activation(x)
        
        if self.d_out > 1:
            # x = self.apply_activations(x)
            out2 = self.weight_activation(x[:,-1:])
            out1 = x[:,:1]
            
            x = torch.cat((out1, out2), dim=1)

        return x
    

