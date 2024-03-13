import numpy as np
import torch
import torch.nn

from .nnutils import Network, Reshape, Permute

class PhiNet(Network):
    def __init__(self,
                 input_shape=2,
                 n_latent_dims=4,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 final_activation=torch.nn.ReLU,
                 encoder_type='mlp'):
        super().__init__()
        self.input_shape = input_shape
        self.layers = []
        self.encoder_type = encoder_type
        
        if self.encoder_type == 'nature_cnn':
            self.layers.append(Permute())
            self.layers.append(torch.nn.Conv2d(input_shape[-1], 32, kernel_size=8, stride=4))
            self.layers.append(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2))
            self.layers.append(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1))
            shape_flat = 2304
            self.layers.extend([Reshape(-1, shape_flat)])
        else:
            shape_flat = np.prod(self.input_shape)
            self.layers.extend([Reshape(-1, shape_flat)])


        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(shape_flat, n_latent_dims)])
        else:
            self.layers.extend([torch.nn.Linear(shape_flat, n_units_per_layer), torch.nn.ReLU()])
            self.layers.extend(
                [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                 torch.nn.ReLU()] * (n_hidden_layers - 1))
            self.layers.extend([
                torch.nn.Linear(n_units_per_layer, n_latent_dims),
            ])
        if final_activation is not None:
            self.layers.extend([final_activation()])
            
        self.phi = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        if self.encoder_type == 'nature_cnn':
            x = x - 0.5
        z = self.phi(x)
        return z
