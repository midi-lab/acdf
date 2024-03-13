import numpy as np
import torch
import torch.nn

from .nnutils import Network, one_hot, extract

class QNet(Network):
    def __init__(self, n_features, n_actions, use_goal_conditioned, n_hidden_layers=1, n_units_per_layer=32):
        super().__init__()
        self.n_actions = n_actions
        self.use_goal_conditioned = use_goal_conditioned

        self.layers = []
        if n_hidden_layers == 0:
            self.layers.extend([torch.nn.Linear(2 * n_features, n_actions)])
        else:
            if self.use_goal_conditioned : 
                self.layers.extend(
                    [torch.nn.Linear(2 * n_features, n_units_per_layer),
                     torch.nn.Tanh()])
                self.layers.extend(
                    [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                     torch.nn.Tanh()] * (n_hidden_layers - 1))
                self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

            else:
                self.layers.extend(
                    [torch.nn.Linear(n_features, n_units_per_layer),
                     torch.nn.Tanh()])
                self.layers.extend(
                    [torch.nn.Linear(n_units_per_layer, n_units_per_layer),
                     torch.nn.Tanh()] * (n_hidden_layers - 1))
                self.layers.extend([torch.nn.Linear(n_units_per_layer, n_actions)])

        self.model = torch.nn.Sequential(*self.layers)


    def forward(self, z_feature, z):

        if self.use_goal_conditioned:   
            XX = torch.cat((z_feature, z), -1)
            q_value = self.model(XX)
            return q_value
        else:
            # XX = z
            q_value = self.model(z)
            return q_value