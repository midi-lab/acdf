import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nnutils import Network, one_hot

def init(module, weight_init, bias_init):
    weight_init(module.weight.data)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


class DRIMLNet(Network):
    def __init__(self,
                 n_latent_dims=128,
                 n_units_per_layer=512,
                 n_actions=15):
        super().__init__()

        self.psi_t = ResBlock_fc_FILM(n_latent_dims,
                                      n_units_per_layer,
                                      n_units_per_layer,
                                      n_actions,
                                      layernorm=True)
        self.psi_tpk = ResBlock_fc(n_latent_dims, n_units_per_layer, n_units_per_layer)

    def forward(self, z_t, z_tpk, action):
        u_t = self.psi_t(z_t, action)
        u_tpk = self.psi_tpk(z_tpk)
        return u_t, u_tpk


class ResBlock_fc(nn.Module):
    """
    Simple 1 hidden layer resblock (for fully-connected inputs)
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 init_fn=lambda m: init(m, nn.init.orthogonal_, lambda x: nn.
                                        init.constant_(x, 0))):
        super(ResBlock_fc, self).__init__()

        self.psi_1 = nn.Linear(in_features, hidden_features, bias=True)
        self.psi_2 = nn.Linear(hidden_features, out_features, bias=True)

        self.W = init_fn(nn.Linear(in_features, out_features, bias=False))

    def forward(self, x, action=None):
        if action is not None:
            x = torch.cat([x, action], dim=1)
        residual = self.W(x)
        x = F.relu(self.psi_1(x))
        x = self.psi_2(x) + residual
        return x


class ResBlock_fc_FILM(nn.Module):
    """
    Simple 1 hidden layer resblock (for fully-connected inputs)
    """
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 action_features,
                 layernorm=True,
                 init_fn=lambda m: init(m, nn.init.orthogonal_, lambda x: nn.
                                        init.constant_(x, 0))):
        super(ResBlock_fc_FILM, self).__init__()

        self.action_features = action_features

        self.psi_1 = nn.Linear(in_features, hidden_features, bias=True)
        self.psi_2 = nn.Linear(hidden_features, out_features, bias=True)

        self.film1 = FILM(hidden_features,
                          action_features,
                          layernorm=layernorm)
        self.film2 = FILM(out_features, action_features, layernorm=layernorm)

        self.W = init_fn(nn.Linear(in_features, out_features, bias=True))

    def forward(self, x, action=None):
        residual = self.W(x)
        action = F.one_hot(action,self.action_features).float()
        x = self.film1(self.psi_1(x), action)
        x = F.relu(x)
        x = self.film2(self.psi_2(x), action)
        x = x + residual
        return x


class FILM(nn.Module):
    def __init__(self, input_dim, cond_dim, layernorm=True):
        super().__init__()
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.layernorm = nn.LayerNorm(input_dim, elementwise_affine=False) \
            if layernorm else nn.Identity()
        self.conditioning = nn.Linear(cond_dim, input_dim * 2)

    def forward(self, input, cond):
        conditioning = self.conditioning(cond)
        gamma = conditioning[..., :self.input_dim]
        beta = conditioning[..., self.input_dim:]
        if len(input.shape) > 2:
            return self.layernorm(input.permute(0, 2, 3, 1)).permute(
                0, 3, 1, 2) * gamma.unsqueeze(-1).unsqueeze(-1).repeat(
                    1, 1, input.shape[2],
                    input.shape[3]) + beta.unsqueeze(-1).unsqueeze(-1).repeat(
                        1, 1, input.shape[2], input.shape[3])
        else:
            return self.layernorm(input) * gamma + beta
