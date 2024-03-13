

import torch
from torch import nn, einsum
import torch.nn.functional as F
import random

class Quantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, n_embed, straight_through=True):
        super().__init__()

        embedding_dim = num_hiddens
        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-4

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, embedding_dim)

    def forward(self, z):

        #print('z shape gumbel in', z.shape)
        #print('proj', self.proj)



        z = z.squeeze(0).unsqueeze(-1).unsqueeze(-1)


        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(z)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()
        diff = 0.0

        ind = soft_one_hot.argmax(dim=1)

        z_q = z_q.squeeze(2).squeeze(2).unsqueeze(0)

        #print('zq gumbel out', z_q.shape)

        return z_q, diff, ind


