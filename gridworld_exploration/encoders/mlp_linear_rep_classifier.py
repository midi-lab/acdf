import torch
import torch.nn as nn
import torch.nn.functional as F

from vqema import VectorQuantizerEMA as Quantize
from utils import Cutout
from models.nnutils import Network, one_hot


class LinearNet(Network):

    def __init__(self, args, num_states, n_latent_dims=512, n_hidden_layers=1, n_units_per_layer=128):
        super().__init__()
        self.num_states = num_states
        self.n_latent_dims = n_latent_dims
        # self.layers = []
        # self.layers.extend([torch.nn.Linear(n_latent_dims, n_units_per_layer), torch.nn.ReLU()] )
        # self.layers.extend([torch.nn.Linear(n_units_per_layer, n_units_per_layer), torch.nn.ReLU()] * (n_hidden_layers - 1))
        # self.layers.extend([torch.nn.Linear(n_units_per_layer, num_states)])
        # self.l_model = torch.nn.Sequential(*self.layers)
        self.l_model = nn.Sequential(nn.Linear(n_latent_dims, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, self.num_states))

    def forward(self, z1):
        
        pred_latent_logits = self.l_model(z1)
        pred_latent = (pred_latent_logits.argmax()).reshape(1)

        return pred_latent_logits, pred_latent


class LinearClassifier(nn.Module):

    def __init__(self, args, ncodes, inp_size, num_states, use_discrete_codes, embed_dim):
        super(LinearClassifier, self).__init__()
        self.args = args        
        self.use_discrete_codes = use_discrete_codes
        self.ncodes = ncodes
        self.embed_dim = embed_dim
        if self.use_discrete_codes:
            self.l_net = LinearNet(args, num_states, n_latent_dims=ncodes)
        else:
            self.l_net = LinearNet(args, num_states, n_latent_dims=512)        

    def forward(self, y1, y1_, embed_pred_y1, embed_pred_y1_):
        pred_latent_logits_last, pred_latent = self.l_net(embed_pred_y1)
        pred_latent_logits_next, pred_latent = self.l_net(embed_pred_y1_)

        loss_last = F.cross_entropy(input=pred_latent_logits_last, target=y1).mean()  

        loss = loss_last

        return loss

    def encode(self, z1, ind):

        if self.use_discrete_codes:
            ind_one_hot = one_hot(ind, depth=self.ncodes)
            pred_latent_logits, pred_latent = self.l_net(ind_one_hot)
        else:
            pred_latent_logits, pred_latent = self.l_net(z1)

        return pred_latent, pred_latent_logits


    def predict(self, y, z ):
        predicted_latent_logits = self.l_net.l_model(z)
        predicted_latent = (predicted_latent_logits.argmax(dim=1))
        return predicted_latent



