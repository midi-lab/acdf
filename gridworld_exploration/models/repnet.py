from collections import defaultdict
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from .nnutils import Network
from .phinet import PhiNet
from .invnet import InvNet
from .fwdnet import FwdNet
from .contrastivenet import ContrastiveNet
from .invdiscriminator import InvDiscriminator
from .vq_layer_kmeans import Quantize, VectorQuantizerEMA
from .infonce_net import DRIMLNet
from .autoencoder import AutoEncoder
from .proto import Proto
from .proto_utils import soft_update_params

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class RepNet(Network):
    def __init__(self,
                 n_actions,
                 input_shape=2,
                 n_latent_dims=128,
                 n_hidden_layers=1,
                 n_units_per_layer=32,
                 lr=0.001,
                 use_vq=None,
                 discrete_cfg=None, #groups and n_embed
                 use_proto=None,
                 phi_target_tau=0.05,
                 coefs=None,
                 encoder_type=None):
        super().__init__()
        self.n_actions = n_actions
        self.n_latent_dims = n_latent_dims
        self.lr = lr
        self.coefs = defaultdict(lambda: 1.0)
        if coefs is not None:
            for k, v in coefs.items():
                self.coefs[k] = v

        self.encoder_type = encoder_type
        #PhiNet - outputs features
        self.phi = PhiNet(input_shape=input_shape,
                          n_latent_dims=n_latent_dims,
                          n_units_per_layer=n_units_per_layer,
                          n_hidden_layers=n_hidden_layers,
                          final_activation=torch.nn.ReLU,
                          encoder_type=self.encoder_type)

        self.use_vq = use_vq
        self.use_proto = use_proto

        # add a VQ layer
        if self.use_vq:
            assert discrete_cfg is not None
            
            # self.vq_layer = VectorQuantizerEMA(n_latent_dims, discrete_cfg['n_embed'], discrete_cfg['groups']) 
            self.vq_layer = Quantize(n_latent_dims, discrete_cfg['n_embed'], discrete_cfg['groups'])

        if self.use_proto:
            # assert discrete_cfg is not None
            self.phi_target = PhiNet(input_shape=input_shape,
                                     n_latent_dims=n_latent_dims,
                                     n_units_per_layer=n_units_per_layer,
                                     n_hidden_layers=n_hidden_layers)
            self.phi_target.load_state_dict(self.phi.state_dict())
            self.phi_target_tau = phi_target_tau
            self.proto = Proto(n_latent_dims, 16, 0.1, 10, 3, 3) # these values may be given in args or config

        self.multi_step_inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=n_latent_dims,
                                n_units_per_layer=n_units_per_layer,
                                n_hidden_layers=n_hidden_layers)

        self.inv_discriminator = InvDiscriminator(n_actions=n_actions,
                                                  n_latent_dims=n_latent_dims,
                                                  n_units_per_layer=n_units_per_layer,
                                                  n_hidden_layers=n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=n_latent_dims,
                                            n_hidden_layers=1,
                                            n_units_per_layer=n_units_per_layer)
        self.driml_net = DRIMLNet(n_latent_dims=n_latent_dims,
                                    n_units_per_layer=n_units_per_layer,
                                    n_actions=n_actions)


        self.autoencoder = AutoEncoder(n_actions=n_actions, input_shape=input_shape, 
                                  n_latent_dims=n_latent_dims,
                                  n_units_per_layer=n_units_per_layer,
                                  n_hidden_layers=n_hidden_layers)

        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def multi_step_inverse_dynamics(self, z0, z1, a):
        if self.coefs['L_genik'] == 0.0:
            return torch.tensor(0.0)

        a_hat = self.multi_step_inv_model(z0, z1)
        return self.cross_entropy(input=a_hat, target=a)


    def contrastive_inverse_loss(self, z0, z1, a):
        if self.coefs['L_coinv'] == 0.0:
            return torch.tensor(0.0).to(device)
        N = len(z0)
        # shuffle next states
        idx = torch.randperm(N)

        a_neg = torch.randint_like(a, low=0, high=self.n_actions).to(device)

        # concatenate positive and negative examples
        z0_extended = torch.cat([z0, z0], dim=0)
        z1_extended = torch.cat([z1, z1], dim=0)
        a_pos_neg = torch.cat([a, a_neg], dim=0)
        is_fake = torch.cat([torch.zeros(N).to(device), torch.ones(N).to(device)], dim=0)

        # Compute which ones are fakes
        fakes = self.inv_discriminator(z0_extended, z1_extended, a_pos_neg)
        contrastive_loss = self.bce_loss(input=fakes, target=is_fake.float())

        return contrastive_loss

    def autoencoder_loss(self, x0, z0, a):
        encoder_out, decoder_out = self.autoencoder(z0, x0)       
        return self.mse(x0, decoder_out)


    def driml_loss(self, z0, z1, a, temp=0.1):
        if self.coefs['L_driml'] == 0.0:
            return torch.tensor(0.0).to(device)
        u_t, u_tpk = self.driml_net(z0, z1, a)

        outer_prod = torch.einsum('ik,jk->ij', u_t, u_tpk)

        scores = F.log_softmax(outer_prod / temp, -1)
        mask = torch.eye(scores.shape[-1]).to(device)

        scores = (mask * scores).sum(-1).mean()
        # maximize InfoNCE ELBO
        return -scores


    def compute_loss(self, x0, z0, z1, a, d, representation_obj):
        if representation_obj == 'genik':
            loss = self.coefs['L_genik'] * self.multi_step_inverse_dynamics (z0, z1, a)
        elif representation_obj == 'inverse':
            loss = self.coefs['L_inv'] * self.multi_step_inverse_dynamics(z0, z1, a) 
        elif representation_obj == 'contrastive':
            loss = self.coefs['L_coinv'] * self.contrastive_inverse_loss(z0, z1, a) 
        elif representation_obj == "driml":
            loss = self.coefs['L_driml'] * self.driml_loss(z0, z1, a)
        elif representation_obj == 'autoencoder':
            loss = self.coefs['L_ae'] * self.autoencoder_loss(x0, z0, a) ## todo for VAE
        else : 
            NotImplementedError

        return loss

    def train_batch(self, x0, x1, a, d, representation_obj):
        self.train()
        self.optimizer.zero_grad()
        z0 = self.phi(x0)
        z1 = self.phi(x1)
        if self.use_vq:
            z0, zq_loss0, ind  = self.vq_layer(z0)
            z1, zq_loss1, ind = self.vq_layer(z1)
            zq_loss = zq_loss0 + zq_loss1

        else:
            zq_loss = 0
        
        loss = self.compute_loss(x0, z0, z1, a, d, representation_obj)
        # loss += zq_loss
        loss = zq_loss + loss
        loss.backward()
        ctr = 0
        grad = 0.
        for name, parameters in self.named_parameters():
            if parameters.grad is not None:
                grad += torch.norm(parameters.grad)
                ctr += 1
        grad = grad / ctr
        # print('Phi grad: %f'%grad)
        self.optimizer.step()

        return loss
