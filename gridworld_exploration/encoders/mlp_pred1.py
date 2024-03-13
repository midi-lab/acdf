import torch
import torch.nn as nn
from quantize import Quantize
from utils import Cutout
import random
import numpy as np
from encoders.mlp_enc1 import Encoder as Encoder

from collections import defaultdict
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F

from models.phinet import PhiNet
from models.invnet import InvNet
from models.fwdnet import FwdNet
from models.contrastivenet import ContrastiveNet
from models.invdiscriminator import InvDiscriminator
from models.autoencoder import AutoEncoder

from mixer import MLP_Mixer

class Classifier(nn.Module):

    def __init__(self, args, ncodes, maxk, inp_size, inp_dim):
        super(Classifier, self).__init__()

        self.inp_size = inp_size
        self.args = args

        self.enc = Encoder(args, ncodes, inp_size, inp_dim)

        self.project = nn.Linear(512,64)#nn.Sequential(nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512,64))
        self.byol_predictor = nn.Sequential(nn.Linear(64 + 10, 512), nn.LeakyReLU(), nn.Linear(512,512), nn.LeakyReLU(), nn.Linear(512, 64))

        self.out = nn.Sequential(nn.Linear(512*3, 1024), nn.LeakyReLU(), nn.Linear(1024,10)) #no-bn        

        if (self.args.use_forward):
            self.out_fwd = nn.Sequential(nn.Linear(512+10, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, ncodes)) # Following the above by assuming <= 10 actions
        #self.out = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3))

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.offset_embedding = nn.Embedding(maxk + 5, 512)

        self.ae_enc = nn.Sequential(nn.Linear(inp_size,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.ae_q = Quantize(512,128,4) #1024,16
        self.ae_dec = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_size))

        #self.gru = nn.GRUCell(10 + 512, 512)
        #self.pred = nn.Sequential(nn.Linear(512,10))


        #self.forward_mlp = nn.Sequential(nn.Linear(512+10, 1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024,1024), nn.Dropout(0.5), nn.LeakyReLU(), nn.Linear(1024, ncodes))

        self.cutout = Cutout(1, 16)

        self.cross_entropy = torch.nn.CrossEntropyLoss()

        n_actions = 4
        n_units_per_layer=32
        n_hidden_layers=1
        n_latent_dims=128
        input_shape=2
        self.inv_model = InvNet(n_actions=n_actions,
                                n_latent_dims=128,
                                n_units_per_layer=32,
                                n_hidden_layers=1)

        self.inv_discriminator = InvDiscriminator(n_actions=n_actions,
                                                  n_latent_dims=n_latent_dims,
                                                  n_units_per_layer=n_units_per_layer,
                                                  n_hidden_layers=n_hidden_layers)
        self.discriminator = ContrastiveNet(n_latent_dims=n_latent_dims,
                                            n_hidden_layers=1,
                                            n_units_per_layer=n_units_per_layer)
        # self.autoencoder = AutoEncoder(n_actions=n_actions, input_shape=input_shape, 
        #                           n_latent_dims=n_latent_dims,
        #                           n_units_per_layer=n_units_per_layer,
        #                           n_hidden_layers=n_hidden_layers)


    def ae(self, x):


        print_ = (random.uniform(0,1) < 0.001)

        if print_:
            save_image(x, 'x_in.png')

        x_in = x*1.0

        if self.training:
            x = self.cutout.apply(x)

        x = x.reshape((x.shape[0], -1))
        x_in = x_in.reshape((x_in.shape[0], -1))

        x = self.ae_enc(x).unsqueeze(0)


        z, diff, ind = self.ae_q(x)
        z = z.squeeze(0)
        x = z*1.0
        x = self.ae_dec(x)

        loss = self.mse(x,x_in.detach())*0.1
        loss += diff

        print_ = False
        if print_:
            # print('rec loss', loss)
            x_rec = x.reshape((x.shape[0], 3, 32, 64))
            save_image(x_rec, 'x_rec.png')

        return loss, z

    def encode(self,x):

        if self.args.use_ae=='true':
            ae_loss_1, z1_low = self.ae(x)
            z1,el_1,ind_1 = self.enc(z1_low, True, False,k=0)
        else:
            z1,el_1,ind_1 = self.enc(x, True, False,k=0)

        return ind_1, z1

    def encode_emb(self,x):
        z1,el_1,ind_1 = self.enc(x, True, False,k=0)

        return z1

    def inverse_dynamics(self, z0, z1, a):
        a_hat = self.inv_model(z0, z1)
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


    def compute_loss(self, z0, z1, a):

        loss = 1.0 * self.inverse_dynamics(z0, z1, a)

        # if representation_obj == 'genik':
        #     loss = self.coefs['L_genik'] * self.multi_step_inverse_dynamics (z0, z1, a)
        # elif representation_obj == 'inverse':
        #     loss = self.coefs['L_inv'] * self.multi_step_inverse_dynamics(z0, z1, a) 
        # elif representation_obj == 'contrastive':
        #     loss = self.coefs['L_coinv'] * self.contrastive_inverse_loss(z0, z1, a) 
        # elif representation_obj == "driml":
        #     loss = self.coefs['L_driml'] * self.driml_loss(z0, z1, a)
        # elif representation_obj == 'autoencoder':
        #     loss = self.coefs['L_ae'] * self.autoencoder_loss(x0, z0, a) ## todo for VAE
        # else : 
        #     NotImplementedError

        return loss




    def byol(self, z_last, z_next, a):
        h_last = self.project(z_last)
        h_next = self.project(z_next)

        pred = self.byol_predictor(torch.cat([h_last, F.one_hot(a,10)], dim=1))

        cs = nn.CosineSimilarity(dim=1)

        loss = -1 * cs(pred, h_next.detach()).mean()

        return loss

    # #s is of size (bs, 256).  Turn into a of size (bs,3).
    def forward(self, x, x_next, do_quantize, reinit_codebook=False, k=0, k_offset=None, action=None):

        if self.args.use_ae=='true':
            ae_loss_1, z1_low = self.ae(x)
            ae_loss_2, z2_low = self.ae(x_next)
            ae_loss = ae_loss_1 + ae_loss_2

            z1,el_1,ind_1 = self.enc(z1_low.detach(), do_quantize, reinit_codebook,k=k)
            z2,el_2,ind_2 = self.enc(z2_low.detach(), do_quantize, reinit_codebook,k=k)

        else:
            z1,el_1,ind_1 = self.enc(x, do_quantize, reinit_codebook,k=k)
            z2,el_2,ind_2 = self.enc(x_next, do_quantize, reinit_codebook,k=k)
            ae_loss = 0.0

        offset_embed = self.offset_embedding(k_offset)

        z = torch.cat([z1,z2, offset_embed],  dim=1)

        out = self.out(z)

        loss = el_1 + el_2 + ae_loss

        return out, loss, ind_1, ind_2, z1, z2
