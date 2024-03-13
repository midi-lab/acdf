
import torch
import torch.nn as nn

from quantize import Quantize
#from gumbel_quantize import Quantize
#from quantize_ema import Quantize

from utils import Cutout

# from torchvision import transforms as transforms

# from torchvision.utils import save_image

import random

import numpy as np

class Encoder(nn.Module):

    def __init__(self, args, ncodes, inp_size):
        super(Encoder, self).__init__()

        self.args = args
        #3*32*64
        if self.args.use_ae == 'true':
            self.enc = nn.Sequential(nn.Linear(512,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        else:
            self.enc = nn.Sequential(nn.Linear(inp_size,1024), nn.LeakyReLU(), nn.Linear(1024, 512))

        self.qlst = []
        self.input_size = inp_size

        self.cutout = Cutout(1, 16)

        for nf in [1,8,32]:
            self.qlst.append(Quantize(512, ncodes, nf))

        self.qlst = nn.ModuleList(self.qlst)
        
        # self.crop = transforms.RandomCrop((30,60))
        # self.resize = transforms.Resize((32,64))
        # self.rotate = transforms.RandomRotation((-5,5))
        # self.color = transforms.ColorJitter(0.1,0.1,0.1,0.1)


    #x is (bs, 3*32*64).  Turn into z of size (bs, 256).  
    def forward(self, x, do_quantize, reinit_codebook=False, k=0): 


        xin = x.reshape((x.shape[0], -1)) 

        input_size = self.input_size
        x = self.enc(xin)

        if do_quantize:
            x = x.unsqueeze(0)
            q = self.qlst[k]
            z_q, diff, ind = q(x, reinit_codebook)
            z_q = z_q.squeeze(0)
        else:
            z_q = x
            diff = 0.0
            ind = None

        return z_q, diff, ind


class Classifier(nn.Module):

    def __init__(self, args, ncodes, maxk, inp_size):
        super(Classifier, self).__init__()

        self.args = args

        self.enc = Encoder(args, ncodes, inp_size)

        self.out = nn.Sequential(nn.Linear(512*3, 1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 10))
        #self.out = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, 3))

        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

        self.offset_embedding = nn.Embedding(maxk + 5, 512)

        self.ae_enc = nn.Sequential(nn.Linear(inp_size,1024), nn.LeakyReLU(), nn.Linear(1024,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        self.ae_q = Quantize(512,128,4) #1024,16
        self.ae_dec = nn.Sequential(nn.Linear(512, 512), nn.LeakyReLU(), nn.Linear(512, inp_size))

        self.cutout = Cutout(1, 16)

    def ae(self, x):


        print_ = (random.uniform(0,1) < 0.001)

        # if print_:
        #     save_image(x, 'x_in.png')

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
        # if print_:
        #     print('rec loss', loss)
        #     x_rec = x.reshape((x.shape[0], 3, 32, 64))
        #     save_image(x_rec, 'x_rec.png')

        return loss, z

    def encode(self,x):
        print('x shape', x.shape)

        if self.args.use_ae=='true':
            ae_loss_1, z1_low = self.ae(x)
            z1,el_1,ind_1 = self.enc(z1_low, True, False,k=0)
        else:    
            z1,el_1,ind_1 = self.enc(x, True, False,k=0)
        
        return ind_1

    #s is of size (bs, 256).  Turn into a of size (bs,3).  
    def forward(self, x, x_next, do_quantize, reinit_codebook=False, k=0, k_offset=None):


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

        #print('k_offset', k_offset)
        #print('k_offset shape', k_offset.shape)

        #print('z1 shape', z1.shape)

        offset_embed = self.offset_embedding(k_offset)

        #print('offset_embed shape', offset_embed.shape)

        #print('offset embed minmax', offset_embed.min(), offset_embed.max())

        z = torch.cat([z1,z2,offset_embed],dim=1)

        #if self.training:
        #    mixind = torch.randperm(z.shape[0])
        #    lam = 1.0 - np.random.beta(0.5,1+0.5) #values close to 1
        #    z = lam*z + (1-lam)*z[mixind]

        out = self.out(z)

        loss = el_1 + el_2 + ae_loss

        return out, loss, ind_1, ind_2, z1, z2
















