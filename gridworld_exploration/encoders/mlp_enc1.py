import torch
import torch.nn as nn
import torch.nn.functional as F
#from quantize import Quantize
from vqema import VectorQuantizerEMA as Quantize
#from gumbel_quantize import Quantize
#from quantize_ema import Quantize
from utils import Cutout
# from torchvision import transforms as transforms
# from torchvision.utils import save_image
from coordconv import AddCoords
from mixer import MLP_Mixer
import random
import numpy as np

class Encoder(nn.Module):

    def __init__(self, args, ncodes, inp_size, inp_dim):
        super(Encoder, self).__init__()

        self.args = args
        if self.args.exo_noise == "two_maze":
            inp_size_use = inp_size//2  
        else:
            inp_size_use = inp_size

        self.inp_size = inp_size
        self.inp_dim = inp_dim

        if self.args.use_ae == 'true':
            self.enc = nn.Sequential(nn.Linear(512,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
        else:
            if self.args.exo_noise == "two_maze" and (args.data == 'maze' or args.data == 'periodic-cart') :
                #self.enc = nn.Sequential(nn.Linear(inp_size_use*2,1024), nn.LeakyReLU(), nn.Linear(1024, 512*2))                
                self.enc = MLP_Mixer(3, 32, 32, 512, args.rows, args.rows*(args.num_exo+1), args.rows, 3)
                self.enc_l = nn.Sequential(nn.Linear(inp_size_use,1024), nn.LeakyReLU(), nn.Linear(1024, 512*1))
                self.enc_r = nn.Sequential(nn.Linear(inp_size_use,1024), nn.LeakyReLU(), nn.Linear(1024, 512*1))
                # self.enc = nn.Sequential(nn.Linear(1000,1024), nn.LeakyReLU(), nn.Linear(1024, 512))
            
            elif self.args.exo_noise == "multi_maze":
                if args.data == 'vis-maze':
                    ### self.enc = MLP_Mixer(6, 64, 64, 512*2, 8, 8*9, 8, 1)
                    ### so the thing to change is 8 means that the num rows/cols is 8 - and 8*9 for 9 mazes along colum-axis 
                    ### and 8 is the patch size (set to be row/col length)
                    ### self.enc = MLP_Mixer(6, 576, 576, 512*2, args.rows * sqrt(args.num_exo + 1), sqrt(args.rows x (args.num_exo+1)), single_maze_len, 1)
                    if args.obs_type == 'high_dim':
                        # x = args.rows * int(np.sqrt(args.num_exo + 1))
                        # y = args.rows * int(np.sqrt(args.num_exo + 1)) * (args.num_exo+1)
                        # self.enc = MLP_Mixer(6, 64, 64, 512, x, y, 18, 1)
                        print('HIGH DIM!')
                        #x = 18
                        #y = 18*(args.num_exo+1)
                        # x = 12
                        # y = 108
                        # self.enc = MLP_Mixer(3, 32, 32, 512*1, x, y, 12, 3)
                        x = self.inp_dim
                        y = self.inp_dim*(args.num_exo+1)
                        # self.enc = MLP_Mixer(3, 32, 32, 512*1, x, y, self.inp_dim, 3)
                        self.enc = MLP_Mixer(3, 32, 32, 512, x, y, self.inp_dim, 3)

                    else:
                        ### TODO : Need to fix here
                        ### for pixels of size : 80x80x3
                        # self.enc = MLP_Mixer(6, 128, 128, 512*1, self.inp_dim, self.inp_dim*9, self.inp_dim, 3) 
                        x = self.inp_dim
                        y = self.inp_dim*(args.num_exo+1)
                        self.enc = MLP_Mixer(3, 128, 128, 512, 80, 80*9, 80, 3) 

                        # self.enc = MLP_Mixer(6, 128, 128, 512*1, x, y, self.inp_dim, 3) 
                        # self.enc = MLP_Mixer(6, 64, 64, 512*1, 80, 80*9, 20, 3)
                        # self.enc = MLP_Mixer(6, 256, 256, 512*1, 80, 80*9, 40, 3) 
                        # self.enc = MLP_Mixer(6, 256, 256, 512*1, 80, 80*9, 80, 3) 
            else:
                self.enc = nn.Sequential(nn.Linear(inp_size_use,1024), nn.LeakyReLU(), nn.Linear(1024, 512*1))

                #self.enc = MLP_Mixer(6, 64, 64, 512*2, args.rows, args.rows*1, args.rows, 1)

        self.post_enc = nn.Sequential(nn.Linear(256,512))
        self.coords = AddCoords()

        self.inp_size_use = inp_size_use
        self.inp_size = inp_size


        self.qlst = []

        #self.cutout = Cutout(1, 16)

        print('input inp size', inp_size)
        # self.emb = nn.Linear(inp_size//2, inp_size//2)
        if self.args.data == 'vis-maze' or self.args.data == 'maze':

            if self.args.exo_noise == 'two_maze':
                self.emb_l = nn.Linear(inp_size//2, inp_size//2)
                self.emb_r = nn.Linear(inp_size//2, inp_size//2)
    
                encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=0.0,batch_first=True)
                self.trans = nn.TransformerEncoder(encoder_layer, num_layers=4)
                #self.mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, dropout=0.0, batch_first=True)

                if torch.cuda.is_available():
                    self.qemb = nn.Parameter(0.1 * torch.randn(size=(1,1,512)).cuda())
                else:
                    self.qemb = nn.Parameter(0.1 * torch.randn(size=(1,1,512)))

                self.emb = nn.Linear(self.inp_size//2, self.inp_size//2)

                self.emb0 = nn.Sequential(nn.Linear(1, 32), nn.LeakyReLU(), nn.Linear(32,16))
                self.emb1 = nn.Linear(self.inp_size*16, self.inp_size//2 * 2)
                self.emb2 = nn.Linear(self.inp_size//4, self.inp_size//2)

            
            elif self.args.exo_noise == 'multi_maze':
                ### TODO HERE
                encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=4, dropout=0.0,batch_first=True)
                self.trans = nn.TransformerEncoder(encoder_layer, num_layers=4)                
                if torch.cuda.is_available():
                    self.qemb = nn.Parameter(0.1 * torch.randn(size=(1,1,512)).cuda())
                else:
                    self.qemb = nn.Parameter(0.1 * torch.randn(size=(1,1,512)))

            else:
                self.emb = nn.Linear(inp_size, 1000)


        for nf in [1]:
            self.qlst.append(Quantize(512, ncodes, nf))

        self.qlst = nn.ModuleList(self.qlst)

        if torch.cuda.is_available():

            self.pos_emb = nn.Parameter(0.1 * torch.randn(1,128,16).cuda())

            self.w1 = nn.Parameter(torch.zeros(1,2,1,2).float().cuda())
        else:
            self.w = nn.Parameter(torch.zeros(1,2).float())

    def forward(self, x, do_quantize=True, reinit_codebook=False, k=0):

        mlp_enc_use = True

        if torch.cuda.is_available():
            x = x.cuda()

        if True:

            if self.args.exo_noise == 'two_maze' or self.args.exo_noise == 'multi_maze':
                xin = x*1.0

                xin = self.coords(xin)
            else:
                xin = x.reshape((x.shape[0], -1))
            
            if self.training:
                xin += 0.01 * torch.randn_like(xin)

            # x = self.enc(xin)            
            #mu = x[:,:256]
            #std = torch.exp(x[:,256:])
            #klb_loss = (mu**2 + std**2 - 2*torch.log(std)).sum(dim=1).mean() * 0.0001
            #if self.training:
            #    x = mu + torch.randn_like(std) * std
            #else:
            #    x = mu
            # klb_loss = 0.0
            #x = self.post_enc(x)

            x = self.enc(xin)    
          
            mu = x[:,:256]

            if False and self.kl_penalty > 0 and self.training:
                std = F.softplus(x[:,256:])
                klb_loss = (mu**2 + std**2 - 2*torch.log(std)).sum(dim=1).mean() * self.kl_penalty
                x = mu + torch.randn_like(std) * std
            else:    
                klb_loss = 0.0
                x = mu

            x = self.post_enc(x)


        if do_quantize:
            x = x.unsqueeze(0)
            q = self.qlst[k]
            z_q, diff, ind = q(x, reinit_codebook)
            z_q = z_q.squeeze(0)
        else:
            z_q = x
            diff = 0.0
            ind = None

        diff += klb_loss

        return z_q, diff, ind
