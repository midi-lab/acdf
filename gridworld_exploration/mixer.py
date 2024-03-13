import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import Sequential
import random

drop_p = 0.5

class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.P = patch_size

    def forward(self, x):
        P = self.P
        B,C,H,W = x.shape                       # [B,C,H,W]                 4D Image
        x = x.reshape(B,C, H//P, P , W//P, P)   # [B,C, H//P, P, W//P, P]   6D Patches
        x = x.permute(0,2,4, 1,3,5)             # [B, H//P, W//P, C, P, P]  6D Swap Axes
        x = x.reshape(B, H//P * W//P, C*P*P)    # [B, H//P * W//P, C*P*P]   3D Patches
                                                # [B, n_tokens, n_pixels]
        return x

class PerPatchMLP(nn.Module):
    def __init__(self, n_pixels, n_channel, n_tokens):
        super().__init__()
        self.mlp = nn.Linear(n_pixels, 128)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(128, n_channel)

        if torch.cuda.is_available():
            self.pe = nn.Parameter(torch.randn(1, n_tokens, 128).cuda() * 0.1)
        else:
            self.pe = nn.Parameter(torch.randn(1, n_tokens, 128) * 0.1)

    def forward(self, x):      
        h = self.mlp(x) + self.pe.repeat(x.shape[0],1,1)  # x*w:  [B, n_tokens, n_pixels] x [n_pixels, n_channel]   
        h = self.gelu(h)                    #       [B, n_tokens, n_channel]                                                
        h = self.mlp2(h)       
                    
        return h

class TokenMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden, layer_ind):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_channel)
        self.mlp1 = nn.Linear(n_tokens, n_hidden)       
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(n_hidden, n_tokens)

        self.gatef_read = nn.Sequential(nn.LayerNorm(n_channel), nn.Linear(n_channel,1))
        self.gatef_write = nn.Sequential(nn.LayerNorm(n_channel), nn.Linear(n_channel,1))

        self.drop = nn.Dropout(drop_p)


        self.layer_ind = layer_ind

    def forward(self, X):

        if self.layer_ind <= 0:
            return X

        z = 1.0*X
        z = self.layer_norm(z)                  # z:    [B, n_tokens, n_channel]

        if self.training:
            write_gate = F.gumbel_softmax(self.gatef_write(X), dim=1, hard=True).repeat(1,1,X.shape[2])         # gate: [B, n_tokens, n_channel]
            read_gate = F.gumbel_softmax(self.gatef_read(X), dim=1, hard=True).repeat(1,1,X.shape[2])
        else:
            write_gate = F.gumbel_softmax(self.gatef_write(X), dim=1, tau=0.0001, hard=True).repeat(1,1,X.shape[2])
            read_gate = F.gumbel_softmax(self.gatef_read(X), dim=1, tau=0.0001, hard=True).repeat(1,1,X.shape[2])

        #read_gate = F.sigmoid(self.gatef_read(X))

        if random.uniform(0,1) < 0.001:
            print('layer-ind', self.layer_ind)
            print('read-gate', read_gate.mean(dim=2).mean(dim=0))
            print('write-gate', write_gate.mean(dim=2).mean(dim=0))

        z = self.drop(z) * write_gate

        z = z.permute(0, 2,1)                   # z:    [B, n_channel, n_tokens]
        z = self.gelu(self.mlp1(z))             # z:    [B, n_channel, n_hidden] 
        z = self.mlp2(z)                        # z:    [B, n_channel, n_tokens]
        z = z.permute(0, 2,1)                   # z:    [B, n_tokens, n_channel]
        #gate = gate.permute(0,2,1).repeat(1,1,z.shape[2])
        U = X + read_gate*self.drop(z)                               # U:    [B, n_tokens, n_channel]
        return U

class ChannelMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_channel)
        self.mlp3 = nn.Linear(n_channel, n_hidden)
        self.gelu = nn.GELU()
        self.mlp4 = nn.Linear(n_hidden, n_channel)

        self.drop = nn.Dropout(drop_p)

    def forward(self, U):
        z = self.layer_norm(U)                  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))             # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)                        # z: [B, n_tokens, n_channel]
        Y = U + self.drop(z)                               # Y: [B, n_tokens, n_channel]
        return Y

class OutputMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_output):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens*1, n_channel])

        self.ln2 = nn.LayerNorm(n_channel)        

        #self.layer_norm = nn.LayerNorm(n_channel)

        self.out_mlp = nn.Linear(n_channel, n_output)

        self.gatef = nn.Linear(n_channel, 1)

    def forward(self, x):
        if self.training:
            gate = F.gumbel_softmax(self.gatef(self.layer_norm(x)),dim=1, tau=1.0, hard=False).repeat(1,1,x.shape[2])
        else:
            gate = F.gumbel_softmax(self.gatef(self.layer_norm(x)),dim=1, tau=0.0001, hard=False).repeat(1,1,x.shape[2])

        if random.uniform(0,1) < 0.01:
            print('gate', gate[:,:,0].mean(dim=0))

        #x = self.ln2(x)                  # x: [B, n_tokens, n_channel]
        x = gate*x
        x = x.mean(dim=1)                       # x: [B, n_channel] 
        x = self.ln2(x)
        return self.out_mlp(x)                  # x: [B, n_output]

class MLP_Mixer(nn.Module):
    def __init__(self, n_layers, n_channel, n_hidden, n_output, image_size_h, image_size_w, patch_size, n_image_channel):
        super().__init__()

        n_tokens = (image_size_w // patch_size) * (image_size_h // patch_size)
        n_pixels = n_image_channel * patch_size**2

        self.ImageToPatch = ImageToPatches(patch_size = patch_size)
        self.PerPatchMLP = PerPatchMLP(n_pixels, n_channel, n_tokens)
        self.MixerStack = nn.Sequential(*[
            nn.Sequential(
                TokenMixingMLP(n_tokens, n_channel, n_hidden, layer_ind),
                ChannelMixingMLP(n_tokens, n_channel, n_hidden)
            ) for layer_ind in range(n_layers)
        ])
        self.OutputMLP = OutputMLP(n_tokens, n_channel, n_output)

    def forward(self, x):
        x = self.ImageToPatch(x)
        x = self.PerPatchMLP(x)
        x = self.MixerStack(x)
        return self.OutputMLP(x)

if __name__ == "__main__":

    #mixer = MLP_Mixer(6, 64, 64, 512, 8, 8, 8, 1)
    mixer = MLP_Mixer(6, 64, 64, 512*2, 8, 8*1, 8, 1)

    x = torch.randn((1, 1, 8, 8))

    print(mixer(x).shape)
