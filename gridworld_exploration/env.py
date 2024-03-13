
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, grad
import random
import numpy as np
from torchvision.utils import save_image

class Env:

    def __init__(self):

        bs = 1

        train_loader = torch.utils.data.DataLoader(datasets.MNIST('data',
                                                         download=True,
                                                          train=True,
                                                          transform=transforms.Compose([
                                                              transforms.Resize((32,32)),
                                                              transforms.ToTensor(), # first, convert image to PyTorch tensor
                                                          ])),
                                           batch_size=bs,
                                            drop_last=True,
                                           shuffle=True)
    

        self.x_lst = []

        self.inp_size = 3*32*32

        for j in range(0,10):
            self.x_lst.append([])

        for (x,y) in train_loader:

            self.x_lst[y[0].item()].append(x[0])

        print(len(self.x_lst))
        for j in range(0, 10):
            self.x_lst[j] = self.x_lst[j][0:100]

    def initial_state(self):
        randind1 = random.randint(0,99)
        randind2 = random.randint(0,99)

        start_class = 9

        x1 = torch.cat(self.x_lst[start_class], dim=0).unsqueeze(1)[randind1:randind1+1]
        y1 = torch.zeros(1).long() + start_class

        randclass = random.randint(0,9)
        x2 = torch.cat(self.x_lst[randclass], dim=0).unsqueeze(1)[randind2:randind2+1]
        y2 = torch.zeros(1).long() + randclass

        x1 = x1.repeat(1,3,1,1)
        x2 = x2.repeat(1,3,1,1)

        c1 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0
        c2 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0

        #c1 = torch.rand(1,3,1,1)
        #c2 = torch.rand(1,3,1,1)
        #c1 = torch.ones(1,3,1,1)
        #c2 = torch.ones(1,3,1,1)

        print(y1.shape)
        print(c1.shape)
        print(x1.shape)

        raise Exception()

        return y1,c1,y2,c2,x1,x2


    def transition_x(self, y_):

        x_choices = self.x_lst[y_.item()]
        x_new = x_choices[random.randint(0, len(x_choices)-1)]

        return x_new


    def transition(self, a1,a2,y1,y2,c1,c2): 

        y1 = y1.cuda()
        y2 = y2.cuda()
        a1 = a1.cuda()
        a2 = a2.cuda()

        y1_ = torch.clamp(y1 + a1,0,9)
        y2_ = torch.clamp(y2 + a2,0,9)

        x1_ = self.transition_x(y1_)
        x2_ = self.transition_x(y2_)

        return x1_, x2_, y1_, y2_


