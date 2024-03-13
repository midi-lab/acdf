
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
import random
import numpy as np

from periodic_cart_builder import PeriodicCartBuilder

class PeriodicCartEnv:

    def __init__(self, args, stochastic_start):

        self.args = args
        self.random_start = stochastic_start


        self.half_period = args.half_period 
        self.side_length = 2 * self.half_period
        self.total_states = self.side_length * 4

        self.inp_size = (2 * self.half_period+1)**2
        assert (2 * self.half_period+1 == args.rows)
        assert (2 * self.half_period+1 == args.cols)

        if (args.reset_actions):
            print("args.reset_actions is set")
            print(args.reset_actions)
            assert 1 ==2
        self.num_exo = args.num_exo

        self.num_actions = 2

        self.initial_state()

    def initial_state(self):

        exo_noise = self.args.exo_noise


        print("========================RESET GRID========================")


        if self.random_start:
            start1 = random.randrange(0,self.total_states)
        else:
            start1 = 0



        self.grid1 = PeriodicCartBuilder(self.half_period,start1)

        self.exo_grids = []

        for j in range(self.num_exo):
            if self.random_start:
                starti = random.randrange(0,self.total_states)
            else:
                starti = 0
            self.exo_grids.append(PeriodicCartBuilder(self.half_period,starti))

        y1 = self.grid1.agent_position
        y2 = self.exo_grids[0].agent_position

        x1 = self.grid1.img()

        x2lst = []
        
        for j in range(self.num_exo):
            x2lst.append((torch.Tensor(self.exo_grids[j].img()).float()).unsqueeze(0).unsqueeze(0))
        x2 = torch.cat(x2lst,dim=3)


        y1 = torch.Tensor([y1]).long()
        y2 = torch.Tensor([y2]).long()
       

        x1 = (torch.Tensor(x1).float()).unsqueeze(0).unsqueeze(0)

        c1 = torch.randn_like(x1) * 0.01
        c2 = torch.randn_like(x2) * 0.01
        self.c1 = c1
        self.c2 = c2

        if exo_noise == 'exo_obs':
            x1 += c1
            x2 += c2

        dummy_x2 = 0
        dummy_y2 = 0
        dummy_c2 = 0

        if exo_noise == 'two_maze':
            return y1,c1,y2,c2,x1,x2
        else:
            return y1, c1, dummy_y2, dummy_c2, x1, dummy_x2

    def get_obs(self):
        x = self.grid1.img()
        x = torch.Tensor(x).float().unsqueeze(0).unsqueeze(0)

        return x

    def get_agent_position(self):
        x = self.grid1.agent_position
        return x 


    def random_action(self, allow_reset=True):

        action = torch.randint(0,2,size=(1,)).item()
        return action

    def reset(self, stochastic_start=False):
        self.grid1.reset()

        for j in range(self.num_exo):
            self.exo_grids[j].reset()

    def step(self,a1,a2): 
        a1 = a1.item()

        self.grid1.step(a1)

        assert type(a1) is int
        assert type(a2) is int

        for j in range(self.num_exo):
            a2 = self.random_action(allow_reset=False)
            self.exo_grids[j].step(a2)

        y1 = self.grid1.agent_position
        y2 = self.exo_grids[0].agent_position

        x1 = self.grid1.img()

        x2lst = []
        for j in range(self.num_exo):
            x2lst.append((torch.Tensor(self.exo_grids[j].img()).float()).unsqueeze(0).unsqueeze(0))        

        x2 = torch.cat(x2lst,dim=3)


        y1_ = torch.Tensor([y1]).long()
        y2_ = torch.Tensor([y2]).long()

        x1_ = torch.Tensor(x1).float().unsqueeze(0).unsqueeze(0)
        x2_ = x2

        if self.args.exo_noise == 'exo_obs':

            x2_ = (torch.randn_like(x2_).round()+4)*0.1

        dum_r = 0
        dum_done = 0


        return y1_, y2_, x1_, x2_, dum_r, dum_done

