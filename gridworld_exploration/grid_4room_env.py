
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
import random
import numpy as np

from grid_4room_builder import GridWorld

class Env:

    def __init__(self, args, stochastic_start):

        self.args = args
        self.random_start = stochastic_start

        self.rows = args.rows
        self.cols = args.cols

        self.use_reset_action = args.reset_actions

        self.inp_size = self.rows*self.cols
        self.total_states = (args.rows)*(args.cols)

        self.num_exo = args.num_exo

        if self.args.exo_noise == 'two_maze':
            self.inp_size *= (self.num_exo + 1)

        self.num_actions = 6

        self.initial_state()
    def sample_initial_correct(self): # prevents sampling initial state in walls
        middle = int(self.rows / 2)
        quarter = int(self.rows / 4)
        pos = random.randint(1,  (self.rows-3)**2 + 4 ) # randint is inclusive
        if pos <= (self.rows-3)**2:
            pos_x = (pos-1)%(self.rows-3)
            pos_y =  (pos-1)//(self.rows-3)
            pos_x += 1
            if (pos_x >= middle):
                pos_x += 1
            pos_y += 1
            if (pos_y >= middle):
                pos_y += 1
            return (pos_x, pos_y)
        elif pos ==  (self.rows-3)**2 + 1:
            return (middle, quarter)
        elif pos ==  (self.rows-3)**2 + 2:
            return (quarter, middle)
        elif pos ==  (self.rows-3)**2 + 3:
            return (middle, self.rows-quarter -1)
        else:
            return (self.rows-quarter - 1, middle)

    def initial_state(self):
        #randind1 = random.randint(0,99)
        #randind2 = random.randint(0,99)


        exo_noise = self.args.exo_noise
        rows = self.rows
        cols = self.cols

        print("========================RESET GRID========================")

        #start_class = 9

        #x1 = torch.cat(self.x_lst[start_class], dim=0).unsqueeze(1)[randind1:randind1+1]
        #y1 = torch.zeros(1).long() + start_class

        #randclass = random.randint(0,9)
        #x2 = torch.cat(self.x_lst[randclass], dim=0).unsqueeze(1)[randind2:randind2+1]
        #y2 = torch.zeros(1).long() + randclass

        #x1 = x1.repeat(1,3,1,1)
        #x2 = x2.repeat(1,3,1,1)

        #c1 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0
        #c2 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0

        if self.random_start:
            # start1 = (random.randint(1,rows-1), random.randint(1,cols-1))
            start1 = self.sample_initial_correct()
        else:
            start1 = (1,1)

        #self.grid1 = GridWorld(7, start=start1, goal=(6,6))
        #self.grid2 = GridWorld(7, start=start2, goal=(6,6))

        self.grid1 = GridWorld(rows, cols, start=start1, goal=(rows-1,cols-1))

        self.exo_grids = []

        for j in range(self.num_exo):
            if self.random_start:
               starti = self.sample_initial_correct()
            else:
                starti = (1,1)
            self.exo_grids.append(GridWorld(rows, cols, start=starti, goal=(rows-1,cols-1)))

        y1 = self.grid1.agent_position
        y2 = self.exo_grids[0].agent_position

        x1 = self.grid1.img()

        x2lst = []
        
        for j in range(self.num_exo):
            x2lst.append((torch.Tensor(self.exo_grids[j].img()).float()).unsqueeze(0).unsqueeze(0))
        x2 = torch.cat(x2lst,dim=3)

        #c1 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0
        #c2 = (torch.rand(1,3,1,1).clamp(0.5,1.0)*10.0).round()/10.0

        y1 = torch.Tensor([y1[0]*(self.rows+1) + y1[1]]).long()
        y2 = torch.Tensor([y2[0]*(self.rows+1) + y2[1]]).long()
       

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

    def agent_reached_terminal(self, x):
        terminal = self.grid1.reached_terminal(x)

        return terminal


    def get_goal_reward(self):
        goal_reward = self.grid1.goal_reward
        return goal_reward

    def random_action(self, allow_reset=True):
        # action = torch.randint(0,4,size=(1,)).item()
        
        if self.use_reset_action and allow_reset:
            action = torch.randint(0,self.num_actions,size=(1,)).item()
        else:
            action = torch.randint(0,4,size=(1,)).item()
        

        return action

    def reset(self, stochastic_start=False):
        self.grid1.reset()

        for j in range(self.num_exo):
            self.exo_grids[j].reset()

    def step(self,a1,a2): 
        a1 = a1.item()
        #a2 = a2.item()

        #print(a1, a2)

        if a1 > 3: #reset action
            self.grid1.reset()
        else:
            self.grid1.step(a1)

        assert type(a1) is int
        assert type(a2) is int

        #self.grid2.reset(start=(random.randint(1,self.rows-1), random.randint(1,self.cols-1)))
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

        #x2 = self.grid2.img()

        y1_ = torch.Tensor([y1[0]*(self.rows+1) + y1[1]]).long()
        y2_ = torch.Tensor([y2[0]*(self.rows+1) + y2[1]]).long()

        x1_ = torch.Tensor(x1).float().unsqueeze(0).unsqueeze(0)
        x2_ = x2

        if self.args.exo_noise == 'exo_obs':
            #x1_ += self.c1 * 200.0
            #x2_ += self.c2 * 200.0
            x2_ = (torch.randn_like(x2_).round()+4)*0.1

        dum_r = 0
        dum_done = 0

        #print('just took action a1', a1, 'about to show grid')
        #self.grid1.show()
        #self.grid2.show()

        return y1_, y2_, x1_, x2_, dum_r, dum_done






