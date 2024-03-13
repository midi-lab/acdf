import torch
import statistics
import numpy as np
import torch
import torch.nn.functional as F
import os

from dijkstra_program import make_ls, update_ls
from scipy import stats as s


'''Stores transition matrix p(s' | s,a) in tabular form.  '''

class Transition:

    def __init__(self, args, ncodes, num_actions, env): 
        self.args = args
        self.ncodes = ncodes
        self.na = num_actions
        self.reset()
        self.decay_period = 2e5
        self.epsilon = 0.05
        self.env = env

        self.code2ground = {}

    def reset(self):

        self.state_transition = torch.zeros(self.ncodes,self.na,self.ncodes)
        
        self.ls, _ = make_ls(torch.zeros(self.ncodes,self.na,self.ncodes), self.ncodes, self.na)

        self.update_step = 1

        self.pair_lst = []
        self.steps_done = 0

        self.tr_lst = []
        self.trn_lst = []
        for j in range(0,self.ncodes):
            self.tr_lst.append([])
            self.trn_lst.append([])


    def update(self, ind_last, ind_new, a1, y1, y1_):


        for j in range(0, ind_last.shape[0]):
            self.state_transition[ind_last.flatten()[j], a1[j], ind_new.flatten()[j]] += 1

            self.pair_lst.append((y1[j], ind_last.flatten()[j], a1[j], y1_[j], ind_new.flatten()[j]))

            update_ls(self.ls, ind_last.flatten()[j].item(), a1[j].item(), ind_new.flatten()[j].item(), self.update_step)

            self.update_step += 1

        for j in range(0,self.ncodes):

            #print('types', ind_last.device, y1.device)

            self.tr_lst[j] += y1[ind_last.cpu().flatten()==j].data.cpu().numpy().tolist()
            self.trn_lst[j] += y1_[ind_new.cpu().flatten()==j].data.cpu().numpy().tolist()

        #for j in range(0,self.ncodes):
        #    self.tr_lst[j] += y1[ind_last.flatten()==j]
        #    self.trn_lst[j] += y1_[ind_new.flatten()==j]


    def save_log(self, logger, env_iter):

        if self.args.use_logger: 
            # fh = open( os.path.join(logger.save_folder, self.args.log_fh + '_' + str(env_iter)) + '.txt','w')
            fh = open( os.path.join(logger.save_folder, self.args.log_fh),'w')
        else:
            fh = open(self.args.log_fh, 'w')

        # ## Create directory to store log files
        # fh = open(self.args.log_fh, 'w')
        count = 0
        for (glast, llast, a, gnext, lnext) in self.pair_lst:
            count += 1
            fh.write("learned:L%d ground:G%d action:a%d next_learned:L%d next_ground:G%d uri:E%d\n" % (llast, glast, a, lnext, gnext, count))

        fh.close()


    def print_codes(self, init_state, next_state, action, goal): 

        ground2count = {}
        code2ground = {}

        for j in range(0,self.ncodes):

            if len(self.tr_lst[j]) > 0:
                # mode = statistics.mode(self.tr_lst[j])
                mode = s.mode(self.tr_lst[j])
                #mode = mode[0][0]
                mode = mode[0]

                ground_y = mode//(self.args.cols+1)
                ground_x = mode%(self.args.cols+1)

                #print('learned code count', j, (ground_x,ground_y), len(self.tr_lst[j]))

                if not (ground_x,ground_y) in ground2count:
                    ground2count[(ground_x, ground_y)] = 0

                ground2count[(ground_x, ground_y)] += len(self.tr_lst[j])
 
                code2ground[j] = (ground_x, ground_y)       
  

        for y in range(0,self.args.rows):
            for x in range(0,self.args.cols):
                if (x,y) in ground2count:
                    print(ground2count[(x,y)], end="\t")
                else:
                    print(0, end="\t") 
            print("")

        self.code2ground = code2ground
        

    def print_modes(self):

        mode_lst = []
        moden_lst = []
        for j in range(0,self.ncodes):
            if len(self.tr_lst[j]) == 0:
                mode_lst.append(-1)
            else:
                mode_lst.append(statistics.mode(self.tr_lst[j]))#torch.Tensor(tr_lst[j]).mode()[0])

            if len(self.trn_lst[j]) == 0:
                moden_lst.append(-1)
            else:
                moden_lst.append(statistics.mode(self.trn_lst[j]))#torch.Tensor(trn_lst[j]).mode()[0])

        corr = 0
        incorr = 0

        coverage = torch.zeros(10,self.na)

        print('state transition matrix!')
        for a in range(0,self.na):
            for k in range(0,self.state_transition.shape[0]):
                if self.state_transition[k,a].sum().item() > 0:
                    print(mode_lst[k], a-1, 'argmax', moden_lst[self.state_transition[k,a].argmax()], 'num', self.state_transition[k,a].sum().item())

                    num = self.state_transition[k,a].sum().item()
                    s1 = mode_lst[k]
                    s2 = moden_lst[self.state_transition[k,a].argmax()]
                    nex = s1 + (a-1)
                    nex = min(nex, 9)
                    nex = max(nex, 0)
                    coverage[nex,a] = 1.0
                    if nex == s2:
                        corr += num
                    else:
                        incorr += num


        print('transition acc', corr*1.0 / (corr+incorr))
        print('coverage', coverage.sum() * 1.0 / (10*self.na))   


    def _goal_reward(self):
        return torch.tensor([1])

    def select_goal(self):
        code_count = self.state_transition.sum(dim=(1,2))

        # print('code count', code_count)
        if self.args.rewardtype == "invsqrt":
            reward = torch.sqrt(1.0 / (code_count + 1.0)) #reward shape : 64 (for ncodes = nstates)
        elif self.args.rewardtype == "expsqrt":
            reward = torch.exp(-1.0 * torch.sqrt(code_count))
        elif self.args.rewardtype == "invexp":
            reward = torch.exp(-1.0 * code_count)
        elif self.args.rewardtype == 'invlog':
            reward = 1.0 / torch.log(code_count + 0.0001)
        elif self.args.rewardtype == 'envreward':
            reward = self.get_env_goal()
            

        else:
            raise Exception()

        return reward


    def get_env_goal(self):
        # sub_goal_size = 10 
        goal_reward = self.env.get_goal_reward()
        agent_pos = self.env.get_agent_position()
        reached_terminal = self.env.agent_reached_terminal(agent_pos)

        reward = goal_reward if reached_terminal else 1.0 
        reward = torch.tensor([reward]).repeat(self.args.ncodes)

        return reward


    def get_sub_goal(self):
        code_count = self.state_transition.sum(dim=(1,2)) + 1
        reward = torch.zeros(code_count.shape[0])
        for i in range(5): ## pick 5 sub-goals
            min_code_count = torch.argmin(code_count)
            code_count[min_code_count] = 500 ## any large value
            reward[min_code_count] = 1
        return reward



    def get_goal(self):
        code_count = self.state_transition.sum(dim=(1,2)) + 1
        min_code_count = torch.argmin(code_count)
        reward = torch.zeros(code_count.shape[0])
        reward[min_code_count] = 1
        return reward



    def get_random_goals(self, k):
        ## if k > 1, pick multiple random goals
        code_count = self.state_transition.sum(dim=(1,2)) 
        reward = torch.zeros(code_count.shape[0])

        sampled_goal = torch.randint(0, self.args.ncodes-1, (k, 1))  

        reward[sampled_goal] = 1

        return reward


    '''
        p(s' | a,s)
        max_a * p(s' | a,s)
    '''
    def select_policy(self, init_state, reward):

        eps = 0.0001
        counts = self.state_transition+eps
        probs = counts / counts.sum(dim=2, keepdim=True)

        best_action = -1
        best_value = -1

        for a in range(0,self.na):
            val = 0.0
            for sn in range(self.ncodes):
                val += reward[sn] * probs[init_state, a, sn]

            if val > best_value:
                best_value = val
                best_action = a


        best_action = torch.Tensor([best_action]).long()


        return best_action


    def random_policy(self):
        return torch.randint(0, self.na,size=(1,))

    def get_epsilon(self):
        alpha = self.steps_done / self.decay_period
        alpha = np.clip(alpha, 0, 1)
        return self.epsilon * alpha + 1 * (1 - alpha)


    def epsilon_greedy_act(self, init_state, reward):

        if np.random.uniform() < self.get_epsilon():
            best_action = np.random.randint(self.na)

        else:
            eps = 0.0001
            counts = self.state_transition+eps
            probs = counts / counts.sum(dim=2, keepdim=True)

            best_action = -1
            best_value = -1

            for a in range(0,self.na):
                val = 0.0
                for sn in range(self.ncodes):
                    val += reward[sn] * probs[init_state, a, sn]

                if val > best_value:
                    best_value = val
                    best_action = a


            best_action = torch.Tensor([best_action]).long()


        return best_action
