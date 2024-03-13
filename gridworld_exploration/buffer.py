import random
import torch
import numpy as np

class Buffer:

    '''
        Stores (a, y1, y1_, c1, y2, y2_, c2, x, x_) as matrices.  
    '''
    def __init__(self, args, ep_length, max_k):
        self.a = []
        self.y1 = []
        self.y1_ = []
        self.y2 = []
        self.y2_ = []
        self.x = []
        self.x_ = []
        self.slst = []
        self.sglst = []
        self.goal_steps = []
        self.steps = []
        self.valid_steps = []
        self.ep_length = ep_length
        self.max_k = max_k
        self.num_ex = 0
        self.args = args

        self.cutoff = 0

        self.goal_lst = []
        self.dp_depth_lst = []

        self.rand_lst = []

        self.s2valid = {}
        self.valid2s = {}

        self.pred_y1 = []
        self.pred_y1_ = []

    def add_example(self, a, y1, y1_,  y2, y2_, x, x_, s, s_g, goal_obs, steps_to_goal, took_ra, step, dp_depth, pred_y1, pred_y1_):


        if took_ra:
            self.valid_steps.append(self.num_ex)

            if not (s,a,dp_depth) in self.s2valid:
                self.s2valid[(s,a,dp_depth)] = []

            self.s2valid[(s,a,dp_depth)].append(self.num_ex)
            self.valid2s[self.num_ex] = (s,a,dp_depth)

        self.a.append(a)
        self.y1.append(y1)
        self.y1_.append(y1_)
        self.y2.append(y2)
        self.y2_.append(y2_)
        self.x.append(x)
        self.x_.append(x_)
        self.steps.append(step)

        self.rand_lst.append(took_ra)

        self.dp_depth_lst.append(dp_depth)

        self.goal_lst.append(goal_obs)

        self.slst.append(s)
        self.sglst.append(s_g)
        self.goal_steps.append(steps_to_goal)

        self.num_ex += 1

        self.pred_y1.append(pred_y1)
        self.pred_y1_.append(pred_y1_)
    '''
        Returns bs of (a, y1, y1_, x, x_)
    '''
    def sample_batch(self, bs, indlst=None, klim=None, only_train_random=True, return_immediate=False): 

        ba = []
        by1 = []
        by1_ = []
        bx = []
        bx_ = []
        if (return_immediate):
            bx_immediate = []
        bk = []
        bd = []
        bv = []

        bg = []

        bpred_y1 = []
        bpred_y1_ = []

        for k in range(0, bs):

            if only_train_random and indlst is None:
                srand = random.choice(list(self.s2valid.keys()))
                j1 = random.randint(0, len(self.s2valid[srand])-1)
                j = self.s2valid[srand][j1]

                #if j < self.cutoff:
                #    raise Exception('invalid', j, self.cutoff)

                #if self.num_ex > 1000 and j < (self.num_ex - 500):
                #    continue

                #j1 = random.randint(0, len(self.valid_steps)-1)
                #j = self.valid_steps[j1]
            elif indlst is None:
                j = random.randint(0, self.num_ex-2)
            else:
                j = indlst[k]

            if j >= self.num_ex - 2:
                continue

            a,y1,_,x,_,s,s_g,steps_to_goal, step, pred_y1, _ = self.sample_ex(j)


            maxk = min(self.max_k, self.ep_length - step)
            maxk = min(maxk, self.num_ex - j - 1)


            genik_goal_clamp = True
            if genik_goal_clamp:
                maxk = min(maxk, steps_to_goal)

            maxk = max(maxk, 1)

            if klim is not None:
                maxk = min(maxk, klim)

            randk = maxk#random.randint(1, maxk)
            if (self.args.policy_selection == 'random'):
                randk = random.randint(1, maxk)

            if randk >= 1:
                j_n = j + randk
                #for r in range(j+1, j_n):
                #    an, _, _, _, _, _, _, _, _ = self.sample_ex(r)
                #    alst.append(an)

                #_, _, y1_n, _, x_n, _, _, _, step_n = self.sample_ex(r)
                _, y1_n, _, x_n, _, _, _, _, step_n, pred_y1_n, _  = self.sample_ex(j_n)
                if (return_immediate):
                    _, _, _, x_imm, _, _, _, _, _, _, _  = self.sample_ex(j + 1)

                #print('step', step, 'step_n', step_n, 'num_ex', self.num_ex)
                #assert step_n > step

                if step >= step_n: # This should also cover x_immediate, which is between x and x_n
                    continue

                y1_ = y1_n
                x_ = x_n
                if (return_immediate):
                    x_immediate = x_imm

                pred_y1_ = pred_y1_n

            try:

                s_goalstep = self.slst[j + steps_to_goal]
                obs_goal = self.x[j + steps_to_goal]

                if s_g == s_goalstep:
                    is_valid = 1.0
                else:
                    is_valid = 0.0

                    #print('found in cache (s,a)', self.valid2s[j], 's should be', self.slst[j])

                    #cachekey = self.valid2s[j]
                    #self.s2valid.pop(cachekey)

                if False and s_g == s_goalstep and ((random.uniform(0,1) < 0.01 and self.dp_depth_lst[j] > 0) or self.dp_depth_lst[j] > 1):
                    print('==========================================================================================================================')
                    print('in buffer did we reach goal?', 'claim goal', s_g, 'goal_reach', s_goalstep, 'in steps', steps_to_goal, 'plan-dpdepth', self.dp_depth_lst[j])
                    print('\t', 'action', a, 'num_steps from start is goal selected', randk)
                    print(x)
                    print('')
                    print(obs_goal)

            except:
                is_valid = 1.0



            #print(a.shape, y1.shape, x.shape)

            if False and random.uniform(0,1) < 0.001:
                print('-----------------------------------------------------------------------------')
                print(x)
                print('action', a, 'offset', randk)
                print(x_)

            if torch.cuda.is_available():
                bd.append(torch.Tensor([self.dp_depth_lst[j]]).cuda())
                ba.append(torch.Tensor([a]).long().cuda())
                by1.append(y1.cuda())
                by1_.append(y1_.cuda())
                bx.append(x.cuda())
                bx_.append(x_.cuda())
                if (return_immediate):
                    bx_immediate.append(x_immediate.cuda())
                bk.append(torch.Tensor([randk]).long().cuda())
                bv.append(torch.Tensor([is_valid]).cuda())
                bg.append(self.goal_lst[j].cuda()) #obs on step where we expect to reach goal.

                bpred_y1.append(pred_y1.cuda())
                bpred_y1_.append(pred_y1_.cuda())

            else:
                ba.append(a)
                by1.append(y1)
                by1_.append(y1_)
                bx.append(x)
                bx_.append(x_)
                if (return_immediate):
                    bx_immediate.append(x_immediate)
                bk.append(torch.Tensor([randk]).long())
                bv.append(torch.Tensor([is_valid]))
                bg.append(self.goal_lst[j]) #obs on step where we expect to reach goal.
                #bx_g.append(x_g) #obs on step where we expect to reach goal.      
                bd.append(torch.Tensor([self.dp_depth_lst[j]]))       

                bpred_y1.append(pred_y1)
                bpred_y1_.append(pred_y1_)   

        if torch.cuda.is_available():
            ba = torch.cat(ba, dim=0).cuda()
            by1 = torch.cat(by1, dim=0).cuda()
            by1_ = torch.cat(by1_, dim=0).cuda()
            bx = torch.cat(bx, dim=0).cuda()
            bx_ = torch.cat(bx_, dim=0).cuda()
            if (return_immediate):
                bx_immediate = torch.cat(bx_immediate, dim=0).cuda()
            bk = torch.cat(bk, dim=0).cuda()
            bv = torch.cat(bv, dim=0).cuda()
            bd = torch.cat(bd, dim=0).cuda()
            bg = torch.cat(bg, dim=0).cuda()

            bpred_y1 = torch.cat(bpred_y1, dim=0).cuda()
            bpred_y1_ = torch.cat(bpred_y1_, dim=0).cuda()

        else:
            ba = torch.cat(ba, dim=0)
            by1 = torch.cat(by1, dim=0)
            by1_ = torch.cat(by1_, dim=0)
            bx = torch.cat(bx, dim=0)
            bx_ = torch.cat(bx_, dim=0)
            if (return_immediate):
                bx_immediate = torch.cat(bx_immediate, dim=0)
            bk = torch.cat(bk, dim=0)
            bv = torch.cat(bv, dim=0)
            bd = torch.cat(bd, dim=0)
            bg = torch.cat(bg, dim=0)

            bpred_y1 = torch.cat(bpred_y1, dim=0)
            bpred_y1_ = torch.cat(bpred_y1_, dim=0)

            #bx_g = torch.cat(bx_g, dim=0)
        if (return_immediate):
            return ba, by1, by1_, bx, bx_, bv, bk, bd, bg, bpred_y1, bpred_y1_, bx_immediate
        else:
            return ba, by1, by1_, bx, bx_, bv, bk, bd, bg, bpred_y1, bpred_y1_

    def sample_dist(self):
 
        x_b = []
        xk_b = []
        k_b = []

        for i in range(0,100):
            j = random.randint(0, self.num_ex-1)
       
            a,y1,y1_,x,x_,s,s_g,steps_to_goal,step, _, _ = self.sample_ex(j)
            klim = min(j + steps_to_goal, self.num_ex-1)
            randk = random.randint(j, klim)

            _,_,_,xk,_,_,_,_,_, _, _ = self.sample_ex(randk)

            x_b.append(x)
            xk_b.append(xk)
            k_b.append(torch.Tensor([randk-j]))

        #print('sample offset size randk', randk)

        # print(len(x_b), len(xk_b), len(k_b))

        if torch.cuda.is_available():
            x_b = torch.cat(x_b, dim=0).cuda()
            xk_b = torch.cat(xk_b, dim=0).cuda()
            k_b = torch.cat(k_b, dim=0).cuda().long()
        else:
            x_b = torch.cat(x_b, dim=0)
            xk_b = torch.cat(xk_b, dim=0)
            k_b = torch.cat(k_b, dim=0).long()


        return x_b, xk_b, k_b

    def sample_byol(self):

        x_b = []
        xn_b = []
        a_b = []

        for i in range(0,100):
            j = random.randint(0, self.num_ex-2)

            a,y1,y1_,x,x_,s,s_g,steps_to_goal,step, _, _ = self.sample_ex(j)

            _,_,_,xn,_,_,_,_,_, _, _= self.sample_ex(j+1)

            x_b.append(x)
            xn_b.append(xn)
            a_b.append(torch.Tensor([a]))

        #print('sample offset size randk', randk)

        if torch.cuda.is_available():        
            x_b = torch.cat(x_b, dim=0).cuda()
            xn_b = torch.cat(xn_b, dim=0).cuda()
            a_b = torch.cat(a_b, dim=0).cuda().long()
        else:
            x_b = torch.cat(x_b, dim=0)
            xn_b = torch.cat(xn_b, dim=0)
            a_b = torch.cat(a_b, dim=0).long()            

        return x_b, xn_b, a_b


    def sample_classifier(self):
        gt_last = []
        gt_next = []
        a_b = []
        x_b = []
        xn_b = []


        pred_gt_last = []
        pred_gt_next = []

        for i in range(0,100):
            j = random.randint(0, self.num_ex-2)
            # a,y1,_,x,_,s,s_g,steps_to_goal, step, pred_y1, _ = self.sample_ex(j)
            # a,y1,x,x_,s,s_g,steps_to_goal,step, pred_y1, pred_y1_ = self.sample_ex(j)

            a,y1,y1_,x,_,s,s_g,steps_to_goal,step, pred_y1, _ = self.sample_ex(j)
            _,_,y1_,xn,_,_,_,_,_, _, pred_y1_= self.sample_ex(j+1)

            gt_last.append(y1)
            gt_next.append(y1_)
            pred_gt_last.append(pred_y1)
            pred_gt_next.append(pred_y1_)
            a_b.append(torch.Tensor([a]))

            x_b.append(x)
            xn_b.append(xn)

        if torch.cuda.is_available():  
            gt_last = torch.cat(gt_last, dim=0).cuda()
            gt_next = torch.cat(gt_next, dim=0).cuda()
            pred_gt_last = torch.cat(pred_gt_last, dim=0).cuda()
            pred_gt_next = torch.cat(pred_gt_next, dim=0).cuda()
            a_b = torch.cat(a_b, dim=0).cuda().long()
            x_b = torch.cat(x_b, dim=0).cuda()
            xn_b = torch.cat(xn_b, dim=0).cuda()

        else:
            gt_last = torch.cat(gt_last, dim=0)
            gt_next = torch.cat(gt_next, dim=0)
            pred_gt_last = torch.cat(pred_gt_last, dim=0)
            pred_gt_next = torch.cat(pred_gt_next, dim=0)
            a_b = torch.cat(a_b, dim=0).long()
            x_b = torch.cat(x_b, dim=0)
            xn_b = torch.cat(xn_b, dim=0)


        return gt_last, gt_next, pred_gt_last, pred_gt_next, a_b, x_b, xn_b


    def set_cutoff(self, cutoff):
        self.cutoff = cutoff

        new_s2valid = {}

        for key in list(self.s2valid.keys()):
            val_lst = self.s2valid[key]
            
            new_val_lst = []

            for val_ind in range(len(val_lst)):
                val = val_lst[val_ind]
                if val > cutoff:
                    new_val_lst.append(val)
                
            if len(new_val_lst) > 0:
                new_s2valid[key] = new_val_lst

        self.s2valid = new_s2valid


    def sample_ex(self, j): 

        return (self.a[j], self.y1[j], self.y1_[j], self.x[j], self.x_[j], self.slst[j], self.sglst[j], self.goal_steps[j], self.steps[j], self.pred_y1[j], self.pred_y1_[j])

