import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
import random
import numpy as np
import time
import pickle
import statistics
from buffer import Buffer
from transition import Transition
from value_iteration import value_iteration, pre_train_policy, get_action_one_step_lookahead
import argparse
from grid_4room_env import Env
import copy
from encoders.mlp_pred1 import Classifier
from encoders.dist_pred import DistPred
from eval_metric import get_dsm_sss_errors
from utils import states_pos_to_int, Logger
from torchvision.utils import save_image
from utils import plot_state_visitations

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from encoders.mlp_linear_rep_classifier import LinearClassifier

parser = argparse.ArgumentParser(description='Maze Task', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', type=str, choices=[ 'mnist', 'maze', 'periodic-cart','vis-maze', 'minigrid', 'cartpole'])

parser.add_argument('--obs_type', type=str, default='high_dim', choices=['pixels', 'high_dim'],
                    help='Which type of observation to use for SpiralWorld-like envs')

parser.add_argument('--exo_noise', type=str, default='multi_maze', choices=['two_maze', 'exo_obs', 'multi_maze', 'none'],
                    help='Which type of exogenous noise to use')

parser.add_argument('--num_iter', type=int, default=500) #2000

parser.add_argument('--model_train_iter', type=int, default=100)

parser.add_argument('--num_exo', type=int, default=8)

parser.add_argument('--batch_size', type=int, default=64) #2000

parser.add_argument('--do_quantize_iter', type=int, default=500) #2000

parser.add_argument('--num_rand_initial', type=int, default=1) #2000

parser.add_argument('--value_iter', type=int, default=5) #iteration steps for VI algo for policy selection

parser.add_argument('--random_start', type=str, choices=('true','false'), default='false')

parser.add_argument('--random_policy', type=str, choices=('true', 'false'), default='false')

parser.add_argument('--max_ep_length', type=int, default=200)

parser.add_argument('--use_ae', type=str, choices=('true', 'false'), default='false')

parser.add_argument('--ep_length', type=int, default=200)

parser.add_argument('--predict_action_sequence', type=bool, default=False) #if true, prediction sequence from a[t] to a[t+k].  Otherwise only predict a[t].  

parser.add_argument('--ncodes', type=int, default=120)

parser.add_argument('--env_iteration', type=int, default=200000)

parser.add_argument('--policy_selection', type=str, default = 'DP-Goal', choices=['greedy', 'random',  'VI', 'VI-Goal', 'VI-Sub-Goal', 'random-sub-goals', 'DP-Goal'])

# arguments for visgrid mazes
parser.add_argument('--walls', type=str, default='spiral', choices=['empty', 'mazeworld', 'spiral', 'loop'],
                    help='The wall configuration mode of gridworld')

parser.add_argument('--rewardtype', type=str, default='invsqrt', choices=['invsqrt', 'expsqrt', 'invexp', 'envreward'])

parser.add_argument('-r','--rows', type=int, default=11, help='Number of gridworld rows')

parser.add_argument('-c','--cols', type=int, default=11, help='Number of gridworld columns')

parser.add_argument('--half_period', type=int, default=5, help='periodicity')

parser.add_argument("--noise_type", type=str, default=None, choices=[None, 'ising', 'ellipse', 'tv'], help='Exo noise to observations')

parser.add_argument("--noise_stationarity", type=str, default='stationary', choices=['non-stationary', 'stationary'], help='resample noise every step?')

parser.add_argument("--log_fh", type=str, default='log.txt')

parser.add_argument("--use_logger", action="store_true", default=False, help='whether to use logging or not')

parser.add_argument("--folder", type=str, default='./results/')

parser.add_argument("--use_pretrained_policy", action="store_true", default=False, help='whether or not to use a pre-trained policy for data collection')

parser.add_argument("--use_forward", action="store_true", default=False, help='whether to use a forward dynamics loss')

parser.add_argument('--forward_loss_freq', type=int, default=5) 

parser.add_argument("--use_best_model", action="store_true", default=False, help='assess performance of best model from moving average loss')

parser.add_argument("--use_gt", type=bool, default=False)

parser.add_argument("--stochastic_start", type=str, default='deterministic', choices=['stochastic', 'deterministic'], help='whether or not to start from stochastic random state')

parser.add_argument("--no_restart", type=str, default='true', choices=['true', 'false'], help='zero environment restarts')

parser.add_argument('--objective', type=str, choices=[ 'genik', 'genik_random_only'], default='genik')

# Noise-specific arguments
parser.add_argument('--ising_beta', type=float, default=0.5,
                    help='Ising model\'s beta parameter')

parser.add_argument('--epsilon', type=float, default=0.2, help='Epsilon parameter for epsilon greedy')

parser.add_argument('--eval_dsm_ss', action='store_true',
                    help='To evaluate dsm and sss erorrs')

parser.add_argument('--k_steps', type=int, default=199) #arg for the k-step of GenIK

parser.add_argument('--tag', type=str, default='test', required=False, help='Tag for identifying experiment')

parser.add_argument("--use_discrete_codes", type=bool, default=False, help="whether or not to use the discrete codes for the linear classifier")

parser.add_argument('--log_every', type=int, default=100)

parser.add_argument('--seed', type=int, default=0)

parser.add_argument('--log_eval_prefix', type=str)

parser.add_argument('--eval_iter', type=int, default=1000) # Should be multiple of model_train_iter

parser.add_argument('--max_kl_penalty', type=float, default=0.0001)

parser.add_argument('--reset_actions', type=bool, default=True)
parser.add_argument('--no_reset_actions', action='store_true')


parser.add_argument("--heatmap_random", type=str, default='false', choices=['true', 'false'], help='purely random policy for exploration')



args = parser.parse_args()

random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

if (args.no_reset_actions):
    args.reset_actions = False
print('args', args)

if args.stochastic_start == 'deterministic':
    stochastic_start = False
elif args.stochastic_start == 'stochastic':
    stochastic_start = True
else:
    raise Exception()

if args.no_restart == 'true':
    args.no_restart = True
elif args.no_restart == 'false':
    args.no_restart = False

if args.data == 'mnist':
    #% ------------------ Define MDP as the mnist environment------------------
    from env import Env
    myenv = Env(random_start=(args.random_start=='true'))
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

elif args.data == 'maze':
    #% ------------------ Define MDP as a Simple Four Rooms Maze ------------------
    myenv = Env(args, stochastic_start=stochastic_start)
    myenv_eval = Env(args, stochastic_start=stochastic_start)
    env_name = 'maze'

elif args.data == 'periodic-cart':
    #% ------------------ Define MDP as Periodic Track ------------------
    from periodic_cart_env import PeriodicCartEnv
    myenv = PeriodicCartEnv(args, stochastic_start=stochastic_start)
    myenv_eval = PeriodicCartEnv(args, stochastic_start=stochastic_start)
    env_name = 'periodic-cart'

elif args.data == 'vis-maze':

    from visgrid.gridworld import GridWorld, TestWorld, SnakeWorld, RingWorld, MazeWorld, SpiralWorld, LoopWorld
    from visgrid.utils import get_parser, MI
    from visgrid.sensors import *
    from visgrid.gridworld.distance_oracle import DistanceOracle

    #% ------------------ Define MDP from VisGrid Mazes ------------------
    if args.walls == 'mazeworld':
        myenv = MazeWorld.load_maze(args=args)
        env_name = 'mazeworld'
    elif args.walls == 'spiral':
        myenv = SpiralWorld(rows=args.rows, cols=args.cols, noise_type=args.noise_type, ising_beta=args.ising_beta, num_exo=args.num_exo, exo_noise=args.exo_noise)
        env_name = 'spiralworld'
    elif args.walls == 'loop':
        myenv = LoopWorld(rows=args.rows, cols=args.cols, noise_type=args.noise_type, ising_beta=args.ising_beta, num_exo=args.num_exo, exo_noise=args.exo_noise)
        env_name = 'loopworld'
    else:
        myenv = GridWorld(rows=args.rows, cols=args.cols, noise_type=args.noise_type, ising_beta=args.ising_beta, num_exo=args.num_exo, exo_noise=args.exo_noise)
        env_name = 'gridworld'

    if args.exo_noise == 'multi_maze':
        myenv_list = []
        for j in range(args.num_exo + 1):
            ###myenv = GridWorld(rows=args.rows, cols=args.cols, noise_type=args.noise_type, ising_beta=args.ising_beta, num_exo=args.num_exo, exo_noise=args.exo_noise)
            # myenv_list.append(myenv)
            myenv_list.append(copy.deepcopy(myenv))


    # if args.exo_noise == 'multi_maze':
    #     myenv_list = [myenv]
    #     all_exo_mazes = [GridWorld, LoopWorld, MazeWorld, SpiralWorld, GridWorld, LoopWorld, SpiralWorld]
    #     for j in range(args.num_exo):
    #         ind = random.randint(0, len(all_exo_mazes)-1)
    #         sampled_env = all_exo_mazes[ind]
    #         myenv_exo = sampled_env(rows=args.rows, cols=args.cols, noise_type=args.noise_type, ising_beta=args.ising_beta, num_exo=args.num_exo, exo_noise=args.exo_noise)
    #         myenv_list.append(myenv_exo)
            # myenv_list.append(copy.deepcopy(myenv))





elif args.data == 'cartpole':
    import gym
    from cartpole.cartpole import CartPoleEnv
    from PIL import Image

    import torchvision.transforms as T
    GRAYSCALE = True ### FALSE IS RGB
    FRAMES = 2
    RESIZE_PIXELS = 60

    if GRAYSCALE == 0:
        resize = T.Compose([T.ToPILImage(), 
                        T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                        T.ToTensor()])
        
        nn_inputs = 3*FRAMES  # number of channels for the nn

    else:
        resize = T.Compose([T.ToPILImage(),
                T.Resize(RESIZE_PIXELS, interpolation=Image.CUBIC),
                T.Grayscale(),
                T.ToTensor()])
        nn_inputs =  FRAMES # number of channels for the nn

    env = gym.make("CartPole-v0").unwrapped    
    # myenv = gym.make("CartPole-v0")
    env_name = "cartpole"
    env.num_actions = env.action_space.n
    myenv = CartPoleEnv(env=env, rows=args.rows, cols=args.cols)

else:
    raise Exception()


num_rand = args.num_rand_initial

always_random = False
if args.policy_selection == 'random':
    always_random = True

if args.policy_selection == 'DP-Goal':
    #from dynamic_program import DP, DP_counts
    from dijkstra_program import DP_counts
    from dijkstra_goal import DP_goals, select_random_reachable_goal

#% ------------------ LOGGERS ------------------
if args.use_logger:
    logger = Logger(args, experiment_name=args.tag, environment_name=env_name, exo_noise = args.exo_noise , grid_size = 'rows_' + str(args.rows) + '_cols_' + str(args.cols), use_policy=args.policy_selection, start_state=str(args.stochastic_start), reset='no_restart_' + str(args.no_restart), folder=args.folder)
    logger.save_args(args)
    print('Saving to', logger.save_folder)
else:
    logger = None

ep_length = args.ep_length
ep_rand = ep_length

ncodes = args.ncodes
genik_maxk = args.k_steps

goal_ep = 0

if args.data == 'vis-maze' and args.obs_type == 'high_dim':
    sensor = myenv.sensor_observation()

def init_model(exo_noise):

    if exo_noise == 'none':
        if args.data == 'maze' or args.data == 'periodic-cart':
            x1 = myenv.get_obs()
            net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=myenv.inp_size, inp_dim=x1.shape[2])
        
        elif args.data == 'cartpole':
            env.reset()
            if GRAYSCALE == 0:
                x1 = myenv.get_screen(resize)
            else:
                x1 = myenv.get_screen(resize)
            net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3], inp_dim=x1.shape[2])

        else:
            raise Exception()

    elif exo_noise == 'two_maze': ## two-maze exo-noise
        if args.data == 'mnist':
            net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=env.inp_size*2, inp_dim=x1.shape[2])

        elif args.data=='maze' or args.data == 'periodic-cart':
            x1 = myenv.get_obs()
            net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3] * (myenv.num_exo+1), inp_dim=x1.shape[2])

        elif args.data == 'vis-maze':
            s1 = myenv.get_state()
            if args.obs_type == 'high_dim':
                x1 = myenv.get_sensor_observe(s1, sensor)
                net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3] * 2, inp_dim=x1.shape[2])
            else:
                x1 = myenv.get_image(s1)
                net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3] * 2 * 3, inp_dim=x1.shape[2])

    elif exo_noise == 'multi_maze':
        if args.data == 'vis-maze':
            s1 = myenv.get_state()
            if args.obs_type == 'high_dim':
                x1 = myenv.get_sensor_observe(s1, sensor)                
                # img = x1.squeeze(0).squeeze(0)
                # img=img.unsqueeze(0)
                # save_image(img, 'multi_maze.png')
                net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3] * (myenv.num_exo+1), inp_dim=x1.shape[2])
            else: ## for pixel based observations on spiralmazes
                x1 = myenv.get_image(s1)
                # img = x1.squeeze(0).squeeze(0)
                # img=img.unsqueeze(0)
                # save_image(img,  str(args.walls) + '_multi_maze.png')
                # net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3] * myenv.num_exo+1 * 3, inp_dim=x1.shape[2])
                net = Classifier(args, ncodes=ncodes, maxk=genik_maxk, inp_size=x1.shape[2] * x1.shape[3] * myenv.num_exo+1 * 1, inp_dim=x1.shape[2])

    else:
        raise Exception('Model not initialized for type of exo noise')

    if torch.cuda.is_available():
        net = net.cuda()

    return net




def initi_linear_classifier(exo_noise):
    if args.data == 'vis-maze':
        s1 = myenv.get_state()
        if args.obs_type == 'high_dim':
            x1 = myenv.get_sensor_observe(s1, sensor)                
            l_net = LinearClassifier(args, ncodes=ncodes, inp_size=x1.shape[2] * x1.shape[3], num_states= (args.rows+1) * args.cols, use_discrete_codes=args.use_discrete_codes, embed_dim=512)
        else:
            x1 = myenv.get_image(s1)
            l_net = LinearClassifier(args, ncodes=ncodes, inp_size=x1.shape[2] * x1.shape[3] * 2 * 3, num_states= (args.rows+1) * args.cols, use_discrete_codes=args.use_discrete_codes, embed_dim=512)
    else:
        raise Exception('Linear Representation Classifier not defined')
    if torch.cuda.is_available():
        l_net = l_net.cuda()
    return l_net



def init_opt(net):
    opt = torch.optim.AdamW(net.parameters(), lr=0.0001, betas=(0.9,0.999), weight_decay=0)
    return opt

def init_opt_classifier(l_net):
    l_opt = torch.optim.Adam(l_net.parameters(), lr=0.0001, betas=(0.9,0.999))
    return l_opt


net = init_model(args.exo_noise)
losses= []
update_count = 0
best_avg_loss = float('inf')
if args.data == 'vis-maze':
    l_net = initi_linear_classifier(args.exo_noise)
    l_opt = init_opt_classifier(l_net)


opt = init_opt(net)


if torch.cuda.is_available():
    distpred = DistPred(maxk = 64, inp_size = 512).cuda()
else:
    distpred = DistPred(maxk = 64, inp_size = 512)

def update_model(model, lin_model,  mybuffer, print_, do_quantize, reinit_codebook,bs, batch_ind=None, klim=None): 

    if (args.use_forward):
        a1, y1, y1_, x_last, x_new, valid_score, k_offset, depth_to_goal, goal_obs_x, embed_pred_y1, embed_pred_y1_,x_imm = mybuffer.sample_batch(bs, batch_ind, klim=klim, return_immediate=True)
    else:
        a1, y1, y1_, x_last, x_new, valid_score, k_offset, depth_to_goal, goal_obs_x, embed_pred_y1, embed_pred_y1_ = mybuffer.sample_batch(bs, batch_ind, klim=klim)

    loss = 0.0

    for k_ind in [0]:
        xl_use = x_last*1.0
        xn_use = x_new*1.0
        out, q_loss, ind_last, ind_new, z1, z2 = model(xl_use, xn_use, do_quantize = do_quantize, reinit_codebook = reinit_codebook, k=k_ind, k_offset=k_offset)

        if (args.use_forward):
            with torch.no_grad():
                _,_,ind_imm = model.enc(x_imm.detach(), do_quantize, reinit_codebook,k=k_ind)
         # this linear classifier loss is trained only only on the random actions; or mostly on the actions genik is trained to predict; 
         # for a general classifier, we should instead train it to predict all actions, instead of random actions
        # lin_model_loss = lin_model(y1, y1_, embed_pred_y1, embed_pred_y1_)
        lin_model_loss = 0


        if torch.cuda.is_available():
            k_offset = k_offset.cuda()
        else:
            k_offset = k_offset

        kpred = distpred.predict_k(z1.detach(), z2.detach())
        valid_score_dist = torch.le(k_offset, kpred).float()


        dtg = torch.eq(depth_to_goal, 0).float()
        valid_score_dist = torch.maximum(valid_score_dist, dtg)

        # valid_score_goal = torch.eq(z_goal_ind.flatten(), ind_new.flatten()).float()
        # valid_score_goal_dt = torch.maximum(valid_score_goal, dtg)

        if True and random.uniform(0,1) < 0.01:
            print('rate reach goal (should be 25%)', valid_score_dist[depth_to_goal > 1].mean())

        # if False and random.uniform(0,1) < 0.01:
        #     print('==================================================================================')
        #     print('pred, offset', 'dtg,', 'actual <= pred,', '(actual <= pred), or dtg==0')
        #     print(kpred[0])
        #     print(k_offset[0])
        #     print(depth_to_goal.long()[0])
        #     print(torch.le(k_offset.cuda(), kpred).float().long()[0])
        #     print(valid_score_dist.long()[0])
            
        # if True and random.uniform(0,1) < 0.01:
        #     print('==================================================================================')
        #     print(xl_use[0])
        #     print('action', a1[0])
        #     print(xn_use[0])

        #a1_onehot = F.one_hot(a1, 10)
        pred_j = out


        if (args.use_forward):
            act = torch.nn.functional.one_hot(a1.detach(), num_classes=10)
            context = torch.cat((z1, act), -1)
            global update_count
            if update_count % args.forward_loss_freq == 0:
                out_2 = model.out_fwd(context)
            else:
                out_2 = model.out_fwd(context.detach())
            loss_fwd = ce(out_2,ind_imm.detach().flatten())
            loss += (loss_fwd).mean()
            update_count += 1
        loss_ce = ce(pred_j, a1)

        # if random.uniform(0,1) < 0.01:
            # print('depth-to-goal', depth_to_goal.long())
            # print('valid score dist (dtg>0)', torch.le(k_offset[depth_to_goal > 0].cuda(), kpred[depth_to_goal > 0]).float().mean())
        #    print('z-goal-score (25%)', valid_score_goal[depth_to_goal > 0].mean())
        #    print('goal-ind', z_goal_ind.flatten())
        #    print('obs-ind', ind_new.flatten())


            #print('valid-score-goal', valid_score_goal)
            #print('depth to goal', depth_to_goal)
            #print('dtg==0', dtg)
            #print('valid-score-goal-dt', valid_score_goal_dt)

        if True or args.policy_selection == "DP-Goal":
            loss += (loss_ce).mean()
        else:
            loss += loss_ce.mean()

        loss += q_loss

    if do_quantize:
        ind_last = ind_last.flatten()
        ind_new = ind_new.flatten()

    return out, loss, ind_last, ind_new, a1, y1, y1_, x_last, k_offset, lin_model_loss


ce = nn.CrossEntropyLoss(reduction='none')


mybuffer = Buffer(args, ep_length=ep_length, max_k=genik_maxk)
transition = Transition(args, ncodes, myenv.num_actions, myenv)
transition.reset()


if args.no_restart : 
    stochastic_start = False ##If no restart, always start from deterministic start state
    myenv.reset(stochastic_start=stochastic_start) # initialize agent deterministically from a start state

is_initial = True 
step = 0
reinit_code = False
Vlast = None
min_visit_count = 0

dp_depth = 0
steps_to_goal = 0
g_dp = -1
took_trainable = True

ls2obs = {} #for each learned state store a list of observations.  

net.eval()


if args.use_pretrained_policy : 
    args.value_iter = value_iter = 25
    a_pre_trained, Vlast_pretrained = pre_train_policy(args, myenv, net, transition, value_iter)


all_classifier_accuracy = []

state_visits = np.zeros(args.rows * args.cols)
all_states_count = np.zeros((args.env_iteration, args.rows * args.cols))

log_data = dict()

if args.heatmap_random == 'true':
    purely_random = True
elif args.heatmap_random == 'false':
    purely_random = False

all_state_visits = []


for env_iteration in range(0, args.env_iteration):

    if env_iteration < 500 : 
        net.enc.kl_penalty = 0.0
    else:
        net.enc.kl_penalty = args.max_kl_penalty

    #do one step in env.  
    if step == ep_length:
        if args.no_restart : ### if no_resart - zero env resets even after episode ends
            is_initial = False
            step += 1
        else: 
            step = 0
            is_initial=True
        #ep_rand = random.randint(ep_length//2, ep_length) #after this step in episode follow random policy
    else:
        step += 1

    if is_initial:

        if args.data == 'vis-maze':
            if args.no_restart :
                stochastic_start = False ## can be True; does not matter; dummy variable
                ## do not reset env; started from deterministic start state
            else:
                myenv.reset(stochastic_start=stochastic_start) 
        else:
            myenv.reset()

        if args.data == 'mnist':
            y1,c1,y2,c2,x1,x2 = myenv.initial_state()
            x = torch.cat([x1*c1,x2*c2], dim=3)

        elif args.data == 'maze' or 'periodic-cart':
            y1,c1,y2,c2,x1,x2 = myenv.initial_state()
            if args.exo_noise == 'two_maze':
                x = torch.cat([x1,x2], dim=3)
            elif args.exo_noise == 'exo_obs':
                x = x1+c1
            else:
                x = x1*1.0

        elif args.data == 'vis-maze': 
            if args.exo_noise == 'two_maze':
                y1, y2, x1, x2 = myenv.initial_state(args.exo_noise, myenv, args.obs_type)
                x = torch.cat([x1,x2], dim=3)
            elif args.exo_noise == 'multi_maze':
                y1, y2, x1, x2 = myenv.initial_state(args.exo_noise, myenv_list, args.obs_type)
                x = torch.cat([x1,x2], dim=3)
                # img = x.squeeze(0).squeeze(0)
                # img=img.unsqueeze(0)
                # save_image(img, 'myimg.png')    
                # img = x.squeeze(0).squeeze(0)
                # img=img.unsqueeze(0)
                # save_image(img,  str(args.walls) + 'exo_mazes.png')
                          

            else:
                y1, y2, x1, x2 = myenv.initial_state(args.exo_noise, myenv, args.obs_type)
                x = x1

        elif args.data == 'cartpole':
            y1, y2, x1, x2 = myenv.initial_state(myenv, resize)
            x = x1

        elif args.data == 'minigrid':
            # exo_noise flag for minigrid during env selection
            y1, y2, x1, x2  = myenv.initial_state()
            x = x1
        else:
            raise Exception()

        is_initial = False 

    net.eval()

    #pick actions randomly or with policy
    if torch.cuda.is_available():
        init_state, z_init = net.encode((x*1.0).cuda())
        if args.data == 'vis-maze':
            init_pred_latent, init_pred_logits = l_net.encode(z_init.cuda(), init_state.cuda())

    else:
        init_state, z_init = net.encode((x*1.0))
        if args.data == 'vis-maze':
            init_pred_latent, init_pred_logits = l_net.encode(z_init, init_state)

    if args.use_gt:
        init_state = y1

    if always_random or mybuffer.num_ex < num_rand:
        a1 = myenv.random_action()
        rand_act = True
        if (always_random):
            steps_to_goal = args.k_steps  # Needs to be set here to be reachable
    else:

        reward = transition.select_goal()        
        t0 = time.time()
        rand_act = False

        if args.policy_selection == 'greedy':
            a1 = transition.select_policy(init_state.cpu().item(), reward)

        elif args.policy_selection == 'VI':
            if args.use_pretrained_policy : 
                a1 = get_action_one_step_lookahead(transition, Vlast_pretrained, init_state, ncodes, reward)
                Vlast = Vlast_pretrained
            else:
                a1, Vlast = value_iteration(transition.state_transition, ncodes, init_state, reward, V=Vlast, max_iter=args.value_iter)     

        elif args.policy_selection == 'VI-Goal':
            reward = transition.get_goal() 
            a1, Vlast = value_iteration(transition.state_transition, ncodes, init_state, reward, V=Vlast, max_iter=args.value_iter) 

        elif args.policy_selection == 'VI-Sub-Goal':
            reward = transition.get_sub_goal() 
            a1, Vlast = value_iteration(transition.state_transition, ncodes, init_state, reward, V=Vlast, max_iter=args.value_iter) 
        
        elif args.policy_selection == 'random-sub-goals':
            reward = transition.get_random_goals(5)
            a1, Vlast = value_iteration(transition.state_transition, ncodes, init_state, reward, V=Vlast, max_iter=args.value_iter) 

        elif args.policy_selection == 'DP-Goal':

            use_random_step = True
            do_abort = False

            greedy_plan = False
            #greedy_plan = (random.uniform(0,1) < 0.1)
            #greedy_plan = (random.uniform(0,1) < 1.0)

            if (steps_to_goal <= 0 and greedy_plan):

                v_dp, dp_depth, g_dp, a1 = DP_counts(transition.ls, init_state.item(), dp_step=transition.update_step, min_visit_count=0, code2ground=transition.code2ground)
                print('\t', 'greedy-plan', v_dp)
                steps_to_goal = dp_depth
                if dp_depth == 0:
                    steps_to_goal = 1
                rand_iter_step = random.randint(env_iteration, env_iteration + steps_to_goal - 1)
            elif (steps_to_goal <= 0):
                print('\t', 'uniform-plan')
                g_dp = select_random_reachable_goal(transition.ls, init_state.item(), dp_step=transition.update_step, code2ground=transition.code2ground)
                
                dp_depth, g_dp, a1 = DP_goals(transition.ls, init_state.item(), goal_index=g_dp, dp_step=transition.update_step, code2ground=transition.code2ground)
                steps_to_goal = dp_depth
                if dp_depth == 0:
                    steps_to_goal = 1
                #rand_iter_step = env_iteration + steps_to_goal + 5
                rand_iter_step = random.randint(env_iteration, env_iteration + steps_to_goal - 1)
            else:
                depth_claim, goal_claim, a1 = DP_goals(transition.ls, init_state.item(), goal_index=g_dp, dp_step=transition.update_step, code2ground=transition.code2ground)

            if dp_depth == args.ncodes+1:
                print('abort - cant find goal path, move randomly')
                do_abort = True
                #dp_depth = 0
                #steps_to_goal = 1

            print('env iteration', env_iteration, 'rand_iter_step', rand_iter_step)

            if purely_random or ((do_abort and use_random_step) or (env_iteration == rand_iter_step and use_random_step)):
                a1 = myenv.random_action()
                took_trainable = True
                print('\t', 'random action', a1)
            else:
                took_trainable = False
            print('\t', 'step', env_iteration, 'state', init_state.item(), 'goal', g_dp, 'action', a1, 'steps_to_goal', steps_to_goal, 'dp_depth-original', dp_depth)

            #print('using DP pick action', a1)
        elif args.policy_selection == 'random':
            took_trainable = True
            steps_to_goal = 20
            g_dp = init_state.item()
            a1 = myenv.random_action().item()

            print('\t', 'step', env_iteration)

    a1 = torch.Tensor([a1]).long()
    a2 = myenv.random_action()

    if args.exo_noise == 'three_maze':
        a3 = myenv.random_action()

    if args.data == 'mnist':
        x1_, x2_, y1_, y2_ = myenv.transition(a1,a2,y1,y2,c1,c2)
        x_ = torch.cat([x1_*c1,x2_*c2], dim=3)

    elif args.data == 'maze' or 'periodic-cart':
        y1_, y2_, x1_, x2_, _, _ = myenv.step(a1,a2)
        ground_state = y1_.item()
        state_visits[ground_state] += 1
        all_states_count[env_iteration, :] = state_visits
        if args.exo_noise == 'two_maze':
            x_ = torch.cat([x1_,x2_], dim=3)
        else:
            x_ = x1_

    elif args.data == 'vis-maze':
        if args.exo_noise == 'two_maze':
            y1_, y2_, x1_, x2_, r, done = myenv.step(a1, a2, myenv, args.obs_type)
            x_ = torch.cat([x1_,x2_], dim=3)
        elif args.exo_noise == 'multi_maze':
            y1_, y2_, x1_, x2_, r, done = myenv.step(a1, a2, myenv_list, args.obs_type)
            x_ = torch.cat([x1_,x2_], dim=3)
        else:
            y1_, y2_, x1_, x2_, r, done = myenv.step(a1, a2, myenv, args.obs_type)
            x_ = x1_            

    elif args.data == 'cartpole':
        y1_, y2_, x1_, x2_, r, done = myenv.transition(a1, resize)
        x_ = x1_

    elif args.data == 'minigrid':
        y1_, y2_, x1_, x2_, r, done = myenv.step_transition(a1, a2)
        x_ = x1_
    else:
        raise Exception()    

    if torch.cuda.is_available():
        x_ = x_.cuda()

    t0 = time.time()
    next_state, z_next = net.encode((x_*1.0))
    if args.data == 'vis-maze':
        next_pred_latent, next_pred_logits = l_net.encode(z_next, next_state) ## condition on action for predicting the next ground state

    if args.use_gt:
        next_state = y1_

    transition.update(init_state, next_state, torch.Tensor([a1]).long(), y1, y1_)

    if dp_depth == 0:
        print('\t', 'set next state to', next_state.item(), 'curr-state', init_state.item(), 'steps_to_goal', steps_to_goal)
        g_dp = next_state.item()

    if not init_state.item() in ls2obs:
        ls2obs[init_state.item()] = []
    ls2obs[init_state.item()].append(x)
   

    if not (type(g_dp) is int):
        print('g_dp not int', g_dp, type(g_dp))
        raise Exception()

    if g_dp in ls2obs:
        goal_obs = random.choice(ls2obs[g_dp])
    else:
        print('in keys', list(ls2obs.keys()))
        print('could not find learned code', g_dp)
        #raise Exception()
        goal_obs = torch.zeros_like(x)

    mybuffer.add_example(a1, y1, y1_, y2, y2_, x, x_, init_state.item(), g_dp, goal_obs, steps_to_goal, took_trainable, step, dp_depth, z_init, z_next)
        
    # transition.print_codes(init_state.item(), next_state.item(), a1, g_dp)

    y1 = y1_
    y2 = y2_
    x = x_.cpu()
    steps_to_goal -= 1

    if mybuffer.num_ex < args.num_rand_initial or mybuffer.num_ex % args.model_train_iter != 0:
        continue

    transition.reset()
    net.train()
    num_iter = args.num_iter

    if (env_iteration+1) % 1000 == 0:
    # if env_iteration > 2000 : 
        eval_state_visit = all_states_count[env_iteration, :]
        all_state_visits.append(eval_state_visit)
        if (logger is not None):
            logger.save_state_visits(all_state_visits, env_iteration)
        # plot_state_visitations(state_visits, args.rows, logger.save_folder, env_iteration)

    for iteration in range(0, num_iter):

        print_ = iteration==num_iter-1

        do_quantize = True
        reinit_code = False

        if args.data != 'vis-maze': ### linear classifier not tested for other envs
            l_net = net

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, xlst, _, _ = update_model(net, l_net, mybuffer, print_, do_quantize, reinit_code, args.batch_size, None, None)        

        ### byol loss
        #obs_last, obs_next, a = mybuffer.sample_byol()
        #z_last, extra_loss_1, _ = net.enc(obs_last)
        #z_next, extra_loss_2, _ = net.enc(obs_next)
        #loss_byol = net.byol(z_last, z_next, a)
        #loss += extra_loss_1 + extra_loss_2 + loss_byol

        if iteration % 100 == 0:
            print('loss', loss)

        opt.zero_grad()
        if (args.use_best_model and iteration >= num_iter*.1):
            losses.append(loss.detach())
            if(len(losses) > 20):
                losses = losses[-20:]
                if(sum(losses)/len(losses) < best_avg_loss):
                    best_avg_loss = sum(losses)/len(losses)
                    print('New best avg loss: ' +str(best_avg_loss))
                    torch.save(net, args.log_eval_prefix + "_it_"+ str(mybuffer.num_ex) + "_seed_" + str(args.seed) + "_checkpoint.pth")
        loss.backward()
        opt.step()

        if args.data == 'vis-maze':
            gt_last, gt_next, pred_z_last, pred_z_next, a_b, obs_last, obs_next = mybuffer.sample_classifier()

            n_last, zb_last = net.encode((obs_last*1.0))
            n_next, zb_next = net.encode((obs_last*1.0))

            _, pred_gt_last_logit = l_net.encode(zb_last, n_last) 
            _, pred_gt_next_logit = l_net.encode(zb_next, n_next) 

            lin_classifier_loss_last = (ce(pred_gt_last_logit, gt_last)).mean()
            lin_classifier_loss_next = (ce(pred_gt_next_logit, gt_next)).mean()
            lin_classifier_loss = lin_classifier_loss_last + lin_classifier_loss_next
            
            l_opt.zero_grad()
            
            lin_classifier_loss.backward()
            l_opt.step()

        pred = out.argmax(1)

        ################## New final Evaluator ##################
        # Build graph from current encoder for entire replay buffer
        if ((iteration+1) % args.eval_iter == 0):
            if (args.use_best_model):
                eval_net = torch.load( args.log_eval_prefix + "_it_"+ str(mybuffer.num_ex) + "_seed_" + str(args.seed) + "_checkpoint.pth")
            else:
                eval_net = net
            codes = []
            codes_ = []
            eval_net.eval()
            with torch.no_grad():
                for x in mybuffer.x:
                    codes.append(eval_net.encode((x*1.0).cuda())[0].item())
                for x_ in mybuffer.x_:
                    codes_.append(eval_net.encode((x_*1.0).cuda())[0].item())
                    
                transition_freq_mat = np.zeros([transition.na, args.ncodes, args.ncodes])
                codes_by_gt = np.zeros([args.ncodes,myenv.total_states]) 
                for i in range(len(mybuffer.x)):
                    transition_freq_mat[mybuffer.a[i],codes[i],codes_[i]] += 1
                    codes_by_gt[codes[i], mybuffer.y1[i]] += 1

                transitions_for_dijkstra = -1 * np.ones([args.ncodes, args.ncodes],dtype=int) # equals an (arbitrary) action if one is available between two codes, -1 otherwise

                for i in range(args.ncodes):
                    for j in range(transition.na):
                        if (transition_freq_mat[j,i].sum() !=0):
                            k = transition_freq_mat[j,i].argmax()
                            transitions_for_dijkstra[i,k] = j


                dijkstra_adjacency_matrix = np.ones([args.ncodes, args.ncodes])
                dijkstra_adjacency_matrix[transitions_for_dijkstra == -1] = np.inf
                dijkstra_adjacency_matrix = csr_matrix(dijkstra_adjacency_matrix)

                # For q samples, get two random observations, plan with dijkstras. If it excecutes a correct path, we score
                q = 1000
                #NOTE: must set stochastic start
                wins = 0

                assert args.exo_noise == 'two_maze'
                assert args.stochastic_start
                for i in range(q):

                    dest_gt,_,_,_,x1_,x2_ = myenv_eval.initial_state()

                    dest_obs = torch.cat([x1_,x2_], dim=3)
                    dest_code = eval_net.encode((dest_obs*1.0).cuda())[0].item()

                    source_gt,_,_,_,x1_,x2_ = myenv_eval.initial_state()

                    source_obs = torch.cat([x1_,x2_], dim=3)
                    source_code = eval_net.encode((source_obs*1.0).cuda())[0].item()

                    dist_matrix, predecessors, _ =  dijkstra(min_only= True, csgraph=dijkstra_adjacency_matrix, directed=True, indices=source_code, return_predecessors=True)

                    if dist_matrix[dest_code] == np.inf:
                        continue
                    curr = dest_code
                    actions = []
                    while(curr != source_code):
                        prev = predecessors[curr]
                        act = transitions_for_dijkstra[prev,curr]
                        assert act != -1
                        actions = [act] + actions
                        curr = prev
                    gt = source_gt
                    for act in actions:
                        gt,_,_,_,_,_ = myenv_eval.step(act, act.item())
                    if (gt == dest_gt):
                        wins += 1
                torch.save({"total": + q, "wins": wins}, args.log_eval_prefix + "_it_"+ str(mybuffer.num_ex) + "_train_it_"+ str(iteration+1) + "_seed_" + str(args.seed) + ".pth")
                print("Codes to GT:")
                for i in range(args.ncodes):
                    sum_ = codes_by_gt[i].sum()
                    inds = np.flip(np.argsort(codes_by_gt[i]))[:4]
                    inds_prob_1 = codes_by_gt[i,inds[0]]/sum_
                    inds_prob_2 = codes_by_gt[i,inds[1]]/sum_
                    inds_prob_3 = codes_by_gt[i,inds[2]]/sum_
                    inds_prob_4 = codes_by_gt[i,inds[3]]/sum_
                    print("Code " + str(i)+": freq: " + str(sum_) + " top GT: " + str(inds[0]) + " (" + str(inds_prob_1*100.) + "%), "+ str(inds[1]) + " (" + str(inds_prob_2*100.) + "%), "+ str(inds[2]) + " (" + str(inds_prob_3*100.) + "%), "+ str(inds[3]) + " (" + str(inds_prob_4*100.) + "%)")
                print("GT to Codes:")
                for i in range(myenv.total_states):
                    sum_ = codes_by_gt[:,i].sum()
                    inds = np.flip(np.argsort(codes_by_gt[:,i]))[:4]
                    inds_prob_1 = codes_by_gt[inds[0],i]/sum_
                    inds_prob_2 = codes_by_gt[inds[1],i]/sum_
                    inds_prob_3 = codes_by_gt[inds[2],i]/sum_
                    inds_prob_4 = codes_by_gt[inds[3],i]/sum_
                    print("GT " + str(i)+": freq: " + str(sum_) + " top Codes: " + str(inds[0]) + " (" + str(inds_prob_1*100.) + "%), "+ str(inds[1]) + " (" + str(inds_prob_2*100.) + "%), "+ str(inds[2]) + " (" + str(inds_prob_3*100.) + "%), "+ str(inds[3]) + " (" + str(inds_prob_4*100.) + "%)")
                print("Out of " + str(q) + " path-finding trials, suceeded in " + str(wins) + ".")
            eval_net.train()
    #Train the dist pred model
    print("TRAINING PRED MODEL")
    for iteration in range(0,100):
        d_x, d_xk, d_k = mybuffer.sample_dist()
        d_z = net.encode_emb(d_x).detach()
        d_zk = net.encode_emb(d_xk).detach()
        distpred.train_net(d_z, d_zk, d_k)

    net.eval()

    ls2obs = {}

    accs = []
    for k in range(0, mybuffer.num_ex, 100):

        ex_lst = list(range(k,min(mybuffer.num_ex, k+100)))

        out, loss, ind_last, ind_new, a1, tr_y1, tr_y1_, xlst, _ , _= update_model(net, l_net,  mybuffer, print_, True, False, len(ex_lst), ex_lst, klim=1)        
        # pred = out.argmax(1)

        ### predict with linear classifier
        if args.data == 'vis-maze':
            print("PREDICT W/ CLASSIFIER MODEL")
            gt_last, _, _, _, _, obs_last, _ = mybuffer.sample_classifier()
            n_last, zb_last = net.encode((obs_last*1.0))
            predicted_gt_last = l_net.predict(gt_last, zb_last)
            classifier_accuracy = predicted_gt_last.eq(gt_last.view_as(predicted_gt_last)).sum().item()
            accuracy =  (classifier_accuracy / len(predicted_gt_last)) * 100
            accs.append(accuracy)
            all_classifier_accuracy.append(accuracy)

        if args.use_logger:
            logger.record_classifier_accuracy(accs)
            logger.record_all_classifier_accuracy(all_classifier_accuracy)
            logger.save()

        if args.use_gt:
            ind_last = tr_y1
            ind_new = tr_y1_

        for j in range(ind_last.shape[0]):
            s = ind_last[j].item()

            if not s in ls2obs:
                ls2obs[s] = []
            ls2obs[s].append(xlst[j:j+1])
                   
        transition.update(ind_last, ind_new, a1, tr_y1, tr_y1_)
    
    transition.save_log(logger, env_iteration)


