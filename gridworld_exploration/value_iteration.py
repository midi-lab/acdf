import sys
# import gym
import numpy as np

import torch
import random

# Calculate a state-value function
def one_step_lookahead(state, V, discount, probs, n_actions, n_states, rewards):
    """
    Helper function to calculate a state-value function.
    
    :param env: object
        Initialized OpenAI gym environment object
    :param V: 2-D tensor
        Matrix of size nSxnA, each cell represents 
        a probability of taking actions
    :param state: int
        Agent's state to consider
    :param discount: float
        MDP discount factor
    
    :return:
        A vector of length nA containing the expected value of each action
    """
    n_actions = n_actions
    action_values = np.zeros(shape=n_actions)

    for action in range(n_actions):
        for next_state in range(0, n_states):
            reward = rewards[next_state]
            prob = probs[state, action, next_state]
            action_values[action] += prob * (reward + discount * V[next_state])

    return action_values



def value_iteration(t_counts, n_states, eval_state, rewards, V=None, discount=0.99, theta=1e-5, max_iter=50, sample_action=False):
    """
    Value iteration algorithm to solve MDP.
    
    :param env: object
        Initaized OpenAI gym environment object
    :param discount: float default 1e-1
        MDP discount factor
    :param theta: float default 1e-9
        Stopping threshold. If the value of all states
        changes less than theta in one iteration
    :param max_iter: int
        Maximum number of iterations that can be ever performed
        (to prevent infinite horizon)
    
    :return: tuple(policy, V)
        policy: the optimal policy determined by the value function
        V: the optimal value determined by the value function
    """
    # initalized state-value function with zeros for each env state
    if V is None:
        V = np.zeros(n_states)
    
    n_actions = 4
    
    eps = 1e-9
    counts = t_counts + eps
    probs = counts / (counts.sum(dim=2, keepdim=True))

    for i in range(int(max_iter)):
        # early stopping condition
        delta = 0
        # update each state
        for state in range(n_states):
            # Do a one-step lookahead to calculate state-action values
            action_value = one_step_lookahead(state, V, discount, probs, n_actions, n_states, rewards)
            # select best action to perform based on the highest state-action values
            best_action_value = np.max(action_value)
            # calculate change in value
            delta = max(delta, np.abs(V[state] - best_action_value))
            # update the value function for current state
            V[state] = best_action_value
            
        # check if we can stop
        if delta < theta:
            # print("Value Iteration converged at iteration", i+1)
            # print(f'Value iteration converged at iteration #{i+1:,}')
            break
    
    # print('values', V)

    action_value = one_step_lookahead(eval_state, V, discount, probs, n_actions, n_states, rewards)
 
    print('action value', action_value)
   

    if sample_action:
        normed = action_value + 1e-3
        normed = normed / normed.sum()
        best_action = np.random.choice(list(range(len(normed))), 1, p=normed)[0]
    else:
        action_value += np.random.normal(0,0.0001,size=(4,))
        best_action = np.argmax(action_value)

    best_action = torch.Tensor([best_action]).long()

    return best_action, V



def get_action_one_step_lookahead(transition, V, state, n_states, rewards, discount=0.99):

    counts = transition.state_transition + 1e-9
    probs = counts / (counts.sum(dim=2, keepdim=True))

    n_actions = 4
    action_values = np.zeros(shape=n_actions)

    for action in range(n_actions):
        for next_state in range(0, n_states):
            reward = rewards[next_state]
            prob = probs[state, action, next_state]
            action_values[action] += prob * (reward + discount * V[next_state])

    action_values += np.random.normal(0,0.0001,size=(4,))
    best_action = np.argmax(action_values)

    best_action = torch.Tensor([best_action]).long()


    return best_action






def pre_train_policy(args, myenv, net, transition, value_iter):

    ep_length = args.ep_length
    step = 0
    Vlast = None


    if step == ep_length:
        is_initial=True
        step = 0
        ep_rand = random.randint(ep_length//2, ep_length) #after this step in episode follow random policy
    else:
        step += 1

    myenv.reset()

    if args.data == 'maze':
        y1,c1,y2,c2,x1,x2 = myenv.initial_state()
        if args.exo_noise == 'two_maze':
            x = torch.cat([x1,x2], dim=3)
        else:
            x = x1*1.0
    elif args.data == 'vis-maze': 
        y1, y2, x1, x2 = myenv.initial_state(args.exo_noise, myenv)
        if args.exo_noise == 'two_maze':
            x = torch.cat([x1,x2], dim=2)
        else:
            x = x1

    #pick actions randomly or with policy
    if torch.cuda.is_available():
        init_state = net.encode((x*1.0).cuda())
    else:
        init_state = net.encode((x*1.0))

    reward = transition.select_goal()  
    a1, Vlast = value_iteration(transition.state_transition, args.ncodes, init_state, reward, V=Vlast, max_iter=value_iter) 

    a2 = myenv.random_action()

    if args.data == 'maze':
        y1_, _, x1_, x2_, _, _ = myenv.step(a1,a2)

        if args.exo_noise == 'two_maze':
            x_ = torch.cat([x1_,x2_], dim=3)
        else:
            x_ = x1_

    elif args.data == 'vis-maze':
        y1_, y2_, x1_, x2_, r, done = myenv.step(a1, a2, args.noise_stationarity, args.obs_type, myenv)
        if args.exo_noise == 'two_maze':
            x_ = torch.cat([x1,x2], dim=2)
        else:
            x_ = x1     

    next_state = net.encode((x_*1.0))
    transition.update(init_state, next_state, a1, y1, y1_)

    transition.reset()

    return a1, Vlast
