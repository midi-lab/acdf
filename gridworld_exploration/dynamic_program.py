# find_min_visits is the primary entry point.  It takes as input a set
# of learned_states (as a list), the current_state (as an index into
# learned_states), and a time_horizon, then applies dynamic
# programming to return the value of the best action and a list of best actions.
#
# The only complex data structure here is a learned_state.  A learned state has several fields.

import time
import random
import copy 
from dataclasses import dataclass

import pickle
import numpy

# obj = pickle.load(open('dp_obj.pkl', 'rb'))

# counts = obj['trans']
# code2ground = obj['code2ground']
# print('counts shape', counts.shape)
# print('code2ground', code2ground)

class learned_action:
    next_states: list #of int, index into learned_state
    counts: list #of int, count of observed transitions
    
class learned_state:
    update_step: int
    dp_step: int #dynamic programming helper
    minimals: tuple #dynamic programming return value
    actions: list #of int, actions available
    action_counts: list #of int, ranging over actions
    transitions: list #of learned_action, ranging over actions

max_visitation = 1e9

max_horizon = 100

def find_minimals(action_counts, effective_depth, goal_lst, time_horizon):
    min_visit_value = 999#action_counts[0]
    min_depth = 999#effective_depth[0]
    best_action = -1#0
    
    r_depth = max_horizon - time_horizon

    #print('\t'*(max_horizon-time_horizon), 'calling find minimals')

    for a_ind in range(len(action_counts)):

        visits = action_counts[a_ind]
        if (visits < min_visit_value):
            #print('\t'*(max_horizon-time_horizon), 'sel visits', visits, 'depth', effective_depth[a_ind], 'action', a_ind)
            min_visit_value = visits
            min_depth = effective_depth[a_ind]
            best_action = a_ind
        elif visits == min_visit_value:
            if effective_depth[a_ind] < min_depth:
                #print('\t'*(max_horizon-time_horizon), 'visits tie', visits, 'depth', effective_depth[a_ind], 'action', a_ind)
                min_visit_value = visits
                min_depth = effective_depth[a_ind]
                best_action = a_ind

    #alst = []
    #for action in range(action_counts.shape[0]):
    #    if (action_counts[action] == min_visit_value):
    #        alst.append(action)


    return (min_visit_value, min_depth, goal_lst[best_action], best_action, r_depth)


def find_min_visits(learned_states, current_state_index, dp_step, time_horizon):

    #print('\t'*(max_horizon-time_horizon), 'Calling find min visits on', code2ground[current_state_index])

    current_state = learned_states[current_state_index]
    if (time_horizon <= 0):
        return (max_visitation, 0, current_state_index, -1, max_horizon-time_horizon)

    if (learned_states[current_state.minimals[2]].update_step <= current_state.dp_step) and (max_horizon - current_state.minimals[4] > time_horizon):
        #print('\t'*(max_horizon-time_horizon), 'early-return', 'horizon-best', max_horizon - current_state.minimals[1], 'time_horizon', time_horizon)
        return current_state.minimals
    #current_state.visited=True
    current_state.dp_step = dp_step

    effective_visitations = copy.deepcopy(current_state.action_counts)
    effective_depth = [0]*len(current_state.action_counts)

    goal_lst = [current_state_index]*len(current_state.action_counts)

    #print('\t'*(max_horizon-time_horizon), 'call find minimals on', code2ground[current_state_index])
    current_state.minimals=find_minimals(current_state.action_counts, effective_depth, goal_lst, time_horizon)

    if (current_state.minimals[0] == 0):
        return current_state.minimals


    for a_ind in current_state.actions:
        next_states = current_state.transitions[a_ind].next_states
        total_visits = 0
        total_depth = 0


        rand_num = random.randint(0, current_state.action_counts[a_ind]-1)
        next_state_visits = 0
        rand_goal_found = False

        for n_index in range(len(next_states)):


            min_visit_found, depth_found, goal_found, alst_found, rdepth_found = find_min_visits(learned_states, next_states[n_index], dp_step, time_horizon-1)
            
            #print('\t'*(max_horizon-time_horizon), 'from state', code2ground[current_state_index], 'going over', list(map(lambda z: code2ground[z], next_states)), 'min_visit_found', min_visit_found, 'horizon', time_horizon)

            next_state_visits += current_state.transitions[a_ind].counts[next_states[n_index]]

            if next_state_visits > rand_num and not rand_goal_found: 
                goal_lst[a_ind] = goal_found
                rand_goal_found = True

            total_visits += min_visit_found * current_state.transitions[a_ind].counts[next_states[n_index]]
            total_depth += (depth_found+1) * current_state.transitions[a_ind].counts[next_states[n_index]]


        total_visits = total_visits / current_state.action_counts[a_ind]
        total_depth = total_depth / current_state.action_counts[a_ind]


        if effective_visitations[a_ind] > total_visits or ((total_visits == effective_visitations[a_ind]) and (total_depth < effective_depth[a_ind])):

            effective_visitations[a_ind] = total_visits
            effective_depth[a_ind] = total_depth

    #print('\t'*(max_horizon-time_horizon), 'call find minimals-end on', code2ground[current_state_index])
    current_state.minimals = find_minimals(effective_visitations, effective_depth, goal_lst, time_horizon)
    
    return current_state.minimals

def make_ls(counts, ns, na):

    state_lst = []

    for i in range(ns):
        ls = learned_state()
        #ls.visited = False
        ls.actions = range(na)
        ls.minimals = (0, 0, i, 0, 0)
        ls.action_counts = counts[i].sum(1)
        ls.update_step = 1
        ls.dp_step = 0

        ls.transitions = []

        for action in ls.actions:
            la = learned_action()
            la.counts = counts[i][action]
            next_states = []
            for lsi in range(ns):
                if counts[i][action][lsi] != 0:
                    next_states.append(lsi)
            la.next_states = next_states
            ls.transitions.append(la)

        state_lst.append(ls)

    return state_lst

def update_ls(ls, s_obs, a_obs, ns_obs, update_step):


    ls_up = ls[s_obs]

    ls_up.update_step = update_step
    ls_up.action_counts[a_obs] += 1

    trans_obj = ls_up.transitions[a_obs]

    if not (ns_obs in trans_obj.next_states):
        trans_obj.next_states.append(ns_obs)

    trans_obj.counts[ns_obs] += 1#[trans_obj.next_states.index(ns_obs)] += 1


def DP(ls, ns, na, init_state, dp_step, max_horizon):

    v,d,g,a,r_depth = find_min_visits(ls, init_state, dp_step, max_horizon)

    return v,d,g,a,r_depth

def DP_counts(counts, init_state, max_horizon):

    #print('calling dp horizon', max_horizon)

    ns = counts.shape[0]
    na = counts.shape[1]

    ls = make_ls(counts,ns,na)

    v,d,g,a,r_depth = find_min_visits(ls, init_state, dp_step=1, time_horizon=max_horizon)

    if a == -1 and max_horizon > 0:
        print('max horizon', max_horizon)
        raise Exception('action -1')

    if max_horizon == 0:
        a = 0#a = random.choice([0,1,2,3])

    return v,d,g,a,r_depth

# if __name__ == "__main__":

#     import numpy as np

#     #ns = 2
#     #na = 5 # 0/1/2/3 up/right/down/left.  

#     #counts = np.random.randint(0,1,size=(ns,na,ns))

#     #for i in range(ns-1):
#     #    counts[i,0,max(0,i-1)] += 50
#     #    counts[i,1,i+1] += 50
#     #counts[ns-1,0,ns-2] += 1
#     #counts[ns-1,1,ns-1] += 1
#     #counts[2,0,1] = 0

#     #counts[0,0,0] = 1.0
#     #counts[0,1,0] = 1.0
#     #counts[0,2,0] = 1.0
#     #counts[0,3,1] = 1.0
#     #counts[0,4,0] = 1.0

#     #counts[1,0,1] = 0.0
#     #counts[1,1,1] = 1.0
#     #counts[1,2,1] = 1.0
#     #counts[1,3,1] = 1.0
#     #counts[1,4,1] = 1.0

#     #for i in range(ns):
#     #    print(i, counts[i])

#     ns = counts.shape[0]
#     na = counts.shape[1]


#     for init_state in code2ground.keys():

#         ls = make_ls(counts,ns,na)

#         print("DP", code2ground[init_state], 'lc', init_state)
#         v,d,g,a,r_depth = DP(ls,ns,na,init_state,dp_step=1,max_horizon=100)

#         print("Called DP on", code2ground[init_state], 'lc', init_state)
#         print('v', v)
#         print('d', d)
#         print('g', code2ground[g])
#         print('r_depth', r_depth)
#         print('a', a)
    

    #print("DP-2")
    #print(DP(ls,ns,na,0,2,10))

    #print('update-ls')
    #update_ls(ls, 9, 1, 9, 3)

    #print("DP-3")
    #print(DP(ls,ns,na,0,3,10))




