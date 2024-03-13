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
import torch
import torch.nn as nn
import torch.nn.functional as F

#obj = pickle.load(open('dp_obj_2.pkl', 'rb'))

#counts = obj['trans']
#code2ground = obj['code2ground']
#print('counts shape', counts.shape)
#print('code2ground', code2ground)


if __name__ == "__main__":
    obj = pickle.load(open('pkl_obj_3.pkl', 'rb'))
    ls = obj['ls']
    code2ground = obj['code2ground']
    dp_step_use = obj['dp_step']


class learned_action:
    next_states: list #of int, index into learned_state
    counts: list #of int, count of observed transitions
 
   
class learned_state:
    update_step: int
    dp_step: int #dynamic programming helper
    minimals: tuple #dynamic programming return value
    goal_minimals: tuple
    actions: list #of int, actions available
    action_counts: list #of int, ranging over actions
    transitions: list #of learned_action, ranging over actions
    in_queue: bool

max_visitation = 1e9

max_horizon = 100

def find_minimals(action_counts, effective_depth, goal_lst, self_trans):
    min_visit_value = action_counts[0]
    min_depth = effective_depth[0]
    best_action = 0
    
    #print('\t'*(max_horizon-time_horizon), 'calling find minimals')

    #print('find-min', action_counts, effective_depth, self_trans)
    #print('goal-vector', list(map(lambda a: code2ground[a], goal_lst)))


    has_self_trans = False
    
    for a_ind in range(len(action_counts)):

        visits = action_counts[a_ind]
        if (visits < min_visit_value):
            #print('\t'*(max_horizon-time_horizon), 'sel visits', visits, 'depth', effective_depth[a_ind], 'action', a_ind)
            min_visit_value = visits
            min_depth = effective_depth[a_ind]
            best_action = a_ind
            has_self_trans = self_trans[a_ind]
        elif visits == min_visit_value:
            if effective_depth[a_ind] < min_depth:
                #print('\t'*(max_horizon-time_horizon), 'visits tie', visits, 'depth', effective_depth[a_ind], 'action', a_ind)
                min_visit_value = visits
                min_depth = effective_depth[a_ind]
                best_action = a_ind
                has_self_trans = self_trans[a_ind]
            elif effective_depth[a_ind] == min_depth and has_self_trans and (self_trans[a_ind]==False):
                min_visit_value = visits
                min_depth = effective_depth[a_ind]
                best_action = a_ind
                has_self_trans = self_trans[a_ind]

    #alst = []
    #for action in range(action_counts.shape[0]):
    #    if (action_counts[action] == min_visit_value):
    #        alst.append(action)


    return (min_visit_value, min_depth, goal_lst[best_action], best_action)

def queue(learned_states, current_state_index,min_visit_count, dp_step):

    qobj = [[current_state_index]]
    learned_states[current_state_index].in_queue = True

    if min(learned_states[current_state_index].action_counts) == min_visit_count:
        return qobj

    for i in range(0, len(learned_states)):
        depth_lst = []

        for q_e in qobj[i]:

            for a_ind in learned_states[q_e].actions:

                if learned_states[q_e].action_counts[a_ind] == 0:
                    return qobj

                next_states = learned_states[q_e].transitions[a_ind].next_states

                next_state_visits = 0
                rand_goal_found = False
             
                random.seed(hash((q_e, a_ind, dp_step)))
                rand_num = random.randint(0, learned_states[q_e].action_counts[a_ind]-1)

                for next_state in next_states:
                    next_st = learned_states[next_state]

                    next_state_visits += learned_states[q_e].transitions[a_ind].counts[next_state]

                    if next_state_visits > rand_num and not rand_goal_found:
                        rand_goal_found = True

                        if (learned_states[next_st.minimals[2]].update_step <= next_st.dp_step and i >= next_st.minimals[1]) or next_st.in_queue:
                            continue
                        elif min(next_st.action_counts) <= min_visit_count:

                            #print('search early-stop queueing on hit', next_state)

                            for d_e in depth_lst:
                                learned_states[d_e].in_queue = False

                            depth_lst = [next_state]
                            qobj.append(depth_lst)
                            next_st.in_queue = True
                            #print('found ideal - early stop')
                            return qobj
                        else:
                            next_st.in_queue = True
                            depth_lst.append(next_state)
                        
        if len(depth_lst) > 0:
            qobj.append(depth_lst)
        else:
            return qobj


def build_dp(learned_states, qobj, min_visit_count, dp_step):

    for j in reversed(range(0, len(qobj))):
        
        for q_e in qobj[j]:

            current_state = learned_states[q_e]

            effective_visitations = copy.deepcopy(current_state.action_counts)
            effective_depth = [0]*len(current_state.action_counts)

            #if q_e == 10:
            #    print('effective initial (2,1)', effective_visitations)

            goal_lst = [q_e]*len(current_state.action_counts)

            self_trans = [False]*len(current_state.actions)
    
            current_state.minimals = find_minimals(effective_visitations, effective_depth, goal_lst, self_trans)
            current_state.dp_step = dp_step

            if current_state.minimals[0] <= min_visit_count:
                #print('early exit min-visit')
                current_state.in_queue = False
                continue

            
            for a_ind in current_state.actions:

                best_goal_found = q_e

                next_states = current_state.transitions[a_ind].next_states
                total_visits = min(current_state.action_counts)
                #print('init total_visits', total_visits)
                goal_depth = 0

                random.seed(hash((q_e, a_ind, dp_step)))
                rand_num = random.randint(0, current_state.action_counts[a_ind]-1)
                next_state_visits = 0
                rand_goal_found = False

                for next_state in next_states:

                    next_state_visits += current_state.transitions[a_ind].counts[next_state]

                    if learned_states[next_state].in_queue:
                        #print('next state in-queue skip', 'skipping', code2ground[next_state])

                        if next_state_visits > rand_num and not rand_goal_found and next_state == q_e:
                            self_trans[a_ind] = True

                        continue

                    min_visit_found, depth_found, goal_found, alst_found = learned_states[next_state].minimals

                    #if q_e == 34:
                    #    print('found at 34', learned_states[next_state].minimals)

                    if next_state_visits > rand_num and not rand_goal_found:
                        best_goal_found = goal_found
                        goal_depth = depth_found + 1
                        total_visits = min_visit_found
                        rand_goal_found = True

                        #print('on a_ind', a_ind, 'setting to goal-found', code2ground[goal_found], 'depth-found', goal_depth, 'backing-up-from', code2ground[next_state])

                    #total_visits += min_visit_found * current_state.transitions[a_ind].counts[next_state]


                #total_visits = total_visits / current_state.action_counts[a_ind]

                #if q_e == 33:
                #    print('ef-33-before', effective_visitations, effective_depth, list(map(lambda a: code2ground[a], goal_lst)), code2ground[goal_found], goal_depth, self_trans)

                #if q_e == 34:
                #    print('ef-34-before', effective_visitations, effective_depth, list(map(lambda a: code2ground[a], goal_lst)), code2ground[goal_found], goal_depth, self_trans)
                

                if (effective_visitations[a_ind] > total_visits or ((total_visits == effective_visitations[a_ind]) and (goal_depth < effective_depth[a_ind]))) and (self_trans[a_ind]==False):
                    effective_visitations[a_ind] = total_visits
                    effective_depth[a_ind] = goal_depth
                    goal_lst[a_ind] = best_goal_found
                
                #if q_e == 33:
                #    print('ef-33-after', effective_visitations, effective_depth, list(map(lambda a: code2ground[a], goal_lst)), code2ground[goal_found], goal_depth, self_trans)

                #if q_e == 34:
                #    print('ef-34-after', effective_visitations, effective_depth, list(map(lambda a: code2ground[a], goal_lst)), code2ground[goal_found], goal_depth, self_trans)

            #if q_e == 10:
            #    print('effectivevisits before fm', effective_visitations)
            #    print('depth before fm', effective_depth)
            #print('effective_visits', effective_visitations)

        

            current_state.minimals = find_minimals(effective_visitations, effective_depth, goal_lst, self_trans)
            current_state.dp_step = dp_step
            #print('found minimals', code2ground[q_e], current_state.minimals)
            
            #if q_e == 33:
            #    print('minimals-33', current_state.minimals)

            current_state.in_queue = False

    return learned_states[qobj[0][0]].minimals

def dijkstra(learned_states, current_state_index, min_visit_count, dp_step, code2ground):

    qobj = queue(learned_states, current_state_index, min_visit_count, dp_step)

    #print('queing in DP-findgoal')

    for i in range(len(qobj)):
        sl = []
        for qe in qobj[i]:
            if qe in code2ground:
                sl.append((qe, code2ground[qe]))
            else:
                sl.append((qe, None))

        #print(i, sl)
    m = build_dp(learned_states, qobj, min_visit_count, dp_step)

    #print('value-seeking found minimals', m)

    return m


def make_ls(counts, ns, na):

    state_lst = []

    min_visit_count = 1e9

    for i in range(ns):
        ls = learned_state()
        #ls.visited = False
        ls.actions = range(na)

        
        ls.action_counts = counts[i].sum(1).numpy().tolist()

        #if i in code2ground:
        #    print('state', code2ground[i], ls.action_counts)

        ls.minimals = (min(ls.action_counts), 0, i, 0)

        ls.goal_minimals = (0, i, 0)

        ls.update_step = 1
        ls.dp_step = 0
        ls.in_queue = False

        ls.transitions = []

        for action in ls.actions:
            la = learned_action()
            la.counts = counts[i][action]

            if la.counts.min() < min_visit_count:
                min_visit_count = la.counts.min().item()

            next_states = []
            for lsi in range(ns):
                if counts[i][action][lsi] != 0:
                    next_states.append(lsi)
            la.next_states = next_states
            ls.transitions.append(la)

        state_lst.append(ls)

    return state_lst, min_visit_count

def update_ls(ls, s_obs, a_obs, ns_obs, update_step):

    ls_up = ls[s_obs]

    ls_up.update_step = update_step
    ls_up.action_counts[a_obs] += 1

    trans_obj = ls_up.transitions[a_obs]

    if not (ns_obs in trans_obj.next_states):
        trans_obj.next_states.append(ns_obs)

    trans_obj.counts[ns_obs] += 1#[trans_obj.next_states.index(ns_obs)] += 1


#def DP(ls, ns, na, init_state, dp_step, max_horizon):
#    v,d,g,a,r_depth = find_min_visits(ls, init_state, dp_step, max_horizon)
#    return v,d,g,a,r_depth

import time

def DP_counts(ls, init_state, dp_step, min_visit_count, code2ground):

    #print('calling dp horizon', max_horizon)

    #ns = counts.shape[0]
    #na = counts.shape[1]

    #t0 = time.time()
    #ls, _ = make_ls(counts,ns,na)
    #print('update time', time.time() - t0)

    #print('min_visit_count', min_visit_count)

    #t0 = time.time()
    v,d,g,a = dijkstra(ls, init_state, min_visit_count, dp_step=dp_step, code2ground=code2ground)
    #print('d time', time.time() - t0)

    return v,d,g,a

if __name__ == "__main__":

    import numpy as np

    #ns = 2
    #na = 5 # 0/1/2/3 up/right/down/left.  

    #counts = np.random.randint(0,1,size=(ns,na,ns))

    #for i in range(ns-1):
    #    counts[i,0,max(0,i-1)] += 50
    #    counts[i,1,i+1] += 50
    #counts[ns-1,0,ns-2] += 1
    #counts[ns-1,1,ns-1] += 1
    #counts[2,0,1] = 0

    #counts[0,0,0] = 1.0
    #counts[0,1,0] = 1.0
    #counts[0,2,0] = 1.0
    #counts[0,3,1] = 1.0
    #counts[0,4,0] = 1.0

    #counts[1,0,1] = 0.0
    #counts[1,1,1] = 1.0
    #counts[1,2,1] = 1.0
    #counts[1,3,1] = 1.0
    #counts[1,4,1] = 1.0

    #for i in range(ns):
    #    print(i, counts[i])

    #counts[29,3,:] *= 0.0
    #counts[29,3,29] += 1.0

    print('code2ground', code2ground)

    for init_state in list(code2ground.keys()):

        print("DP", code2ground[init_state], 'lc', init_state)
        v,d,g,a = DP_counts(ls,init_state,dp_step=1,min_visit_count=0,code2ground=code2ground)

        print("Called DP on", code2ground[init_state], 'lc', init_state)
        print('v', v)
        print('d', d)
        print('g', code2ground[g])
        print('a', a)
    

    #print("DP-2")
    #print(DP(ls,ns,na,0,2,10))

    #print('update-ls')
    #update_ls(ls, 9, 1, 9, 3)

    #print("DP-3")
    #print(DP(ls,ns,na,0,3,10))