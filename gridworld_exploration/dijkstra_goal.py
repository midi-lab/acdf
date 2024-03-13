# find_min_visits is the primary entry point.  It takes as input a set
# of learned_states (as a list), the current_state (as an index into
# learned_states), and a goal, then applies dynamic
# programming to return the value of the best action and a list of best actions.
#
# The only complex data structure here is a learned_state.  A learned state has several fields.

import time
import random
random.seed(42)
import copy 
from dataclasses import dataclass

import math
from dijkstra_program import learned_action, learned_state

import pickle
import numpy
import numpy as np

if __name__ == "__main__":
    obj = pickle.load(open('pkl_obj_3.pkl', 'rb'))
    ls = obj['ls']
    code2ground = obj['code2ground']
    dp_step_use = obj['dp_step']

max_visitation = 1e9

max_horizon = 100

def find_minimals(effective_depth, goal_index, self_trans):
    min_depth = effective_depth[0]
    best_action = 0
    has_self_trans = False


    for a_ind in range(len(effective_depth)):
        if effective_depth[a_ind] < min_depth:
            min_depth = effective_depth[a_ind]
            best_action = a_ind
            has_self_trans = self_trans[a_ind]
        elif effective_depth[a_ind] == min_depth and has_self_trans and (self_trans[a_ind]==False):
            min_depth = effective_depth[a_ind]
            best_action = a_ind
            has_self_trans = self_trans[a_ind]

    return min_depth, goal_index, best_action

def queue_for_goal(learned_states, current_state_index, goal_index, dp_step):
    qobj = [[current_state_index]]


    learned_states[current_state_index].in_queue = True

    

    if current_state_index == goal_index:
        return qobj

    for i in range(0, len(learned_states)):
        depth_lst = []

        for q_e in qobj[i]:
            for a_ind in learned_states[q_e].actions:
                next_states = learned_states[q_e].transitions[a_ind].next_states
            
                random.seed(hash((q_e, a_ind, dp_step)))
                #print('seed', (q_e, a_ind, dp_step))
                rand_num = random.randint(0, max(0,learned_states[q_e].action_counts[a_ind]-1))
                #print('rand_num_q', rand_num)

                next_state_visits = 0
                rand_goal_found = False

                for next_state in next_states:

                    next_state_visits += learned_states[q_e].transitions[a_ind].counts[next_state]

                    next_st = learned_states[next_state]

                    #print('c1', next_st.minimals[1] == goal_index)
                    #print('c2', learned_states[next_st.minimals[1]].update_step <= next_st.dp_step)
                    #print('c3', i >= next_st.minimals[0])

                    #print('next-st dp step', next_st.dp_step)

                    if next_state_visits > rand_num and not rand_goal_found:
                        rand_goal_found = True
                        if (next_st.goal_minimals[1] == goal_index
                            and learned_states[next_st.goal_minimals[1]].update_step <= next_st.dp_step
                            and i >= next_st.goal_minimals[0]) or next_st.in_queue:
                            continue
                        elif next_state == goal_index:
                            for d_e in depth_lst:
                                learned_states[d_e].in_queue = False
                            depth_lst = [next_state]
                            qobj.append(depth_lst)
                            next_st.in_queue = True
                            return qobj
                        else:
                            next_st.in_queue = True
                            depth_lst.append(next_state)
                        
                        

        if len(depth_lst) > 0:
            qobj.append(depth_lst)
        else:
            return qobj


def build_dp_for_goal(learned_states, qobj, goal_index, dp_step):



    for j in reversed(range(0, len(qobj))):
        
        for q_e in qobj[j]:
            current_state = learned_states[q_e]

            self_trans = [False]*len(current_state.actions)
            if q_e == goal_index:
                effective_depth = [0]*len(current_state.action_counts)
                current_state.goal_minimals = find_minimals(effective_depth, goal_index, self_trans)
                current_state.dp_step = dp_step
                current_state.in_queue = False
                #print('said found goal', code2ground[q_e], 'minimals', current_state.goal_minimals)
                continue
            else:
                effective_depth = [len(learned_states)+1]*len(current_state.action_counts)
                for a_ind in current_state.actions:
                    next_states = current_state.transitions[a_ind].next_states
                    goal_depth = len(learned_states)+1
                

                    if current_state.action_counts[a_ind] == 0:
                        continue

                    #print('in-state', code2ground[q_e], 'action', a_ind)
                    #print('rand action counts', 0, current_state.action_counts[a_ind]-1)

                    #print('rand seed', (q_e, a_ind, dp_step))
                    random.seed(hash((q_e, a_ind, dp_step)))
                    rand_num = random.randint(0, current_state.action_counts[a_ind]-1)

                    #print('rand draw', rand_num)

                    next_state_visits = 0
                    rand_goal_found = False
                    
                    for next_state in next_states:
                        next_state_visits += current_state.transitions[a_ind].counts[next_state]
                    
                        if learned_states[next_state].in_queue:
                            if next_state_visits > rand_num and not rand_goal_found and next_state == q_e:
                                self_trans[a_ind] = True
                            continue

                        depth_found, goal_found, _ = learned_states[next_state].goal_minimals

                        if goal_found == goal_index and next_state_visits > rand_num and not rand_goal_found:
                            goal_depth = depth_found + 1
                            rand_goal_found = True
                            
                            break

                    if goal_depth < effective_depth[a_ind] and self_trans[a_ind] == False:
                        effective_depth[a_ind] = goal_depth

                current_state.goal_minimals = find_minimals(effective_depth, goal_index, self_trans)
                current_state.dp_step = dp_step
                current_state.in_queue = False

    return learned_states[qobj[0][0]].goal_minimals


def select_random_reachable_goal(learned_states, current_state_index, dp_step, code2ground):

    #print('queue-status reach - should all be false')
    for ls in learned_states:
        ls.in_queue = False
        #print('ls-inqueue', ls.in_queue)

    qobj = queue_for_goal(learned_states, current_state_index, goal_index = len(learned_states)+1, dp_step = dp_step)

    max_depth = 25#len(qobj)
    depth_vals = []
    for d in range(1,max_depth+1):
        depth_vals.append(1.0 / math.sqrt(d))

    dsum = sum(depth_vals)    

    for d in range(0, len(depth_vals)):
        depth_vals[d] = depth_vals[d] / dsum

    rand_depth = np.random.choice(list(range(1,max_depth+1)), p=depth_vals).tolist()

    print('rand depth', rand_depth)
    print('max depth', max_depth)
    #rand_depth = random.randint(1,depths)
    #rand_depth = random.randint(1,rand_depth)

    for ls in learned_states:
        ls.in_queue = False
  
    qlst = []
    for ql in qobj[:rand_depth]:
        qlst += ql

    qel = []

    invcount_sum = 0.0

    for qe in qlst:

        counts = sum(learned_states[qe].action_counts)

        if qe in code2ground:
            qel.append((qe, code2ground[qe], counts))
            invcount_sum += 1.0/(counts+0.001)
        else:
            qel.append((qe, None))
            invcount_sum += 1.0

    p = []

    for qe in qlst:
        counts = sum(learned_states[qe].action_counts)
        if qe in code2ground:
            p.append((1.0/(counts+0.001)) / invcount_sum)
        else:
            p.append(1.0 / invcount_sum)

    print('current state', current_state_index)
    print(qel)

    print('qlst', qlst)
    print('p', p)

    goal = np.random.choice(qlst, p=p).tolist()

    print('goal-choice', goal, 'dtype', type(goal))

    #if goal in code2ground:
    #    print('picked goal', (goal, code2ground[goal]))
    #else:
    #    print('picked goal', (goal, None))

    return goal

def dijkstra_for_goal(learned_states, current_state_index, goal_index, dp_step, code2ground):

    #for si in code2ground.keys():
    #    print('in_queue-before-q', code2ground[si], learned_states[si].in_queue)

    qobj = queue_for_goal(learned_states, current_state_index, goal_index, dp_step)

    #print('queue inside of dp-goal-seek')
    #for ql in qobj:
    #    qlst = []
    #    for qe in ql:
    #        if qe in code2ground:
    #            qlst.append((qe, code2ground[qe]))
    #        else:
    #            qlst.append((qe, None))
    #    print('queue-level', qlst)

    #for si in code2ground.keys():
    #    print('in_queue-before-dp', code2ground[si], learned_states[si].in_queue)

    m = build_dp_for_goal(learned_states, qobj, goal_index, dp_step)

    #for si in code2ground.keys():
    #    print('in_queue-after-dp', code2ground[si], learned_states[si].in_queue)

    return m


def make_ls(counts, ns, na):

    state_lst = []

    min_visit_count = 1e9

    for i in range(ns):
        ls = learned_state()
        #ls.visited = False
        ls.actions = range(na)

        
        ls.action_counts = counts[i].sum(1)

        #if i in code2ground:
        #    print('state', code2ground[i], ls.action_counts)

        ls.minimals = (0, i, 0)

        ls.update_step = 1
        ls.dp_step = 0
        ls.in_queue = False

        ls.transitions = []

        for action in ls.actions:
            la = learned_action()
            la.counts = counts[i][action]

            if la.counts.min() < min_visit_count:
                min_visit_count = la.counts.min()

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


import time

def DP_goals(ls, init_state, goal_index, dp_step, code2ground):
    d,g,a = dijkstra_for_goal(ls, init_state, goal_index, dp_step=dp_step, code2ground=code2ground)

    return d,g,a

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

    #for s in range(len(ls)):
    #    ls[s].action_counts = ls[s].action_counts.numpy().tolist()

 #   print('code2ground', code2ground)

    #for code in [95]:
    #    print(code2ground[code])
    #    ns = []
    #    for a in [0,1,2,3]:
    #        ns.append(ls[code].transitions[a].next_states)

#        print('next states', ns)



    for init_state in [176]:#list(code2ground.keys()):

        #20
        for goal_index in [20]:#list(code2ground.keys()):

            print('goal_ind', goal_index)
            print("DP", code2ground[init_state], 'goal', code2ground[goal_index])
            d,g,a = DP_goals(ls,init_state, goal_index, dp_step=dp_step_use, code2ground=code2ground)

            print("Called DP on", code2ground[init_state], 'lc', init_state, 'get to goal', code2ground[goal_index])
            print('d', d)
            print('g', code2ground[g])
            print('a', a)
    
            #for s in list(code2ground.keys()):
            #    print('minimals for', code2ground[s], ls[s].minimals)

    #print("DP-2")
    #print(DP(ls,ns,na,0,2,10))

    #print('update-ls')
    #update_ls(ls, 9, 1, 9, 3)

    #print("DP-3")
    #print(DP(ls,ns,na,0,3,10))
