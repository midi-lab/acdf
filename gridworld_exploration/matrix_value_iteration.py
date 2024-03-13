import torch
import numpy as np


def value_iteration(transition, num_states, init_state, reward, V=None,
                    max_iter=1000, alpha0=1.0, gamma=0.99):
    """
    :param transition: a numpy array of size num_states x num_actions x num_states
    :param num_states: the number of states
    :param init_state: the initial state
    :param reward: a numpy array of size num_states
    :param V: the previous value of V (if provided) which can be used for initialization
    :param max_iter: maximum number of iteration
    :param alpha0: initial value of decay parameter controlling convergence
    :param gamma: discount factor
    :return: tuple(policy, V)
            action: action taken by the learned optimal value for init_state
            V: the optimal value determined by the value function
    """

    if V is None:
        V = np.zeros(num_states).astype(np.float32)

    if torch.cuda.is_available():
        V = torch.from_numpy(V).cuda()
        transition = torch.from_numpy(transition).cuda()
        reward = torch.from_numpy(reward).cuda()

    for it in range(1, max_iter + 1):

        alpha = alpha0 / float(it)

        # Q(s, a) = r(s) + \sum_{s'} T(s' | s, a) V(s')
        Q = reward[None, :] + gamma * (transition * V[:, None, None]).sum(0)        # num_actions x num_states

        # V(s) = (1 - alpha) * V(s) + alpha * {r(s) + gamma * \max_{a} \sum_{s'} T(s' | s, a) V(s')}
        V = (1 - alpha) * V + alpha * Q.max(0)                                      # num_states

    actions = Q.argmax(0)                                                           # num_states

    if torch.cuda.is_available():
        return int(actions[init_state].item()), float(V[init_state].item())
    else:
        return int(actions[init_state]), float(V[init_state])
