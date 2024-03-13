
import os, sys, time
import torch
from torch.autograd import Variable
import numpy as np

import random
import os
import time
import json
import torch



# Do not modify this cell!

def plot_state_visitations(data,  size, folder, env_iteration):
    import matplotlib.pyplot as plt
    i = 0
    wall_ends = [None,-1]

    state_visits = data
    average_state_visits = np.mean(state_visits, axis=0)

    grid_state_visits = np.rot90(   state_visits.reshape(  (11,11)   ).T    )

    grid_state_visits[0:11, 0] = np.nan
    grid_state_visits[0:11, 10] = np.nan

    grid_state_visits[0, 0:11] = np.nan
    grid_state_visits[10, 0:11] = np.nan
    

    grid_state_visits[5, 1] = np.nan
    grid_state_visits[5, 3:8] = np.nan
    grid_state_visits[5, 9] = np.nan


    grid_state_visits[3:8, 5] = np.nan
    grid_state_visits[1, 5] = np.nan
    grid_state_visits[9, 5] = np.nan

    plt.pcolormesh(grid_state_visits, edgecolors='black', linewidth=1, cmap='viridis', vmin=0, vmax=20, shading='flat')
    # plt.pcolormesh(grid_state_visits, edgecolors='black', linewidth=1, cmap='viridis')
    # plt.pcolormesh(grid_state_visits, edgecolors='red', linewidth=1, cmap='viridis')
    plt.text(2+0.5, 9+0.5, 'S', horizontalalignment='center', verticalalignment='center')
    # plt.text(8+0.5, 5+0.5, 'G', horizontalalignment='center', verticalalignment='center')
    plt.axis('off')
    cm = plt.get_cmap()
    # cm.set_bad('gray')
    cm.set_bad('white')

    plt.subplots_adjust(bottom=0.0, right=0.7, top=1.0)
    cax = plt.axes([1., 0.0, 0.075, 1.])
    cbar = plt.colorbar(cax=cax)

    plt.savefig(folder + "/" + "heatmap_%s.png" % env_iteration)









# Code based on: 
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

create_folder = lambda f: [ os.makedirs(f) if not os.path.exists(f) else False ]


class Logger(object):
      def __init__(self, args, experiment_name,  environment_name='',  exo_noise='' , grid_size = '', use_policy ='', start_state='', reset='', folder='./results'):
            """
            Original: Original implementation of the algorithms
            HDR: Used Qhat
            HDR_RG: Uses Qhat where graph is retained
            DR: Uses Qhat-Vhat
            DR_RG: Uses
            """
            self.rewards = []
              
            self.save_folder = os.path.join(folder, experiment_name, environment_name, exo_noise, grid_size, use_policy, start_state, reset)

            create_folder(self.save_folder)


      def record_classifier_accuracy(self, acc):
            self.acc = acc

      def record_all_classifier_accuracy(self, all_acc):
            self.all_acc = all_acc


      def save_state_visits(self, state_visits, env_iteration):
            self.state_visits = state_visits
            np.save(os.path.join(self.save_folder, "state_visit_count.npy"), self.state_visits)


      def save(self):
            np.save(os.path.join(self.save_folder, "classifier_acc.npy"), self.acc)
            np.save(os.path.join(self.save_folder, "all_classifier_acc.npy"), self.all_acc)

      def save_args(self, args):
            """
            Save the command line arguments
            """
            with open(os.path.join(self.save_folder, 'params.json'), 'w') as f:
                  json.dump(dict(args._get_kwargs()), f)






def states_pos_to_int(test_s0, test_s1, rows, cols):
    test_s0_positions = np.zeros(test_s0.shape[0])
    test_s1_positions = np.zeros(test_s1.shape[0])
    gridtostate = {}
    statetogrid = {}
    count = 0
    for y in range(rows):
        for x in range(cols):
            for j in range(rows):
                for i in range(cols):
                    gridtostate[(x, y, i, j)] = count
                    statetogrid[count] = (x, y, i, j)
                    count += 1

    for i in range(test_s0.shape[0]):
        test_s0_positions[i] = gridtostate[tuple(test_s0[i])]
        test_s1_positions[i] = gridtostate[tuple(test_s1[i])]

    return test_s0_positions, test_s1_positions, statetogrid


# test_s0_positions, test_s1_positions, statetogrid = states_pos_to_int(test_s0, test_s1, rows, cols)


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def apply(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(2)
        w = img.size(3)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = int(np.clip(y - self.length / 2, 0, h))
            y2 = int(np.clip(y + self.length / 2, 0, h))
            x1 = int(np.clip(x - self.length / 2, 0, w))
            x2 = int(np.clip(x + self.length / 2, 0, w))

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()
        img = img * mask

        return img







