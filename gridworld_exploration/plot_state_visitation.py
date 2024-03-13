
import os, sys, time
import numpy as np

import random
import os
import time
import json
import argparse


# Do not modify this cell!

def plot_state_visitations(data, env_iteration, method):
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

    print ("saving heatmap plot")
    plt.savefig("./heatmap_plots" + "/" +  method + "_" + "heatmap_%s.png" % env_iteration)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='plot heatmaps', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--iter', type=int, default=2) #2000
    parser.add_argument('--data', type=str, choices=[ 'random', 'goal_seek'], default='goal_seek')

    args = parser.parse_args()

    goal_seek_data = np.load('state_counts/goalseek_visit_count.npy')
    random_seek_data = np.load('state_counts/random_visit_count.npy')


    goal_data = goal_seek_data[args.iter, :]
    random_data = random_seek_data[args.iter, :]

    # if args.data == 'goal_seek':
    #     goal_data = goal_seek_data[args.iter, :]
    # elif args.data == 'random':
    #     random_data = random_data[args.iter, :]
    # else:
    #     raise Exception()

    plot_state_visitations(goal_data, args.iter, "goak_seek")
    plot_state_visitations(random_data, args.iter, "random_data")







