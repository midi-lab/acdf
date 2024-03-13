import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import argparse
import glob
import logging
import os
from numpy import genfromtxt 
sns.set_context("paper", font_scale=2.0)


def main_plot(list_of_data, error_type='', smoothing_window=1,
              file_name='figure', saving_folder='', labels=None, title="Reward Plot",
              x_label='Iterations',
              y_label='Rewards', 
              types='same_method'):

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.xaxis.get_offset_text().set_fontsize(20)
    axis_font = {'fontname':'Arial', 'size':'20'}

    # get a list of colors here.
    colors = sns.color_palette('colorblind', n_colors=len(list_of_data))
    y_values = []

    for data, label, color in zip(list_of_data, labels, colors):

        x_values = np.arange( len(list_of_data) )  
        if types =='same_method':
            x_values_strings = [ '6x6', '9x9', '12x12'  ]
            x_label = "Grid Sizes"
        elif types == 'diff_method':
            x_values_strings = [ 'DP-Goal', 'VI-Goal', 'VI', 'Random'  ]
            x_label = "Policy Selection"
        y_values = np.array(list_of_data)
        plt.xticks(x_values, x_values_strings)

        plt.plot(x_values, y_values, color=color, linewidth=1.5,  label=label)

    ax.legend(loc='lower right', prop={'size' : 28})
    ax.set_xlabel(x_label,**axis_font)
    ax.set_ylabel(y_label, **axis_font)
    ax.set_title(title, **axis_font)

    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)
    fig.savefig('{}/{}.png'.format(saving_folder, error_type + file_name))
    
    return fig


def get_paths(glob_path):
    return glob.glob(glob_path)

def load_and_stack_npy(glob_path):
    path_to_csv = get_paths(glob_path)

    for path in path_to_csv:
        print ("Path", path)
        data = genfromtxt(path, delimiter=',')

        learned_state_entropy = data[0, 1] 
        ground_state_entropy = data[1, 1] 
        SSS = data[2, 1] * 100
        DSM = data[3, 1] * 100
        Dynamics = data[4, 1] * 100
 
    return learned_state_entropy, ground_state_entropy, SSS, DSM, Dynamics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", help="Glob paths to the folder with data", nargs='+')
    parser.add_argument('--labels', default=[], nargs='+')
    parser.add_argument('--title', default='Reward Plot')
    parser.add_argument('--xlabel', default='Episodes')
    parser.add_argument('--ylabel', default='Rewards')
    parser.add_argument('--smoothing_window', default=1, type=int)
    parser.add_argument('--saving_folder', default='plot_analysis', type=str)
    parser.add_argument('--file_name', default='Result Plot')
    parser.add_argument('--type', default='same_method')

    args = parser.parse_args()

    if len(args.labels) < len(args.paths):
        args.labels.extend([''] * (len(args.paths)-len(args.labels)))

    print('Number of paths provided: {}'.format(len(args.paths)))
    dsm_data = []
    sss_data = []
    learned_data = []
    ground_data = []
    dynamics_data = []
    for path in args.paths:

        learned_state_entropy, ground_state_entropy, SSS, DSM, Dynamics = load_and_stack_npy(path)

        dsm_data.append(DSM)
        sss_data.append(SSS)
        learned_data.append(learned_state_entropy)
        ground_data.append(ground_state_entropy)
        dynamics_data.append(Dynamics)

    datas = [ dsm_data, sss_data, learned_data, ground_data, dynamics_data ] 
    datas_dict = { "dsm":dsm_data, "sss" : sss_data, "learned" : learned_data, "ground" : ground_data, "dynamics" : dynamics_data  }

    for data in sorted(datas_dict.items()):
        print ("data", data)
        main_plot(data[1],
                  error_type=data[0], 
                  smoothing_window=args.smoothing_window,
                  file_name=args.file_name.replace(' ', ''),
                  saving_folder=args.saving_folder,
                  labels=args.labels,
                  title= 'SpiralMaze',
                  x_label=args.xlabel,
                  y_label= data[0] + '- ' + 'Error (%)', 
                  types=args.type)

if __name__ == '__main__':
    main()