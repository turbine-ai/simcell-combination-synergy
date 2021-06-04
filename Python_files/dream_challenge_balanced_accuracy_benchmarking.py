import os
import numpy as np
import csv
import argparse
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pylab import rcParams

rcParams['figure.figsize'] = 14, 10


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    '''
    creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def parse_antagonism_values(file_path: str):
    with open(file_path, mode='r') as input_ant:
        input_ant = csv.DictReader(input_ant, delimiter=',')
        ant_dict = dict()
        for row in input_ant:
            cell_line = row['Cell_line']
            drug_1 = row['Drug_1']
            drug_2 = row['Drug_2']
            max_ant = row['Max_syn']
            max_ant_subgrid = row['Max_syn_ic50']

            combiname = ':'.join(sorted([drug_1, drug_2]))

            if cell_line not in ant_dict:
                ant_dict[cell_line] = {}

            if combiname not in ant_dict[cell_line]:
                ant_dict[cell_line][combiname] = {}

            if str(max_ant) != 'nan':
                ant_dict[cell_line][combiname]['Max_ant'] = float(max_ant) * 100

            if str(max_ant_subgrid) != 'nan':
                ant_dict[cell_line][combiname]['Max_ant_subgrid'] = float(max_ant_subgrid) * 100

    return ant_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_antagonism_table_path", type=str,
                        help='Path of combination postprocess antagonism results.',
                        default='../data/processed/combi_metrics_ant_astra_full.txt'
                        )

    parser.add_argument("--input_dream_vs_turbine_synergy_scores_table", type=str,
                        help='Path of table containing Dream challenge benchmark and Turbine synergy scores',
                        default="../data/processed/cl_comp_intersect-Combimetrics_DREAM_unique.tsv")

    parser.add_argument("--output_folder_path", type=str, help='Path of folder to save results.',
                        default="../results/")

    params = parser.parse_args()

    ant_dict = parse_antagonism_values(params.input_antagonism_table_path)

    combi_dream = pd.read_csv(params.input_dream_vs_turbine_synergy_scores_table, sep='\t')

    split = combi_dream['cl_comb_pairs'].str.split('_', n=1, expand=True)

    combi_dream['combiname'] = split[1]
    combi_dream['cell_line'] = split[0]

    cell_line = combi_dream['cell_line'].values
    combiname = combi_dream['combiname'].values

    bliss = combi_dream['Bliss_max'].values
    bliss_subgrid = combi_dream['Bliss_max_IC50'].values
    dream_scores = combi_dream['DREAM_synergy_score'].values

    new_bliss = []
    new_bliss_subgrid = []

    for i in range(0, len(bliss)):

        splitted = str(combiname[i]).split(':')

        name = ':'.join(list(sorted([splitted[0], splitted[1]])))

        if bliss[i] == 0:
            try:
                print(cell_line[i], name, bliss[i] + -1 * float(ant_dict[cell_line[i]][name]['Max_ant']))

                new_bliss.append(float(bliss[i] + -1 * float(ant_dict[cell_line[i]][name]['Max_ant'])))

            except KeyError:
                new_bliss.append(float(bliss[i]))
        else:
            new_bliss.append(float(bliss[i]))

        if bliss_subgrid[i] == 0:
            try:
                new_bliss_subgrid.append(
                    float(bliss_subgrid[i] + -1 * float(ant_dict[cell_line[i]][name]['Max_ant_subgrid'])))
            except KeyError:
                new_bliss_subgrid.append(float(bliss_subgrid[i]))
        else:
            new_bliss_subgrid.append(float(bliss_subgrid[i]))

    thrshlds = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])

    balanced_accuracies = []

    for thrshld in thrshlds:
        for thrshld_2 in thrshlds:
            Y = 1 * (new_bliss > thrshld)
            Y_dream = 1 * (dream_scores > thrshld_2)

            balanced_accuracies.append(round(balanced_accuracy_score(Y_dream, Y), 3))

            if balanced_accuracy_score(Y_dream, Y) > 0.6:
                print('Our threshold: {}, Dream threshold: {}, Balanced Accuracy: {}'.
                      format(str(thrshld), str(thrshld_2), str(balanced_accuracy_score(Y_dream, Y) * 100)))

    balanced_accuracies = np.array(balanced_accuracies)
    balanced_accuracies = balanced_accuracies.reshape((9, 9))

    fig, ax = plt.subplots()

    colorscale = [[0.0, 'rgb(255, 255, 255)'],
                  [1.0, 'rgb(255,165,160)']]

    custom_cmap = get_continuous_cmap(hex_list=["#FF8782", "#5AC8BE"])

    img = ax.matshow(balanced_accuracies, cmap=custom_cmap)
    colorbar = plt.colorbar(img)
    colorbar.set_label('Balanced accuracy', size=15)
    colorbar.ax.tick_params(labelsize=13)

    for (i, j), z in np.ndenumerate(balanced_accuracies):
        if i == 0:
            i += 0.15
        if i == 8:
            i -= 0.2
        ax.text(j, i, '{}'.format(z), ha='center', va='top', fontdict={'size': 12})

    rcParams.update({'font.size': 16})
    plt.xticks(np.arange(9), list(thrshlds), fontsize=13)
    plt.yticks(np.arange(9), list(thrshlds), fontsize=13)
    plt.xlabel('Dream threshold', fontdict={'size': 17})
    plt.ylabel('Turbine threshold', fontdict={'size': 17})
    plt.title('Turbine balanced accuracy across various synergy threshold', fontdict={'size': 18})
    plt.savefig(os.path.join(params.output_folder_path, 'bliss_max_DREAM_comparison_bal_acc.png'), dpi=500)
