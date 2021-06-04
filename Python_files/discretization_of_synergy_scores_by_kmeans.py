import csv
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import os
import argparse
from sklearn.cluster import KMeans
from pylab import rcParams
rcParams['figure.figsize'] = 14, 10
matplotlib.rcParams.update({'font.size': 17})


def get_array_to_discretize(path: str, delimiter=",") -> np.array:

    with open(path, mode='r') as input_csv:
        synergies = csv.reader(input_csv, delimiter=delimiter)
        max_syn = []

        for idx, elem in enumerate(synergies):
            if idx == 0:
                continue
            elif str(elem[7]) == 'nan':
                elem[7] = 0

            max_syn.append(elem[7])

        return np.array(max_syn, dtype=float)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_synergy_table_path", type=str, help='Path of combination postprocess results.',
                        default='../data/processed/combi_metrics_syn_astra_full.txt'
                        )
    parser.add_argument("--output_folder_path", type=str, help='Path of folder to save results.',
                        default="../results/")

    params = parser.parse_args()
    
    synergy_array = get_array_to_discretize(params.input_synergy_table_path)
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(synergy_array.reshape(-1, 1))
    labels = np.array(kmeans.labels_)
    mask_0 = (labels == 0)
    mask_1 = (labels == 2)
    mask_2 = (labels == 1)

    print('Cluster 0: ', np.average(synergy_array[mask_0]), np.std(synergy_array[mask_0]), np.amin(synergy_array[mask_0]),
          np.amax(synergy_array[mask_0]))
    print('Cluster 1: ', np.average(synergy_array[mask_1]), np.std(synergy_array[mask_1]), np.amin(synergy_array[mask_1]),
          np.amax(synergy_array[mask_1]))
    print('Cluster 2: ', np.average(synergy_array[mask_2]), np.std(synergy_array[mask_2]), np.amin(synergy_array[mask_2]),
          np.amax(synergy_array[mask_2]))

    sns.distplot(synergy_array[mask_0], hist=False, kde_kws={"color": "#FF8782", "lw": 3,
                                                     "label": "Cluster 0 = weak/no synergy", "shade": True})
    sns.distplot(synergy_array[mask_1], hist=False, kde_kws={"color": "#5AC8BE", "lw": 3,
                                                     "label": "Cluster 1 = moderate synergy", "shade": True})
    sns.distplot(synergy_array[mask_2], hist=False, kde_kws={"color": "#D7413C", "lw": 3,
                                                     "label": "Cluster 2 = strong synergy", "shade": True})

    plt.axvline(x=np.amax(synergy_array[mask_0]), ymax=.65, label='First cut-off value = {}'.format(
        round(np.amax(synergy_array[mask_0]),2)), c='orange')
    plt.axvline(x=np.amin(synergy_array[mask_2]), ymax=.65, label='Second cut-off value = {}'.format(
        round(np.amin(synergy_array[mask_2]),2)), c='black')

    plt.legend(loc='best')
    plt.xlabel('Bliss max ic50', fontdict={'size': 17})
    plt.ylabel('Probability density', fontdict={'size': 17})
    plt.title('Determine cut-off values of Bliss max ic50 with k-means clustering (k=3)')
    plt.tight_layout()
    plt.savefig(os.path.join(params.output_folder_path, "synergy_values_discretization.png"), dpi=500)
