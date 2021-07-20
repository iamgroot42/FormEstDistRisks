import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import flash_utils
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    args = parser.parse_args()
    flash_utils(args)

    # Set font size
    plt.rcParams.update({'font.size': 6})

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    targets = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]

    fill_data = np.zeros((len(targets), len(targets)))
    mask = np.ones((len(targets), len(targets)), dtype=bool)
    annot_data = [[None] * len(targets) for _ in range(len(targets))]
    raw_data = [
        [
            [50.0, 86.55, 87.55, 85.45, 79.6, 75.5, 87.1, 85.05, 86.5, 88.35],
            [98.8, 97.7, 98.0, 99.2, 99.7, 96.65, 99.6, 98.3, 98.2, 98.6],
            [99.55, 99.7, 99.7, 99.55, 99.9, 99.95, 99.7, 98.05, 99.95, 99.8],
            [99.9, 99.95, 99.95, 99.85, 99.95, 99.9, 99.85, 99.9, 99.9, 99.95],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 100.0],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        ],
        [
            [86.65, 88.2, 83.7, 61.0, 85.25, 76.05, 54.1, 85.15, 77.2, 63.4],
            [94.7, 95.7, 93.45, 92.4, 95.35, 97.25, 96.6, 94.25, 95.9, 98.0],
            [50.0, 99.1, 98.15, 99.5, 99.7, 99.2, 95.75, 99.15, 98.65, 98.85],
            [100.0, 99.9, 100.0, 99.95, 99.9, 99.85, 99.9, 99.95, 99.95, 99.95],
            [100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 99.9, 100.0, 100.0, 100.0]
        ],
        [
            [70.95, 63.85, 63.25, 69.75, 59.2, 66.3, 74.6, 63.4, 69.45, 65.55],
            [79.8, 93.55, 88.1, 79.35, 93.25, 95.5, 83.1, 84.35, 95.55, 94.85],
            [97.95, 98.65, 97.35, 93.8, 99.25, 97.85, 92.8, 91.25, 98.1, 99.55],
            [98.4, 99.95, 99.5, 99.65, 100.0, 100.0, 100.0, 100.0, 100.0, 99.95]
        ],
        [
            [50.0, 58.9, 61.55, 67.0, 58.55, 62.15, 59.85, 58.75, 62.65, 65.65],
            [70.2, 90.3, 75.7, 83.75, 86.0, 89.45, 85.1, 83.7, 78.35, 72.35],
            [99.45, 99.2, 98.75, 98.4, 97.1, 92.05, 96.0, 96.75, 96.6, 97.35]
        ],
        [
            [50.25, 56.6, 57.55, 60.75, 53.8, 57.45, 56.35, 54.2, 53.35, 59.1],
            [76.5, 77.65, 89.95, 65.15, 66.6, 74.0, 76.0, 69.8, 80.25, 77.9]
        ],
        [
            [58.55, 54.9, 57.55, 56.65, 52.45, 54.45, 51.2, 50.75, 53.35, 50.95]
        ]
    ]

    for i in range(len(targets)):
        for j in range(len(targets)-(i+1)):
            m, s = np.mean(raw_data[i][j]), np.std(raw_data[i][j])
            fill_data[i][j+i+1] = m
            mask[i][j+i+1] = False
            annot_data[i][j+i+1] = r'%d $\pm$ %d' % (m, s)

    sns_plot = sns.heatmap(fill_data, xticklabels=targets, yticklabels=targets,
                           annot=annot_data, mask=mask, fmt="^") #, vmin=0, vmax=100)
    sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
    sns_plot.figure.savefig("./meta_heatmap.png")
