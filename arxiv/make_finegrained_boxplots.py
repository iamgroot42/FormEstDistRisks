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
    parser.add_argument('--mode', choices=["meta", "threshold"],
                        default="meta")
    args = parser.parse_args()
    flash_utils(args)

    # Set font size
    plt.rcParams.update({'font.size': 8})
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('axes', labelsize=10)

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    targets = ["9", "10", "11", "12", "14", "15", "16", "17"]

    fill_data = np.zeros((len(targets), len(targets)))
    mask = np.ones((len(targets), len(targets)), dtype=bool)
    annot_data = [[None] * len(targets) for _ in range(len(targets))]
    raw_data_meta = [
        [
            [51.5, 54.6, 52.7],
            [93.9, 76.8, 97.55],
            [100, 100, 99.8, 93.4, 94.55],
            [99.7, 99.5, 97.7],
            [99.35, 100, 99.7, 97.7],
            [97.65, 95.9, 100, 100, 100],
            [100, 100, 99.95, 100],
            [99.75, 100, 100, 100],
        ],
        [
            [56.4, 97.65, 97.55, 50.1],
            [91, 79.15, 98.15, 99.95, 92.55],
            [99.05, 94.4, 93.75],
            [91.75, 99.5, 96.1, 89.4, 100],
            [100, 100, 99.95, 100, 97.45],
            [100, 100, 97.55, 100, 100],
            [99.07, 100, 100, 99.85, 100]
        ],
        [
            [78.65, 99.1, 50.6],
            [93, 90.5, 93.75],
            [99.95, 100, 98.5, 99.95, 83.15],
            [100, 100, 92.45, 100],
            [99.7, 100, 99.7, 100],
            [99.6, 100, 98.29, 100]
        ],
        [
            [96.1, 84.1, 93.45],
            [100, 100, 100],
            [87.15, 100, 100, 100, 92.15],
            [99, 100, 100, 100, 100],
            [100, 100, 100, 100]
        ],
        [
            [87.2, 92.9, 90.5],
            [98.34, 89.45, 99.95, 99.95, 96.93, 100, 92.3, 100],
            [99.9, 99.9, 95.4],
            [100, 100, 99.9, 99.3, 99.5, 98.3, 100, 98.5, 100]
        ],
        [
            [99.9, 98.95, 100, 73.8, 99.6],
            [96.75, 100, 96, 97.15, 100],
            [100, 100, 100, 100]
        ],
        [
            [53.75, 58.4, 86.85, 60.15],
            [99.5, 74.21, 64.22, 76.92, 83.14]
        ],
        [
            [100, 70.8, 99.25, 96.04, 82.24]
        ]
    ]
    raw_data_threshold = [
        [
            [51.25, 52.25, 51.6],
            [52.8, 52.25, 54],
            [51.85, 53.55, 54.1],
            [51.1, 51.95, 50.2],
            [59.3, 56.05, 64.15],
            [67.05, 61.9, 66.5],
            [76.05, 72.4, 75.05],
            [55.04, 52.99, 53.39]
        ],
        [
            [49.85, 49.85, 49.85],
            [51.05, 50.35, 51.4],
            [50.4, 50.05, 52.95],
            [54.45, 54.75, 54.3],
            [58.35, 50.65, 56.85],
            [72.4, 65.9, 74],
            [57.75, 61.67, 53.79]
        ],
        [
            [51.85, 51.45, 55.65],
            [50.45, 50.5, 50.6],
            [54.65, 51.8, 72.1],
            [55.15, 56.2, 55.75],
            [67.05, 76.55, 64.45],
            [61.97, 60.41, 54.39]
        ],
        [
            [50.4, 50.1, 50.65],
            [52.45, 53.9, 51.65],
            [59.45, 59.8, 54.6],
            [67, 60.7, 67.55],
            [54.69, 62.42, 57.4]
        ],
        [
            [50.7, 50.95, 50.15],
            [50.8, 50.15, 50.3],
            [51.1, 51.95, 50.2],
            [53.79, 53.94, 51.43]
        ],
        [
            [57.2, 51.9, 54.7],
            [57.75, 60.8, 52.95],
            [57.85, 53.54, 60.46]
        ],
        [
            [52.4, 52.35, 52.3],
            [51.93, 51.98, 51.58]
        ],
        [
            [53.59, 53.49, 52.23]
        ]
    ]

    raw_data_loss = [
        [53.07, 54.33, 53.28, 50, 53.13, 50.07, 50, 50],
        [54.57, 58.68, 64.1, 60.5, 53.82, 50, 50],
        [54.38, 56.63, 66.55, 50.65, 50, 50],
        [53.9, 68.1, 50.38, 50, 50],
        [55.8, 50, 50, 50],
        [50, 50, 50],
        [50, 50],
        [50]
    ]

    if args.mode == "meta":
        data_use = raw_data_meta
    else:
        data_use = raw_data_threshold
        for i in range(len(targets)):
            for j in range(len(targets)-(i+1)):
                fill_data[j+i+1][i] = raw_data_loss[i][j]
                mask[j+i+1][i] = False
                annot_data[j+i+1][i] = r'%d' % (raw_data_loss[i][j])

        for i in range(len(targets)):
            fill_data[i][i] = 0
            mask[i][i] = False
            annot_data[i][i] = "N.A."

    for i in range(len(targets)):
        for j in range(len(targets)-(i+1)):
            m, s = np.mean(data_use[i][j]), np.std(data_use[i][j])
            fill_data[i][j+i+1] = m
            mask[i][j+i+1] = False
            annot_data[i][j+i+1] = r'%d $\pm$ %d' % (m, s)

    sns_plot = sns.heatmap(fill_data, xticklabels=targets, yticklabels=targets,
                           annot=annot_data, mask=mask, fmt="^",
                           vmin=50, vmax=100)
    sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
    sns_plot.figure.savefig("./arxiv_heatmap_%s.png" % (args.mode))
