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
    raw_data_threshold = [
        [
            [50.85, 52.35, 50.4, 52.5, 52.3],
            [51, 57.6, 55.35, 50.4, 51.2],
            [58.10, 68.35, 57.50, 63.55, 60.10, 63.05, 62.10, 57.75, 63.60, 65.20],
            [72.75, 66.95, 80.55, 64.4, 62.95],
            [81.55, 70.5, 82.2, 78.25, 75.25],
            [88.8, 86.8, 80.75, 82.1, 89.1]
        ],
        [
            [54.75, 52.8, 50.4, 54.75, 51.1],
            [58.05, 59.40, 51.30, 59.25, 54.30, 56.55, 54.80, 58.95, 55.50, 57.85],
            [62.95, 66.2, 64.65, 63.3, 60.05],
            [67.75, 72.15, 71, 69.15, 69.05],
            [83.8, 69.45, 66.05, 80.05, 77.15]
        ],
        [
            [55.55, 51.95, 53.15, 51.60, 52.85, 51.50, 50.35, 52.50, 51.25, 50.60],
            [56.35, 63, 55.25, 57.75, 58],
            [67.1, 63.55, 64.25, 64.85, 66.65],
            [67.75, 72, 78, 70.35, 67.55]
        ],
        [
            [50.10, 52.10, 50.45, 51.45, 50.80, 53.75, 50.45, 53.00, 50.35, 53.95],
            [52.75, 56.20, 51.60, 51.50, 52.75, 58.70, 53.70, 55.60, 53.25, 54.10],
            [55.85, 64.45, 58.40, 56.40, 59.05, 64.65, 50.45, 57.25, 54.60, 69.95]
        ],
        [
            [52.45, 53.7, 52.4, 50.45, 50.3],
            [58.75, 58.25, 62.7, 51.4, 60.35]
        ],
        [
            [54.25, 51.15, 56.6, 52.25, 55.5]
        ]
    ]

    raw_data_loss = [
        [53.5, 60.05, 64.9, 80.8, 86.35, 84.6],
        [50.4, 64.3, 71.55, 69.8, 87.35],
        [59.0, 61.5, 65.65, 73.65],
        [57.0, 56.8, 66.9],
        [50.45, 65.1],
        [54.8]
    ]

    if args.mode == "meta":
        data_use = raw_data
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

    #track max values
    from utils import get_n_effective, bound
    max_values = np.zeros_like(fill_data)
    eff_vals = np.zeros_like(fill_data)
    for i in range(len(targets)):
        for j in range(len(targets)-(i+1)):
            max_values[i][j+i+1] = max(raw_data_loss[i]
                                       [j], max(raw_data_threshold[i][j]))
            max_values[i][j+i+1] = max(max_values[i]
                                       [j], max(raw_data[i][j]))
            n_eff = get_n_effective(
                max_values[i][j+i+1] / 100, float(targets[i]), float(targets[j]))
            eff_vals[i][j] = np.abs(n_eff)
            print(i, j, bound(float(targets[i]), float(targets[j]), n_eff))
    print(eff_vals)

    sns_plot = sns.heatmap(fill_data, xticklabels=targets, yticklabels=targets,
                           annot=annot_data, mask=mask, fmt="^",
                           vmin=50, vmax=100)
    sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
    sns_plot.figure.savefig("./boneage_heatmap_%s.png" % (args.mode))
