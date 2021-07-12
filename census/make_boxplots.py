import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
from utils import flash_utils
import numpy as np
from model_utils import BASE_MODELS_DIR
from data_utils import PROPERTY_FOCUS, SUPPORTED_PROPERTIES
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--legend', action="store_true",
                        help='Add legend to plots')
    parser.add_argument('--novtitle', action="store_true",
                        help='Remove Y-axis label')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    args = parser.parse_args()
    flash_utils(args)

    first_cat = " 0.5"

    # Set font size
    plt.rcParams.update({'font.size': 18})

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    data = []
    columns = [
        r'%s proportion of training data ($\alpha$)' % PROPERTY_FOCUS[args.filter],
        "Accuracy (%)"
    ]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % first_cat)
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % first_cat)

    targets = ["0.0", "0.1", "0.2", "0.3",
               "0.4", "0.6", "0.7", "0.8", "0.9", "1.0"]

    if args.filter == "sex":
        raw_data = [
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            [75.1, 78.23, 71.83, 68.567, 75.73, 75.2, 74.767, 74.467, 67.3, 71.03],
            [63.836, 56.7676, 63.2676, 57.567, 50.6764, 59.234, 60.367, 57.13, 56.53, 54.367],
            [55.067, 52.734, 56.13, 52.5, 50.2, 55.734, 52.7676, 51.03, 52.867, 50.367],
            [47.3, 50.867, 46.1674, 51.2676, 47.13, 49.6, 49.336, 51.467, 47.1, 53.367],
            [52.734, 55.234, 53.53, 52.1, 59.336, 55.2676, 58.0, 57.7, 61.93, 51.43],
            [64.667, 62.567, 61.63, 62.6, 63.7676, 63.93, 64.267, 58.43, 63.7, 63.067],
            [67.73, 68.067, 67.3, 66.9, 66.83, 67.067, 67.367, 67.8, 67.7, 68.2],
            [71.467, 72.1, 70.53, 70.23, 71.0, 70.867, 70.367, 70.934, 70.3, 70.8],
            [82.067, 99.73, 99.0, 81.267, 80.1, 77.867, 78.634, 77.73, 78.8, 82.367]
        ]
    else:
        raw_data = [
            [100.0, 99.95, 100.0, 100.0, 100.0, 100.0, 99.95, 100.0, 100.0, 100.0],
            [56.25, 53.75, 56.6, 56.3, 55.55, 57.55, 55.8, 55.95, 56.3, 58.7],
            [55.1, 53.8, 54.65, 53.0, 53.4, 53.7, 53.65, 54.65, 54.7, 52.2],
            [54.45 , 53.1 , 52.85 , 52.35 , 51.75 , 53.05 , 51.85 , 53.25 , 52.25 , 51.8],
            [50.25 , 51.6 , 49.2 , 49.6 , 50.05 , 49.7 , 49.35 , 50.2 , 50.4 , 50.85],
            [51.15, 49.3, 51.15, 48.7, 49.7, 49.55, 50.35, 50.3, 50.25, 51.1],
            [51.95, 51.8, 49.45, 49.9, 50.9, 51.7, 53.9, 49.45, 49.55, 49.25],
            [54.95, 52.4, 56.3, 49.9, 51.1, 56.7, 54.9, 50.4, 54.3, 49.5],
            [51.26, 51.21, 51.56, 53.39, 51.41, 54.13, 49.73, 51.81, 50.07, 53.19],
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
        ]

    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([float(targets[i]), raw_data[i][j]])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='white' if args.darkplot else 'black',
                linewidth=1.0, linestyle='--')

    # Map range to numbers to be plotted
    if args.filter == 'race':
        # For race
        baselines = [57, 50, 50, 50, 50, 50, 50.35, 50.05, 50.05, 50.05]
        thresholds = [
            [80.10, 79.70, 72.90, 78.65, 78.20, 72.10, 78.50, 77.45, 75.25, 79.90],
            [69.15, 70.70, 63.50, 63.75, 63.30, 56.80, 62.95, 72.50, 68.95, 68.25],
            [66.15, 62.00, 64.85, 60.95, 61.80, 66.10, 63.65, 64.20, 63.35, 60.85],
            [60.85, 61.00, 58.55, 67.60, 60.20, 52.85, 62.50, 65.05, 64.25, 57.75],
            [52.45, 60.75, 57.75, 58.20, 57.95, 57.85, 58.70, 58.60, 57.40, 59.60],
            [53.70, 54.55, 56.25, 57.20, 58.30, 55.15, 56.70, 53.65, 55.20, 54.65],
            [58.70, 63.05, 62.85, 60.30, 65.05, 63.70, 62.00, 62.25, 65.95, 65.80],
            [65.70, 58.70, 63.80, 62.20, 53.80, 59.30, 60.70, 56.85, 55.85, 58.75],
            [55.15, 51.35, 50.70, 54.40, 54.75, 51.15, 51.65, 52.55, 51.05, 50.35],
            [50.95, 59.00, 58.00, 59.00, 56.85, 59.00, 56.30, 59.70, 59.00, 59.70]
        ]
    else:
        # For sex
        baselines = [70.5, 49.7, 50, 50.7, 50.4, 50, 52, 51.8, 50.3, 50.1]
        thresholds = [
            [64.05, 61.60, 63.45, 63.60, 59.00, 61.65, 64.35, 63.35, 62.25, 63.55],
            [55.35, 60.35, 65.00, 59.65, 61.20, 54.35, 60.90, 56.70, 60.55, 57.60],
            [65.45, 62.85, 55.05, 61.35, 59.15, 60.00, 59.40, 66.25, 60.90, 61.40],
            [60.95, 67.40, 61.25, 64.10, 66.60, 58.00, 62.40, 64.60, 62.60, 66.00],
            [61.85, 59.10, 54.95, 56.65, 60.55, 61.95, 60.80, 61.20, 61.75, 59.90],
            [64.05, 63.65, 63.90, 62.35, 64.20, 62.90, 64.15, 61.50, 62.85, 63.15],
            [68.35, 76.40, 69.65, 74.05, 76.30, 76.20, 74.85, 75.30, 75.80, 75.90],
            [76.65, 77.90, 77.15, 74.05, 78.50, 74.90, 80.55, 82.80, 74.10, 74.00],
            [70.80, 71.55, 70.25, 68.50, 70.30, 65.45, 70.15, 70.45, 66.00, 71.35],
            [60.30, 62.20, 61.30, 53.80, 57.15, 62.75, 62.50, 62.00, 60.40, 61.85]
        ]

    # Plot baselines
    targets_scaled = range(int((upper - lower)))
    plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    # Plot numbers for threshold-based accuracy
    means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
    plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

    # Custom legend
    if args.legend:
        meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
        baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{baseline}$')
        threshold_patch = mpatches.Patch(color='C2', label=r'$Acc_{threshold}$')
        plt.legend(handles=[meta_patch, baseline_patch, threshold_patch])

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Make sure axis label not cut off
    plt.tight_layout()

    # Save plot
    sns_plot.figure.savefig("./meta_boxplot.png")
