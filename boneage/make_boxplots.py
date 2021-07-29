import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
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
    args = parser.parse_args()

    first_cat = " 0.5"

    if args.darkplot:
        # Set dark background style
        plt.style.use('dark_background')

    # Set font size
    plt.rcParams.update({'font.size': 18})

    data = []
    columns = [
        r'Female proportion of training data ($\alpha$)',
        "Accuracy (%)"
    ]

    categories = ["0.2", "0.3", "0.4", "0.6", "0.7", "0.8"]
    raw_data = [
        [99.55, 99.7, 99.7, 99.55, 99.9, 99.95, 99.7, 98.05, 99.95, 99.8],
        [94.7, 95.7, 93.45, 92.4, 95.35, 97.25, 96.6, 94.25, 95.9, 98.0],
        [70.95, 63.85, 63.25, 69.75, 59.2, 66.3, 74.6, 63.4, 69.45, 65.55],
        [50.0, 58.9, 61.55, 67.0, 58.55, 62.15, 59.85, 58.75, 62.65, 65.65],
        [70.2, 90.3, 75.7, 83.75, 86.0, 89.45, 85.1, 83.7, 78.35, 72.35],
        [99.45, 99.2, 98.75, 98.4, 97.1, 92.05, 96.0, 96.75, 96.6, 97.35]
    ]
    for i in range(len(raw_data)):
        for j in range(len(raw_data[i])):
            data.append([categories[i], raw_data[i][j]])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='white' if args.darkplot else 'black',
                linewidth=1.0, linestyle='--')

    # Map range to numbers to be plotted
    baselines = [64.9, 64.3, 59.0, 57.0, 56.8, 66.9]
    targets_scaled = range(int((upper - lower)))
    plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    # Plot numbers for threshold-based accuracy
    thresholds = [
        [58.10, 68.35, 57.50, 63.55, 60.10, 63.05, 62.10, 57.75, 63.60, 65.20],
        [58.05, 59.40, 51.30, 59.25, 54.30, 56.55, 54.80, 58.95, 55.50, 57.85],
        [55.55, 51.95, 53.15, 51.60, 52.85, 51.50, 50.35, 52.50, 51.25, 50.60],
        [50.10, 52.10, 50.45, 51.45, 50.80, 53.75, 50.45, 53.00, 50.35, 53.95],
        [52.75, 56.20, 51.60, 51.50, 52.75, 58.70, 53.70, 55.60, 53.25, 54.10],
        [55.85, 64.45, 58.40, 56.40, 59.05, 64.65, 50.45, 57.25, 54.60, 69.95],
    ]
    means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
    plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

    if args.legend:
        # Custom legend
        meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
        baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{baseline}$')
        threshold_patch = mpatches.Patch(color='C2', label=r'$Acc_{threshold}$')
        plt.legend(handles=[meta_patch, baseline_patch, threshold_patch])

    # Make sure axis label not cut off
    plt.tight_layout()

    sns_plot.figure.savefig("./meta_boxplot.png")
