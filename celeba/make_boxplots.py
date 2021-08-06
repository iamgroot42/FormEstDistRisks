import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from utils import flash_utils
import numpy as np
from data_utils import SUPPORTED_PROPERTIES
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
    parser.add_argument('--multimode', action="store_true",
                        help='Plots for meta-classifier methods')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        default="Male",
                        help='name for subfolder to save/load data from')
    args = parser.parse_args()
    flash_utils(args)

    first_cat = " 0.5"

    # Set font size
    if args.multimode:
        plt.rcParams.update({'font.size': 13})
    else:
        plt.rcParams.update({'font.size': 18})

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    # Changed male ratios to female ratios by flipping data
    # Be mindful of this change when modifying/inserting data

    data = []
    columns = [
        r'Female proportion of training data ($\alpha$)',
        "Accuracy (%)",
        "Feature-extraction method"
    ]

    targets = ["0.0", "0.1", "0.2", "0.3",
               "0.4", "0.6", "0.7", "0.8", "0.9", "1.0"]

    if args.filter == "Young":
        fc_perf = [
            [80.05, 74.6, 79.2, 76.95, 78.15], # 0.0, 100
        ]

        conv_perf = [
            [67.5, 59.05, 57.9, 53.5, 60.65], # 0.0, 100
        ]

        #800 models, 100 epochs
        combined_perf = [
            [61.11, 65.13, 66.50, 70.78, 79.44],
            [57.69, 55.26, 54.96, 59.73, 55.75],
            [gpu2 (1600 above)]
            [50.93, 61.33, 52.91, 50.61, 51.40],
            [50.83, 49.69, 50.29, 53.99, 47.94],
            [gpu2],
            [gpu2],
            []
        ]

        thresholds = [
            [50.27, 50.28, 50.28], # 0.0
            [49.95, 50, 49.95],  # 0.1
            [51.95, 48, 52.06],  # 0.2
            [49.93, 49.88, 49.78],  # 0.3
            [51.4, 50.05, 53.35],  # 0.4
            [51.15, 48.65, 51.87],  # 0.6
            [50.90, 50.7, 52.06],  # 0.7
            [55.34, 51.17, 51.78],  # 0.8
            [51.91, 51.96, 50.60],  # 0.9
            [53.35, 59.55, 54.2], # 1.0
        ]

        baselines = [57.7, 55.97, 59.95, 51.55, 52.08, 47.83, 50.13, 71.48, 62.82, 86.9]

    elif args.filter == "Male":
        fc_perf = [
            [69.13, 71.12, 71.23, 70.43, 50],  # 0.0
            [61.47, 52.83, 64.32, 60.97, 63.82],  # 0.1
            [56.16, 55.83, 58.51, 44.17, 55.88],  # 0.2
            [52.87, 50.02, 49.98, 49.98, 50.02],  # 0.3
            [48.90, 48.90, 51.10, 48.9, 48.90],  # 0.4
            [47.63, 52.89, 50.54, 52.16, 48.2],  # 0.6
            [54.67, 49.98, 53.27, 49.98, 50.47],  # 0.7
            [75.54, 82.18, 82.18, 82.18, 72.41],  # 0.8
            [61.62, 56.08, 57.22, 50.02, 50.02],  # 0.9
            [64.97, 62.77, 49.98, 62.72, 60.12],  # 1.0
        ]

        conv_perf = [
            [82.92, 73.18, 91.66, 73.98, 86.86],  # 0.0
            [80.11, 77.06, 83.21, 81.67, 77.01],  # 0.1
            [68.82, 65.03, 67.09, 70.05, 64.64],  # 0.2
            [61.42, 59.12, 58.02, 59.22, 58.32],  # 0.3
            [52.94, 51.10, 52.07, 54.47, 52.17],  # 0.4
            [54.19, 54.19, 47.83, 52.79, 53.62],  # 0.6
            [56.37, 57.22, 56.57, 61.62, 63.22],  # 0.7
            [80.54, 75.86, 82.18, 82.18, 81.03],  # 0.8
            [81.91, 83.86, 81.46, 79.86, 74.71],  # 0.9
            [90.35, 94.1, 91.95, 91.6, 88.06],  # 1.0
        ]

        # all_perf = [
        #     [90.0, 92.96, 78.47, 69.0, 80.37],  # 0.0
        #     [82.81, 81.91, 77.66, 81.91, 79.31],  # 0.1
        #     [70.72, 61.07, 64.47, 62.41, 72.50],  # 0.2
        #     [60.52, 63.37, 50.02, 57.22, 56.87],  # 0.3
        #     [51.10, 49.11, 48.90, 52.32, 52.27],  # 0.4
        #     [52.16, 49.61, 47.84, 51.22, 53.20],  # 0.6
        #     [55.42, 57.67, 63.07, 56.47, 57.12],  # 0.7
        #     [63.38, 77.75, 69.62, 80.38, 59.61],  # 0.8
        #     [66.6, 67.27, 81.81, 83.16, 83.46],  # 0.9
        #     [95.2, 84.66, 91.40, 73.46, 92.2],  # 1.0
        # ]

        combined_perf = [
            [92.81, 91.56, 88.87, 84.37, 80.72],  # 0.0
            [78.86, 72.76, 74.51, 75.81, 86.81],  # 0.1
            [55.38, 66.10, 70.38, 57.28, 68.88],  # 0.2
            [56.42, 56.72, 56.52, 57.57, 53.42],  # 0.3
            [51.15, 48.75, 49.97, 51.45, 52.58],  # 0.4
            [53.88, 47.84, 52.16, 49.92, 54.56],  # 0.6
            [63.17, 64.27, 56.12, 51.67, 58.77],  # 0.7
            [77.01, 82.18, 81.77, 73.15, 73.32],  # 0.8
            [84.16, 81.46, 80.01, 78.41, 82.36],  # 0.9
            [80.1, 92.45, 93.65, 85.26, 89.26],  # 1.0
        ]

        # combined_smaller_perf = [
        #     [70.73, 83.52, 90.36, 90.06, 85.41],  # 0.0
        #     [84.71, 81.21, 75.01, 82.61, 77.46],  # 0.1
        #     [69.05, 63.02, 71.95, 70.44, 68.93],  # 0.2
        #     [59.17, 58.47, 64.57, 62.42, 62.62],  # 0.3
        #     [51.86, 62.63, 52.42, 51.97, 48.9],  # 0.4
        #     [51.22, 52.17, 51.33, 47.84, 53.67],  # 0.6
        #     [55.22, 63.32, 62.32, 55.97, 56.37],  # 0.7
        #     [67.98, 81.44, 82.68, 81.86, 82.27],  # 0.8
        #     [73.51, 84.56, 71.46, 80.11, 85.66],  # 0.9
        #     [82.81, 93.05, 89.91, 80.16, 85.71],  # 1.0
        # ]   

        # Baseline method results
        baselines = [51.15, 50.53, 50.34, 50.01,
                     50.44, 50.78, 57.26, 56.9, 46.62, 50.1]
        thresholds = [
            [54.2, 56.9, 58.05],
            [50.48, 55.57, 53.66],
            [51.63, 51.77, 52.57],
            [52.09, 53.34, 52.44],
            [51.14, 52.05, 50.97],
            [55.68, 55.41, 51.96],
            [60.68, 60.43, 60.18],
            [62.3, 60.9, 63.05],
            [66.26, 68.39, 67.63],
            [72.45, 72.4, 69.05]
        ]

    else:
        raise ValueError("Requested data not available")

    if args.multimode:
        fc_perf = fc_perf[::-1]
        for i in range(len(fc_perf)):
            for j in range(len(fc_perf[i])):
                data.append([float(targets[i]), fc_perf[i][j], "Only-FC"])

        conv_perf = conv_perf[::-1]
        for i in range(len(conv_perf)):
            for j in range(len(conv_perf[i])):
                data.append([float(targets[i]), conv_perf[i][j], "Only-Conv"])

        # all_perf = all_perf[::-1]
        # for i in range(len(all_perf)):
        #     for j in range(len(all_perf[i])):
        #         data.append([float(targets[i]), all_perf[i][j], "FC||Conv"])

        # combined_smaller_perf = combined_smaller_perf[::-1]
        # for i in range(len(combined_smaller_perf)):
        #     for j in range(len(combined_smaller_perf[i])):
        #         data.append([float(targets[i]), combined_smaller_perf[i][j], "Full-Model (fewer)"])

    combined_perf = combined_perf[::-1]
    for i in range(len(combined_perf)):
        for j in range(len(combined_perf[i])):
            data.append([float(targets[i]), combined_perf[i][j], "Full-Model"])

    df = pd.DataFrame(data, columns=columns)
    if args.multimode:
        sns_plot = sns.boxplot(
            x=columns[0], y=columns[1], data=df,
            hue=columns[2], showfliers=False)
    else:
        sns_plot = sns.boxplot(
            x=columns[0], y=columns[1], data=df,
            color='C0', showfliers=False)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint,
                color='white' if args.darkplot else 'black',
                linewidth=1.0, linestyle='--')

    baselines = baselines[::-1]
    thresholds = thresholds[::-1]

    if not args.multimode:
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
    sns_plot.figure.savefig("./meta_boxplot_%s.png" % args.filter)
