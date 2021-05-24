import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from model_utils import BASE_MODELS_DIR
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


if __name__ == "__main__":
    first_cat = " 0.5"

    # Set dark background style
    # plt.style.use('dark_background')

    data = []
    columns = [
        r'Ratio of females ($\alpha$) in dataset that model is trained on',
        "Meta-classifier accuracy (%) differentiating between models"
    ]

    batch_size = 1000
    num_train = 700
    n_tries = 5

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/0.5/")
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/0.5/")

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
    sns_plot.set(ylim=(50, 100))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    print(upper, lower)
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='black',
                linewidth=1.0, linestyle='--')

    # Map range to numbers to be plotted
    baselines = [64.9, 64.3, 59.0, 57.0, 56.8, 66.9]
    targets_scaled = range(int((upper - lower)))
    plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

    # Custom legend
    meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
    baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{baseline}$')
    plt.legend(handles=[meta_patch, baseline_patch])

    # Make sure axis label not cut off
    plt.tight_layout()

    sns_plot.figure.savefig("./meta_boxplot.png")
