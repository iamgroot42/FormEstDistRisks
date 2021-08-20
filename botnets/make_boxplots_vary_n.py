import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
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

    if args.darkplot:
        # Set dark background style
        plt.style.use('dark_background')

    # Set font size
    plt.rcParams.update({'font.size': 18})
    # plt.rcParams.update({'font.size': 14})

    data = []
    columns = [
        "Accuracy (%)",
        r'Layer $n$'
    ]

    raw_data = {
        "1": {
            800: [
                [50, 50, 49.25, 51.62, 50]
            ]
        },
        "2": {
            800: [
                [98.75, 99.5, 99.5, 93.38, 93.5]
            ]
        },
        "3": {
            800: [
                [99.62, 92.88, 90.25, 98, 99.12]
            ]
        },
        "4": {
            800: [
                [99.25, 99.88, 99.12, 97.12, 92.25]
            ]
        },
        "5": {
            800: [
                [95.88, 99.38, 99.75, 98.38, 100]
            ]
        },
        "6": {
            800: [
                [100, 100, 99.88, 100, 100]
            ]
        },
        "All": {
            800: [
                [100, 99.62, 99.75, 100, 99.75]
            ]
        },
    }

    focus_n = 800
    for n, v1 in raw_data.items():
        v2 = v1[focus_n]
        for i in range(len(v2)):
            for j in range(len(v2[i])):
                data.append([v2[i][j], n])

    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(x=columns[1], y=columns[0],
                           data=df, showfliers=False,
                           color="C0")

    if args.novtitle:
        plt.ylabel("", labelpad=0)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(45, 101))

    if not args.legend:
        plt.legend([], [], frameon=False)

    # Make sure axis label not cut off
    plt.tight_layout()

    sns_plot.figure.savefig("./boxplot_varying_n.png")
