import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


data = []
columns = [
    r"Mean-degree of training data ($\alpha$)",
    "Accuracy (%)"
]

darkmode = False
add_legend = True
novtitle = False

if darkmode:
    # Set dark background style
    plt.style.use('dark_background')

# Set font size
# plt.rcParams.update({'font.size': 13})
plt.rcParams.update({'font.size': 18})

# Degree 9
data.append([9, 99.7])
data.append([9, 99.5])
data.append([9, 97.7])

# Degree 10
data.append([10, 99.05])
data.append([10, 94.4])
data.append([10, 92.65])

# Degree 11
data.append([11, 93])
data.append([11, 90.5])
data.append([11, 93.75])

# Degree 12
data.append([12, 96.1])
data.append([12, 84.1])
data.append([12, 93.45])

# Degree 14
data.append([14, 87.2])
data.append([14, 92.9])
data.append([14, 90.5])

# Degree 15
values = [98.34, 89.45, 99.95, 99.95, 96.93, 100, 92.3, 100]
for v in values:
    data.append([15, v])

# Degree 16
data.append([16, 99.9, ])
data.append([16, 99.9, ])
data.append([16, 95.4, ])

# Degree 17
values = [100, 100, 99.9, 99.3, 99.5, 98.3, 100, 98.5, 100]
for v in values:
    data.append([17, v])


df = pd.DataFrame(data, columns=columns)
wanted = [9, 10, 11, 12, 13, 14, 15, 16, 17]

sns_plot = sns.boxplot(
    x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)
sns_plot.set(ylim=(45, 101))

if novtitle:
    plt.ylabel("", labelpad=0)

# Add dividing line in centre
lower, upper = plt.gca().get_xlim()
midpoint = (lower + upper) / 2
plt.axvline(x=midpoint, color='w' if darkmode else 'black',
            linewidth=1.0, linestyle='--')

# Map range to numbers to be plotted
baselines = [50, 64.1, 56.63, 53.9, 55.8, 50, 50, 50]
targets_scaled = range(int((upper - lower)))
plt.plot(targets_scaled, baselines, color='C1', marker='x', linestyle='--')

# Plot numbers for threshold-based accuracy
thresholds = [
    [51.1, 51.95, 50.2],
    [50.4, 50.05, 52.95],
    [50.45, 50.5, 50.6],
    [50.4, 50.1, 50.65],
    [50.7, 50.95, 50.15],
    [50.8, 50.15, 50.3],
    [51.1, 51.95, 50.2],
    [53.79, 53.94, 51.43],
]
means, errors = np.mean(thresholds, 1), np.std(thresholds, 1)
plt.errorbar(targets_scaled, means, yerr=errors, color='C2', linestyle='--')

if add_legend:
    # Custom legend
    meta_patch = mpatches.Patch(color='C0', label=r'$Acc_{meta-classifier}$')
    baseline_patch = mpatches.Patch(color='C1', label=r'$Acc_{loss test}$')
    threshold_patch = mpatches.Patch(color='C2', label=r'$Acc_{threshold test}$')
    plt.legend(handles=[meta_patch, baseline_patch,
                        threshold_patch], prop={'size': 13})

# Make sure axis label not cut off
plt.tight_layout()

sns_plot.figure.savefig("./meta_boxplot.png")
