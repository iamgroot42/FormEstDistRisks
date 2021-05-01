from numpy.core.getlimits import iinfo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


data = []
columns = [
    "Mean-degree of dataset that model is trained on",
    "Meta-classifier accuracy (%) differentiating between models"
]

# Set dark background style
plt.style.use('dark_background')

# Degree 9
data.append([9, 99.7, ])
data.append([9, 99.5, ])
data.append([9, 97.7, ])

# Degree 10
data.append([10, 99.05, ])
data.append([10, 94.4, ])
data.append([10, 92.65, ])

# Degree 11
data.append([11, 93, ])
data.append([11, 90.5, ])
data.append([11, 93.75, ])

# Degree 12
data.append([12, 96.1, ])
data.append([12, 84.1, ])
data.append([12, 93.45, ])

# Degree 12.5
data.append([12.5, 76.93, ])
data.append([12.5, 78.30, ])
data.append([12.5, 78.60, ])

# Degree 13.5
data.append([13.5, 72.75, ])
data.append([13.5, 79.65, ])
data.append([13.5, 51.2, ])

# Degree 14
data.append([14, 87.2, ])
data.append([14, 92.9, ])
data.append([14, 90.5, ])

# Degree 15
data.append([15, 99.8, ])
data.append([15, 97.2, ])
data.append([15, 92.6, ])

# Degree 16
data.append([16, 99.9, ])
data.append([16, 99.9, ])
data.append([16, 99.9, ])

# Degree 17
data.append([17, 99.93, ])
data.append([17, 98.57, ])
data.append([17, 100, ])

df = pd.DataFrame(data, columns=columns)

limit_range = np.arange(9, 17.5, 0.5)
wanted = [9, 10, 11, 12, 12.5, 13, 13.5, 14, 15, 16, 17]

sns_plot = sns.boxplot(x=columns[0], y=columns[1], data=df, order=limit_range)
sns_plot.set(ylim=(50, 100))

# Set ticks where needed
sns_plot.set_xticks(wanted)
wanted_x = [0, 2, 4, 6, 7, 8, 9, 10, 12, 14, 16]
plt.xticks(range(0, len(limit_range)), [
           limit_range[i] if i in wanted_x else '' for i in range(len(limit_range))],
           rotation=45)

# Add vertical dashed line to signal comparison ratio
plt.axvline(x=8, color='w', linewidth=1.0, linestyle='--')

# Add labels only for boxes
medians = df.groupby(columns[0])[columns[1]].median().values
ax = plt.gca()
for i, wx in enumerate(wanted_x):
    i_ = i
    if i == 5:
        continue
    if i > 5:
        i_ -= 1
    ax.text(wx,
            medians[i_] - 3.5,
            wanted[i],
            horizontalalignment='center',
            size='x-small',
            color='w',
            weight='semibold')

# Make sure axis label not cut off
plt.tight_layout()

sns_plot.figure.savefig("./meta_boxplot.png")
