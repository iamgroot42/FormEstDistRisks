from numpy.core.getlimits import iinfo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


data = []
columns = [
    "Ratio of females in dataset that model is trained on",
    "Meta-classifier accuracy (%) differentiating between models"
]

# Set dark background style
plt.style.use('dark_background')

data.append([0.2, 99.95, ])
data.append([0.2, 99.8, ])
data.append([0.2, 97.85, ])

data.append([0.3, 96.10, ])
data.append([0.3, 96.7, ])
data.append([0.3, 96.45, ])

data.append([0.4, 61, ])
data.append([0.4, 64.9, ])
data.append([0.4, 69, ])

data.append([0.6, 67, ])
data.append([0.6, 70.95, ])
data.append([0.6, 68.7, ])

data.append([0.7, 93.65, ])
data.append([0.7, 94.9, ])
data.append([0.7, 89.05, ])

data.append([0.8, 98, ])
data.append([0.8, 98.65, ])
data.append([0.8, 99.75, ])

df = pd.DataFrame(data, columns=columns)

wanted = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8]

sns_plot = sns.boxplot(x=columns[0], y=columns[1], data=df)
sns_plot.set(ylim=(50, 100))

# Add vertical dashed line to signal comparison ratio
plt.axvline(x=2.5, color='w', linewidth=1.0, linestyle='--')

# Make sure axis label not cut off
plt.tight_layout()

sns_plot.figure.savefig("./box_plot_meta.png")
