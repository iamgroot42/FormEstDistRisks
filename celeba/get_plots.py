import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


x = np.load("laterz/all_preds.npy")

sorting = np.argsort(x[0][0])
specific_ones = np.mean(np.mean(x, 0), 0)
positive = np.nonzero(specific_ones >= 0)[0]

for i in range(10):
    plt.hist(x[0][i][positive], color='red', bins=100)

for i in range(10):
    plt.hist(x[1][i][positive], color='blue', bins=100)

for i in range(10):
    plt.hist(x[2][i][positive], color='green', bins=100)


for i in range(10):
    plt.hist(x[3][i][positive], color='orange', bins=100)

plt.savefig('../visualize/celeb0scores.png')