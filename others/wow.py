import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

targets = ["0.0", "0.1", "0.2", "0.3", "0.4",
           "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"]


fill_data = np.zeros((len(targets), len(targets)))
mask = np.zeros((len(targets), len(targets)), dtype=bool)
annot_data = [[""] * len(targets) for _ in range(len(targets))]

for i in range(len(targets)):
    for j in range(len(targets)-(i+1)):
        mask[j+i+1][i] = True

plt.rcParams.update({'font.size': 5})
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('axes', labelsize=10)

fill_data_other = np.zeros((len(targets), len(targets)))

# Census Meta
# fill_data[0][1:] = [100, 100, 99, 100, 100, 99, 100, 100, 100, 99]
# fill_data[1][2:] = [52, 59, 66, 73, 76, 78, 79, 83, 74]
# fill_data[2][3:] = [50, 56, 57, 64, 66, 65, 70, 83]
# fill_data[3][4:] = [50, 52, 59, 60, 59, 64, 79]
# fill_data[4][5:] = [49, 53, 55, 55, 62, 79]
# fill_data[5][6:] = [55, 62, 67, 70, 83]
# fill_data[6][7:] = [50, 52, 56, 77]
# fill_data[7][8:] = [51, 59, 76]
# fill_data[8][9:] = [55, 76]
# fill_data[9][10:] = [77]
# annot_data[0][1:] = [r'100 $\pm$ 0', r'100 $\pm$ 0', r'99 $\pm$ 0', r'100 $\pm$ 0', r'100 $\pm$ 0', r'99 $\pm$ 0', r'100 $\pm$ 0', r'100 $\pm$ 0', r'100 $\pm$ 0', r'99 $\pm$ 0']
# annot_data[1][2:] = [r'52 $\pm$ 0', r'59 $\pm$ 1', r'66 $\pm$ 1', r'73 $\pm$ 3', r'76 $\pm$ 1', r'78 $\pm$ 3', r'79 $\pm$ 3', r'83 $\pm$ 2', r'74 $\pm$ 5']
# annot_data[2][3:] = [r'50 $\pm$ 1', r'56 $\pm$ 1', r'57 $\pm$ 3', r'64 $\pm$ 1', r'66 $\pm$ 3', r'65 $\pm$ 3', r'70 $\pm$ 3', r'83 $\pm$ 13']
# annot_data[3][4:] = [r'50 $\pm$ 0', r'52 $\pm$ 2', r'59 $\pm$ 0', r'60 $\pm$ 3', r'59 $\pm$ 3', r'64 $\pm$ 3', r'79 $\pm$ 11']
# annot_data[4][5:] = [r'49 $\pm$ 2', r'53 $\pm$ 1', r'55 $\pm$ 1', r'55 $\pm$ 1', r'62 $\pm$ 2', r'79 $\pm$ 13']
# annot_data[5][6:] = [r'55 $\pm$ 3', r'62 $\pm$ 1', r'67 $\pm$ 0', r'70 $\pm$ 0', r'83 $\pm$ 7']
# annot_data[6][7:] = [r'50 $\pm$ 0', r'52 $\pm$ 1', r'56 $\pm$ 1', r'77 $\pm$ 13']
# annot_data[7][8:] = [r'51 $\pm$ 0', r'59 $\pm$ 1', r'76 $\pm$ 9']
# annot_data[8][9:] = [r'55 $\pm$ 0', r'76 $\pm$ 9']
# annot_data[9][10:] = [r'77 $\pm$ 11']
# fill_data_other[1][:1] = [800]
# fill_data_other[2][:2] = [800, 0.1]
# fill_data_other[3][:3] = [9.05, 0.4, 0.1]
# fill_data_other[4][:4] = [800, 0.6, 0.07, 0.01]
# fill_data_other[5][:5] = [800, 0.4, 0.04, 0.004, 0]
# fill_data_other[6][:6] = [3.5, 0.38, 0.12, 0.06, 0.009, 0.06]
# fill_data_other[7][:7] = [800, 0.35, 0.11, 0.05, 0.02, 0.18, 0]
# fill_data_other[8][:8] = [800, 0.27, 0.07, 0.04, 0.01, 0.26, 0.01, 0]
# fill_data_other[9][:9] = [800, 0.26, 0.12, 0.07, 0.07, 0.3, 0.04, 0.13, 0.08]
# fill_data_other[10][:10] = [800, 0.13, 0.36, 0.34, 0.45, 0.83, 0.68, 0.88, 1.41, 3.27]

# CelebA Meta
# fill_data[0][1:] = [53, 61, 68, 81, 87, 91, 91, 92, 96, 92]
# fill_data[1][2:] = [50, 57, 67, 77, 87, 91, 95, 95, 94]
# fill_data[2][3:] = [53, 58, 63, 75, 81, 91, 90, 92]
# fill_data[3][4:] = [52, 56, 68, 76, 86, 92, 94]
# fill_data[4][5:] = [50, 58, 6, 81, 83, 92]
# fill_data[5][6:] = [51, 58, 77, 81, 88]
# fill_data[6][7:] = [49, 60, 72, 67]
# fill_data[7][8:] = [53, 58, 71]
# fill_data[8][9:] = [54, 57]
# fill_data[9][10:] = [52]
# annot_data[0][1:] = [r'53 $\pm$ 2', r'61 $\pm$ 4', r'68 $\pm$ 8', r'81 $\pm$ 6', r'87 $\pm$ 4', r'91 $\pm$ 2', r'91 $\pm$ 7', r'92 $\pm$ 2', r'96 $\pm$ 1', r'92 $\pm$ 4']
# annot_data[1][2:] = [r'50 $\pm$ 2', r'57 $\pm$ 2', r'67 $\pm$ 6', r'77 $\pm$ 4', r'87 $\pm$ 1', r'91 $\pm$ 2', r'95 $\pm$ 3', r'95 $\pm$ 2', r'94 $\pm$ 2']
# annot_data[2][3:] = [r'53 $\pm$ 2', r'58 $\pm$ 1', r'63 $\pm$ 6', r'75 $\pm$ 5', r'81 $\pm$ 8', r'91 $\pm$ 3', r'90 $\pm$ 2', r'92 $\pm$ 3']
# annot_data[3][4:] = [r'52 $\pm$ 1',  r'56 $\pm$ 1', r'68 $\pm$ 3', r'76 $\pm$ 2', r'86 $\pm$ 5', r'92 $\pm$ 2', r'94 $\pm$ 1']
# annot_data[4][5:] = [r'50 $\pm$ 1', r'58 $\pm$ 2', r'63 $\pm$ 3', r'81 $\pm$ 6', r'83 $\pm$ 4', r'92 $\pm$ 2']
# annot_data[5][6:] = [r'51 $\pm$ 2', r'58 $\pm$ 4', r'77 $\pm$ 3', r'81 $\pm$ 1', r'88 $\pm$ 4']
# annot_data[6][7:] = [r'49 $\pm$ 1', r'60 $\pm$ 16', r'72 $\pm$ 3', r'67 $\pm$ 7']
# annot_data[7][8:] = [r'53 $\pm$ 10', r'58 $\pm$ 8', r'71 $\pm$ 8']
# annot_data[8][9:] = [r'54 $\pm$ 10', r'57 $\pm$ 11']
# annot_data[9][10:] = [r'52 $\pm$ 4']
# fill_data_other[1][:1] = [0.03]
# fill_data_other[2][:2] = [0.22, 0]
# fill_data_other[3][:3] = [0.39, 0.08, 0.03]
# fill_data_other[4][:4] = [0.95, 0.30, 0.09, 0.01]
# fill_data_other[5][:5] = [1.14, 0.59, 0.15, 0.04, 0]
# fill_data_other[6][:6] = [1.22, 0.98, 0.42, 0.25, 0.06, 0.02]
# fill_data_other[7][:7] = [0.93, 1.02, 0.49, 0.37, 0.15, 0.08, 0]
# fill_data_other[8][:8] = [0.76, 1.10, 0.80, 0.75, 0.70, 0.73, 0.14, 0.03]
# fill_data_other[9][:9] = [0.81, 0.76, 0.68, 1.11, 0.71, 0.83, 0.53, 0.10, 0.05]
# fill_data_other[10][:10] = [800, 0.65, 0.76, 1.24, 1.33, 1.24, 0.24, 0.54, 0.09, 0.02]

# CelebA Threshold
fill_data[0][1:] = [50, 53, 54, 58, 56, 60, 64, 71, 76, 75]
fill_data[1][2:] = [51, 51, 52, 53, 58, 63, 68, 73, 74]
fill_data[2][3:] = [51, 51, 51, 57, 62, 70, 72, 75]
fill_data[3][4:] = [50, 52, 55, 61, 65, 68, 73]
fill_data[4][5:] = [51, 55, 60, 65, 68, 71]
fill_data[5][6:] = [54, 60, 62, 67, 71]
fill_data[6][7:] = [55, 58, 62, 67]
fill_data[7][8:] = [53, 57, 59]
fill_data[8][9:] = [52, 56]
fill_data[9][10:] = [54]
annot_data[0][1:] = [r'50 $\pm$ 0', r'53 $\pm$ 0', r'54 $\pm$ 1', r'58 $\pm$ 2', r'56 $\pm$ 1', r'60 $\pm$ 0', r'64 $\pm$ 2', r'71 $\pm$ 0', r'76 $\pm$ 0', r'75 $\pm$ 0']
annot_data[1][2:] = [r'51 $\pm$ 0', r'51 $\pm$ 0', r'52 $\pm$ 1', r'53 $\pm$ 2', r'58 $\pm$ 0', r'63 $\pm$ 2', r'68 $\pm$ 1', r'73 $\pm$ 1', r'74 $\pm$ 0']
annot_data[2][3:] = [r'51 $\pm$ 0', r'51 $\pm$ 0', r'51 $\pm$ 0', r'57 $\pm$ 1', r'62 $\pm$ 2', r'70 $\pm$ 1', r'72 $\pm$ 0', r'75 $\pm$ 0']
annot_data[3][4:] = [r'50 $\pm$ 0', r'52 $\pm$ 0', r'55 $\pm$ 1', r'61 $\pm$ 1', r'65 $\pm$ 1', r'68 $\pm$ 2', r'73 $\pm$ 1']
annot_data[4][5:] = [r'51 $\pm$ 0', r'55 $\pm$ 1', r'60 $\pm$ 1', r'65 $\pm$ 2', r'68 $\pm$ 2', r'71 $\pm$ 2']
annot_data[5][6:] = [r'54 $\pm$ 1', r'60 $\pm$ 0', r'62 $\pm$ 0', r'67 $\pm$ 0', r'71 $\pm$ 1']
annot_data[6][7:] = [r'55 $\pm$ 1', r'58 $\pm$ 2', r'62 $\pm$ 2', r'67 $\pm$ 1']
annot_data[7][8:] = [r'53 $\pm$ 1', r'57 $\pm$ 1', r'59 $\pm$ 2']
annot_data[8][9:] = [r'52 $\pm$ 0', r'56 $\pm$ 1']
annot_data[9][10:] = [r'54 $\pm$ 0']
fill_data_other[1][:1] = [0]
fill_data_other[2][:2] = [0.02, 0, ]
fill_data_other[3][:3] = [0.02, 0, 0]
fill_data_other[4][:4] = [0.05, 0, 0, 0]
fill_data_other[5][:5] = [0.02, 0, 0, 0, 0]
fill_data_other[6][:6] = [0.04, 0.03, 0.03, 0.02, 0.02, 0.04]
fill_data_other[7][:7] = [0.07, 0.06, 0.06, 0.06, 0.09, 0.12, 0.07]
fill_data_other[8][:8] = [0.12, 0.09, 0.13, 0.1, 0.14, 0.13, 0.09, 0.03]
fill_data_other[9][:9] = [0.14, 0.11, 0.14, 0.13, 0.17, 0.21, 0.15, 0.08, 0.01]
fill_data_other[10][:10] = [800, 0.11, 0.18, 0.20, 0.21, 0.28, 0.24, 0.09, 0.07, 0.06]

annot_data_other = [[""] * len(targets) for _ in range(len(targets))]
mask_other = np.logical_not(mask)
for i in range(mask_other.shape[0]):
    mask_other[i][i] = False

temp = fill_data_other.copy()
temp[temp > 100] = 0
maz = np.max(temp)

for i in range(1, len(targets)):
    for j in range(i):
        if fill_data_other[i][j] > 100:
            annot_data_other[i][j] = r"$>100^\dagger$"
            fill_data_other[i][j] = maz
        else:
            annot_data_other[i][j] = "%.2f" % fill_data_other[i][j]

# # First heatmap (top-right)
# for i in range(len(targets)):
#     for j in range(len(targets)-(i+1)):
#         m = raw_data_loss[i][j]
#         fill_data[i][j+i+1] = m
#         mask[i][j+i+1] = False
#         annot_data[i][j+i+1] = r'%d' % m


# Second heatmap (bottom-left)
# mask = np.zeros_like(mask)
# for i in range(len(targets)):
#     for j in range(len(targets)-(i+1)):
#         m = eff_vals[i][j]
#         if m > 100:
#             fill_data[i][j+i+1] = 100
#             annot_data[i][i] = r'$\dagger$'
#         else:
#             fill_data[i][j+i+1] = m
#             annot_data[i][i] = "%.1f" % m
#         mask[i][j+i+1] = False

for i in range(mask.shape[0]):
    mask[i][i] = True
    mask_other[i][i] = True

sns_plot = sns.heatmap(fill_data, xticklabels=targets,
                       yticklabels=targets,
                       annot=annot_data,
                       cbar=False,
                       mask=mask, fmt="^",
                       vmin=50, vmax=100,)

sns_plot = sns.heatmap(fill_data_other, xticklabels=targets,
                       yticklabels=targets,
                       annot=annot_data_other,
                       mask=mask_other, fmt="^",
                       cmap="YlGnBu",)

sns_plot.set(xlabel=r'$\alpha_0$', ylabel=r'$\alpha_1$')
sns_plot.figure.savefig("./yeah.png")
