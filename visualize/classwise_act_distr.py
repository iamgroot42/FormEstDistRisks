import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


stats_path     = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"
stats = np.load(os.path.join(stats_path, "feature_all.npy"))
x = list(range(stats.shape[2]))

for y in range(stats.shape[0]):
	mean = stats[y][0]
	std  = stats[y][1]
	# plt.errorbar(x, mean, yerr=std, label='class_' + str(y+1))
	plt.errorbar(x, mean, label='class_' + str(y+1))
	if y == 1:
		break

plt.title('CIFAR-10 (custom reg)')
plt.legend()
plt.savefig("just_classwise_acts.png")
