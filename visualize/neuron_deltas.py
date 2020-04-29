import numpy as np
import utils
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm


# path = ("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt")
# path = ("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt")
# path = ("/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/", "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3.txt")
path = ("/p/adversarialml/as9rw/cifar10_vgg_stats/l2/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/l2/deltas.txt")

(mean, std) = utils.get_stats(path[0])

# Only plot non-inf delta values
senses = utils.get_sensitivities(path[1])

# Get ordering of neurons (seemingly easiest to attack -> hardest to attack)
scaled_senses = (senses - np.expand_dims(mean, 1)) / (np.expand_dims(std, 1))
# Replace INF with largest non-INF values
scaled_senses[scaled_senses == np.inf] = np.max(scaled_senses[scaled_senses != np.inf])
sorting = np.argsort(np.mean(scaled_senses, 1))
# For NAT:
# neurons_to_try = [sorting[10], sorting[20], sorting[30], sorting[40], sorting[50], sorting[60], 
# 				sorting[-90], sorting[-80], sorting[-70], sorting[-60], sorting[-50], sorting[-40]]
# For LINF:
# neurons_to_try = [sorting[10], sorting[15], sorting[20], sorting[25], sorting[30], sorting[35], 
# 				sorting[-175], sorting[-170], sorting[-165], sorting[-160], sorting[-155], sorting[-150]]
# For Sensitive:
neurons_to_try = [sorting[10], sorting[15], sorting[20], sorting[25], sorting[30], sorting[35], 
				sorting[-105], sorting[-100], sorting[-95], sorting[-90], sorting[-85], sorting[-80]]
colors = [(0, 0, 1), (1, 0.5, 0)]

for i, neuron in enumerate(neurons_to_try):
	valid_indices = senses[neuron] != np.inf
	valid_senses = senses[neuron, valid_indices]
	if std[neuron] > 0:
		senses_mean = (valid_senses - mean[neuron]) / (std[neuron])
		n_bins = 200
		if i < len(neurons_to_try) // 2:
			index = i + 1
			label = "easy %d" % (index)
			color = colors[0]
		else:
			color = colors[1]
			index = i + 1 - len(neurons_to_try) // 2
			label = "hard %d" % (index)
		ax = sns.distplot(np.log(senses_mean), norm_hist=True, bins=n_bins, hist=False, label=label, color=color + ((index / (len(neurons_to_try) // 2),)))

# FOR NAT:
# ax.set(xlim=(0,20))
# FOR LINF, L2, SENSE:
ax.set(xlim=(0,15))
ax.set_xlabel("Scaled $\log(\Delta)$ Values", fontsize=15)
ax.set_ylabel("Fraction of Examples", fontsize=15)
ax.set_title("$L_2$ Robust Model")

plt = ax.get_figure()
plt.savefig("delta_values_specific_neurons.png")
