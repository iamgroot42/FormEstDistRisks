import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm


# stats_path = "/p/adversarialml/as9rw/binary_stats/linf/stats"
# sense_path = "/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt"
# stats_path = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
# sense_path = "/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt"
# stats_path = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"
# sense_path = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1.txt"
paths = [("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt"),
		 ("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt"),
		 ("/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/", "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3.txt")
		]
names = ['standard', 'adversarial training', 'robust']

for j, tup in enumerate(paths):
	(mean, std) = utils.get_stats(tup[0])
	x = list(range(mean.shape[0]))
	# plt.errorbar(x, mean, yerr=std, ecolor='turquoise', color='blue', label='activations')

	# Only plot non-inf delta values
	senses   = utils.get_sensitivities(tup[1])
	x, senses_mean, senses_std = [], [], []
	threshold = np.inf #1e3
	eps = 1e-10
	for i in tqdm(range(senses.shape[0])):
		valid_indices = senses[i] != np.inf
		valid_senses = senses[i, valid_indices]
		if np.mean((valid_senses - mean[i]) / (std[i] + eps)) < threshold:
			# senses_mean.append(np.mean((valid_senses - mean[i]) / std[i]))
			# senses_std.append(np.std((valid_senses - mean[i]) / std[i]))
			senses_mean.append(np.mean((valid_senses - mean[i]) / (std[i] + eps)))
			x.append(i)

	# plt.errorbar(x, senses_mean, yerr=senses_std,  ecolor='lightcoral', color='red', label='deltas', fmt='o')
	plt.errorbar(x, senses_mean, fmt='.', label=names[j])

plt.yscale('log')
plt.legend()
plt.xlabel('Neuron Index')
plt.ylabel('Scaled $\delta$ value')
plt.title('CIFAR-10 : scaled sensitivity values')
plt.savefig("acts_deltas.png")
