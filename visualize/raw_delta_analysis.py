import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gmean
import seaborn as sns
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm
import sys

# stats_path = "/p/adversarialml/as9rw/binary_stats/linf/stats"
# sense_path = "/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt"
# stats_path = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
# sense_path = "/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt"
# stats_path = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"
# sense_path = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1.txt"
# paths = [("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt"),
# 		 ("/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/", "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3.txt"),
# 		 ("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt"),
# 		]
# names = ['standard',
# 	'sensitivity training',
# 	'adversarial training',
# 	]
# colors = ['C0', 'C2', 'C1']

focus_feature = int(sys.argv[1])

# paths = [("/u/as9rw/work/fnb/generic_deltas_nat/%d.txt" % focus_feature, "/u/as9rw/work/fnb/generic_stats/nat/%d/" % focus_feature),
# 		("/u/as9rw/work/fnb/generic_deltas_linf/%d.txt" % focus_feature, "/u/as9rw/work/fnb/generic_stats/linf/%d/" % focus_feature)]

paths = [("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/"),
		("/u/as9rw/work/fnb/linf_2.txt", "/u/as9rw/work/fnb/linf_2/"),
 		 ("/u/as9rw/work/fnb/linf_4.txt", "/u/as9rw/work/fnb/linf_4/"),
 		 ("/u/as9rw/work/fnb/linf_6.txt", "/u/as9rw/work/fnb/linf_6/"),
		 ("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/"),
		 ("/u/as9rw/work/fnb/linf_16.txt", "/u/as9rw/work/fnb/linf_16/"),]

# names = ['Standard', 'L-inf Robust Model']
# colors = ['C0', 'C2']
names = ['Standard', '2/255', '4/255', '6/255', '8/255', '16/255']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

for j, tup in enumerate(paths):
	(mean, std) = utils.get_stats(tup[1])
	# Flatten out mean, std
	mean, std = mean.flatten(), std.flatten()

	# Only plot non-inf delta values
	senses   = utils.get_sensitivities(tup[0])
	print(senses.shape, names[j])
	senese_stats = []
	eps = 0 #1e-15
	for i in tqdm(range(senses.shape[1])):
		valid_senses = np.abs(senses[:, i] - mean) < 10 * std
		# valid_senses = np.abs(senses[:, i]) < 1e0
		
		if len(valid_senses) > 0:
			# senese_stats.append(np.sum(valid_senses))
			# value = np.mean(np.abs(senses[valid_senses, i]))
			value = np.mean(np.abs(senses[valid_senses, i]) / std[valid_senses])
			if value == value:
				senese_stats.append(value)

	plt.plot(list(range(len(senese_stats))), sorted(senese_stats), label=names[j], color=colors[j])


# for j, tup in enumerate(paths):
# 	(mean, std) = utils.get_stats(tup[1])
# 	# Flatten out mean, std
# 	mean, std = mean.flatten(), std.flatten()

# 	# Only plot non-inf delta values
# 	senses   = utils.get_sensitivities(tup[0])
# 	print(senses.shape, names[j])
# 	senese_stats = []
# 	eps = 0 #1e-15
# 	for i in tqdm(range(senses.shape[0])):
# 		# Filter out NANs
# 		# valid_indices = senses[i] != np.inf
# 		# valid_indices = senses[i] < 5e0
# 		valid_indices = np.abs(senses[i] - mean[i]) < 3 * std[i]
# 		# valid_indices = senses[i] != np.inf
# 		valid_senses = senses[i, valid_indices]
		
# 		if std[i] > 0:
# 			# Only consider count of neurons for which delta is within +- 3 delta range
# 			valid_ones = np.sum(np.abs(valid_senses - mean[i]) <= 3 * std[i])
# 			ratio = valid_ones / senses[i].shape
# 			senese_stats.append(100 * ratio)

# 			# if len(valid_senses) > 0:
# 				# senese_stats.append(np.mean(np.abs(valid_senses) / std[i]))
# 				# senese_stats.append(np.log(np.mean(np.abs(valid_senses) / std[i])))

# 	n_bins = 200
# 	plt.plot(list(range(len(senese_stats))), sorted(senese_stats), label=names[j], color=colors[j])

# paths = [("/u/as9rw/work/fnb/deltas_together/latent/nat.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/", "/u/as9rw/work/fnb/CORRS/latent/nat.txt"),
# 		 ("/u/as9rw/work/fnb/deltas_together/latent/linf.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/", "/u/as9rw/work/fnb/CORRS/latent/linf.txt")]


# def get_groupings(path):
# 	clusters = []
# 	with open(path, 'r') as f:
# 		for line in f:
# 			clusters.append([int(x) for x in line.rstrip('\n').split(',')])
# 	return clusters


# for j, tup in enumerate(paths):
# 	(mean, std) = utils.get_stats(tup[1])
# 	# Flatten out mean, std
# 	mean, std = mean.flatten(), std.flatten()

# 	clusters = get_groupings(tup[2])

# 	# Only plot non-inf delta values
# 	senses   = utils.get_sensitivities(tup[0])

# 	print(senses.shape, names[j])
# 	x, senese_stats = [], []
# 	eps = 0 #1e-15
# 	for i in tqdm(range(senses.shape[0])):
# 		valid_indices = senses[i] != np.inf
# 		valid_senses = senses[i, valid_indices]

# 		if std[clusters[i][0]] > 0:
# 			# print(valid_senses)
# 			# senese_stats.append(np.median(valid_senses / (len(clusters[i]) * std[clusters[i][0]])))
# 			# senese_stats.append(np.sum(valid_senses / (len(clusters[i]) * std[clusters[i][0]]) < 1e1))
# 			desired_quantity = valid_senses / len(clusters[i])
# 			ratio = np.sum(np.abs(desired_quantity - mean[clusters[i][0]]) < 3 * std[clusters[i][0]])
# 			senese_stats.append(100 * ratio / len(senses[i]))
		
# 		# if std[i] > 0:
# 		# 	# Only consider count of neurons for which delta is within +- 3 delta range
# 		# 	valid_ones = np.sum(np.abs(valid_senses - mean[i]) <= 3 * std[i])
# 		# 	ratio = valid_ones / senses[i].shape
# 		# 	senese_stats.append(100 * ratio)
# 		# 	x.append(i)

# 	n_bins = 200
# 	plt.plot(list(range(len(senese_stats))), sorted(senese_stats), label=names[j], color=colors[j])


# ax.set(xlim=(0,20))

# ax.set_xlabel("Scaled $\log(\Delta)$ Values", fontsize=15)
# ax.set_ylabel("Fraction of Neurons", fontsize=15)
# plt.xlabel('Scaled $\log(\delta)$ Values')
# plt.ylabel('Fraction of Neurons')
# ax.legend(fontsize=13)
# ax.legend_.remove()
# plt = ax.get_figure()
plt.legend()
# plt.title("Analysis for feature layer %d" % focus_feature)
xvals = [10.0, 9.0, 6.0]

plt.savefig("raw_delta_values.png")
