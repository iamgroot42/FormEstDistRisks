import numpy as np
import utils
# import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import gmean
import seaborn as sns
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm
import sys

from labellines import labelLines

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
mode          = int(sys.argv[2])

# paths = [("/u/as9rw/work/fnb/generic_deltas_nat/%d.txt" % focus_feature, "/u/as9rw/work/fnb/generic_stats/nat/%d/" % focus_feature),
# 		# ("/u/as9rw/work/fnb/generic_deltas_l2/%d.txt" % focus_feature, "/u/as9rw/work/fnb/generic_stats/l2/%d/" % focus_feature),
# 		("/u/as9rw/work/fnb/generic_deltas_linf/%d.txt" % focus_feature, "/u/as9rw/work/fnb/generic_stats/linf/%d/" % focus_feature)]
# # names = ['Standard', 'L-2 Robust Model', 'L-inf Robust Model']
# # colors = ['C0', 'C1', 'C2']

# names = ['Standard', 'L-inf Robust Model']
# colors = ['C0', 'C2']


paths = [("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/"),
 		 ("/u/as9rw/work/fnb/linf_2.txt", "/u/as9rw/work/fnb/linf_2/"),
 		 ("/u/as9rw/work/fnb/linf_4.txt", "/u/as9rw/work/fnb/linf_4/"),
 		 ("/u/as9rw/work/fnb/linf_6.txt", "/u/as9rw/work/fnb/linf_6/"),
 		 ("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/"),
 		 ("/u/as9rw/work/fnb/linf_16.txt", "/u/as9rw/work/fnb/linf_16/"),]

names = ['Standard', '2/255', '4/255', '6/255', '8/255', '16/255']
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']


# paths = [("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/"),
# 		("/p/adversarialml/as9rw/cifar10_vgg_stats/l2/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/l2/stats/"),
# 		("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/"),
# 		("/u/as9rw/work/fnb/i_tried_delta.txt", "/u/as9rw/work/fnb/i_tried_delta")]
# names = ['Standard', 'L-2 Robust Model', 'L-inf Robust Model', 'Failed Sensitive']
# colors = ['C0', 'C1', 'C2', 'C3']

for j, tup in enumerate(paths):
	(mean, std) = utils.get_stats(tup[1])
	# Flatten out mean, std
	mean, std = mean.flatten(), std.flatten()
	x = list(range(mean.shape[0]))
	# plt.errorbar(x, mean, yerr=std, ecolor='turquoise', color='blue', label='activations')

	# Only plot non-inf delta values
	senses   = utils.get_sensitivities(tup[0])
	print(senses.shape, names[j])
	x, senese_stats = [], []
	eps = 0 #1e-15
	for i in tqdm(range(senses.shape[0])):
		valid_indices = senses[i] != np.inf
		valid_senses = senses[i, valid_indices]
		if std[i] > 0:
			if mode == 1:
				# Use Mean
				senese_stats.append(np.mean(valid_senses / (std[i] + eps)))
			elif mode == 2:
				# Use Median
				senese_stats.append(np.median(valid_senses / (std[i] + eps)))
			else:
				# Geometric mean
				senese_stats.append(np.mean(np.log(np.abs(valid_senses / (std[i] + eps)))))
			x.append(i)
	print("%d has non-zero std out of %d for %s" % (len(x), mean.shape[0], names[j]))

	n_bins = 200
	if mode > 2:
		ax = sns.distplot(senese_stats, norm_hist=True, bins=n_bins, label=names[j], hist=False, color=colors[j])
	else:
		ax = sns.distplot(np.log(np.abs(senese_stats)), norm_hist=True, bins=n_bins, label=names[j], hist=False, color=colors[j])

# plt.xscale('log')
# ax.set(xlabel='', ylabel='')
# AM, Median
ax.set(xlim=(0,20))
# GM
# ax.set(xlim=(0,30))

ax.set_xlabel("Scaled $\log(\Delta)$ Values", fontsize=15)
ax.set_ylabel("Fraction of Neurons", fontsize=15)
# plt.xlabel('Scaled $\log(\delta)$ Values')
# plt.ylabel('Fraction of Neurons')
# ax.legend(fontsize=13)
ax.legend_.remove()
plt = ax.get_figure()
plt.legend()
# plt.title("Analysis for feature layer %d" % focus_feature)
xvals = [10.0, 9.0, 6.0]
# labelLines(plt.gca().get_lines(), align=False, fontsize=13, backgroundcolor=(1.0, 1.0, 1.0, 0.75), xvals=xvals)
# Mean mode
# plt.text(0.15, .5, r'adversarial training', {'color': 'C1', 'fontsize': 13}))
# plt.text(0.5, .8, r'sensitivity training', {'color': 'C2', 'fontsize': 13}))
# Median mode
# plt.text(0.15, .2, r'adversarial training', {'color': 'C1', 'fontsize': 13})
# plt.text(0.35, .8, r'sensitivity training', {'color': 'C2', 'fontsize': 13})
# Geometric Mean mode
# plt.text(0.5, .2, r'adversarial training', {'color': 'C1', 'fontsize': 13})
# plt.text(0.34, .8, r'sensitivity training', {'color': 'C2', 'fontsize': 13})
# ax.annotate('standard', xy=(9, 0.1), xytext=(15, 0.2), color='C0', fontsize=13,
#             arrowprops=dict(color='C0', arrowstyle="->"),
#             )
# plt.text(-1, .30, r'gamma: $\gamma$', {'color': 'r', 'fontsize': 20})

plt.savefig("delta_values.png")
