import numpy as np
import utils
# import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm

from labellines import labelLines

# stats_path = "/p/adversarialml/as9rw/binary_stats/linf/stats"
# sense_path = "/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt"
# stats_path = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
# sense_path = "/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt"
# stats_path = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"
# sense_path = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1.txt"
paths = [("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt"),
		 ("/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/", "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3.txt"),
		 ("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt"),
		]
names = ['standard',
	'sensitivity training',
	'adversarial training',
	]
colors = ['C0', 'C2', 'C1']

for j, tup in enumerate(paths):
	(mean, std) = utils.get_stats(tup[0])
	x = list(range(mean.shape[0]))
	# plt.errorbar(x, mean, yerr=std, ecolor='turquoise', color='blue', label='activations')

	# Only plot non-inf delta values
	senses   = utils.get_sensitivities(tup[1])
	x, senses_mean, senses_std = [], [], []
	eps = 0 #1e-15
	for i in tqdm(range(senses.shape[0])):
		valid_indices = senses[i] != np.inf
		valid_senses = senses[i, valid_indices]
		if std[i] > 0:
			senses_mean.append(np.mean((valid_senses - mean[i]) / (std[i] + eps)))
			x.append(i)
	print("%d has non-zero std out of %d for %s" % (len(x), mean.shape[0], names[j]))

	# plt.errorbar(x, senses_mean, yerr=senses_std,  ecolor='lightcoral', color='red', label='deltas', fmt='o')
	n_bins = 200
	# plt.hist(senses_mean, bins=n_bins, label=names[j], density=True)
	ax = sns.distplot(np.log(senses_mean), norm_hist=True, bins=n_bins, label=names[j], hist=False, color=colors[j])

# plt.xscale('log')
# ax.set(xlabel='', ylabel='')
ax.set(xlim=(0,20))
ax.set_xlabel("Scaled $\log(\Delta)$ Values", fontsize=15)
ax.set_ylabel("Fraction of Neurons", fontsize=15)
# plt.xlabel('Scaled $\log(\delta)$ Values')
# plt.ylabel('Fraction of Neurons')
# ax.legend(fontsize=13)
ax.legend_.remove()
plt = ax.get_figure()
xvals = [10.0, 9.0, 6.0]
# labelLines(plt.gca().get_lines(), align=False, fontsize=13, backgroundcolor=(1.0, 1.0, 1.0, 0.75), xvals=xvals)
plt.text(0.15, .5, r'adversarial training', {'color': 'C1', 'fontsize': 13})
plt.text(0.5, .8, r'sensitivity training', {'color': 'C2', 'fontsize': 13})
ax.annotate('standard', xy=(9, 0.1), xytext=(15, 0.2), color='C0', fontsize=13,
            arrowprops=dict(color='C0', arrowstyle="->"),
            )
# plt.text(-1, .30, r'gamma: $\gamma$', {'color': 'r', 'fontsize': 20})


# plt.legend()
plt.savefig("delta_values.png")
