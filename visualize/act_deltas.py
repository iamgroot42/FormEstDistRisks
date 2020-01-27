import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm


stats_path = "/p/adversarialml/as9rw/binary_stats/linf/stats"
sense_path = "/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt"
# stats_path = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
# sense_path = "/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt"

(mean, std) = utils.get_stats(stats_path)
x = list(range(mean.shape[0]))
plt.errorbar(x, mean, yerr=std, ecolor='turquoise', color='blue', label='activations')

# Only plot non-inf delta values
senses   = utils.get_sensitivities(sense_path)
x, senses_mean, senses_std = [], [], []
threshold = 1e2
for i in tqdm(range(senses.shape[0])):
	valid_indices = senses[i] != np.inf
	valid_senses = senses[i, valid_indices]
	if np.mean(valid_senses) < threshold:
		senses_mean.append(np.mean(valid_senses))
		senses_std.append(np.std(valid_senses))
		x.append(i)

plt.errorbar(x, senses_mean, yerr=senses_std,  ecolor='lightcoral', color='red', label='deltas', fmt='o')

plt.title('CIFAR-10 (binary) model : Linf')
plt.legend()
plt.savefig("acts_deltas.png")
