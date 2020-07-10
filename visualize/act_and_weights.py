# import torch as ch
import numpy as np
# from robustness.datasets import CIFAR
# from robustness.model_utils import make_and_restore_model

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import utils


from matplotlib import rc
rc('text', usetex=True)

if __name__ == "__main__":
	scale_custom = "1e1_1e2_1e-2_16_3/"
	scale_nat = "nat_stats/"
	scale_linf = "linf_stats"

	from matplotlib import rc
	rc('text', usetex=True)

	# Load model
	def populate(scale_path, colors, prefix):

		# Get scaled activations values
		(mean, std) = utils.get_stats(scale_path)
		indices = np.argsort(mean)
		mean = mean[indices]
		std  = std[indices]
		x = list(range(mean.shape[0]))

		# Normalize means to be in [0,1] range
		mean -= mean.min()
		mean /= mean.max()

		plt.errorbar(x, mean, label=prefix)

	populate(scale_nat, (('turquoise', 'blue'), ('lightcoral', 'red')), 'standard')
	populate(scale_linf, (('thistle', 'violet'), ('lightsteelblue', 'cornflowerblue')), 'adversarial training')
	populate(scale_custom, (('lightyellow', 'yellow'), ('lightgreen', 'green')), 'sensitivity training')
	
	plt.text(50, 0.75, r'sensitivity training', {'color': 'C2', 'fontsize': 13})
	plt.text(300, 0.2, r'adversarial training', {'color': 'C1', 'fontsize': 13})
	plt.text(200, 0.3, r'standard', {'color': 'C0', 'fontsize': 13})
	# plt.text(0.15, .5, r'adversarial training', {'color': 'C1', 'fontsize': 13})

	plt.ylim(0, None)
	plt.xlabel('Neuron Index', fontsize=15)
	plt.ylabel('Scaled Feature Activation Value', fontsize=15)
	# plt.legend(fontsize=13)
	plt.grid(True)
	plt.savefig("all_acts.png")
