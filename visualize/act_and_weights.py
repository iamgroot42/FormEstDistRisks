import torch as ch
import numpy as np
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import utils

from labellines import labelLines


if __name__ == "__main__":
	# model_nat    = "/p/adversarialml/as9rw/models_cifar10/cifar_nat.pt"
	# model_l2     = "/p/adversarialml/as9rw/models_cifar10/cifar_l2_0_5.pt"
	# model_linf   = "/p/adversarialml/as9rw/models_cifar10/cifar_linf_8.pt"
	# scale_nat    = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
	# scale_l2     = "/p/adversarialml/as9rw/cifar10_stats/l2/stats"
	# scale_linf   = "/p/adversarialml/as9rw/cifar10_stats/linf/stats"

	# model_nat    = "/p/adversarialml/as9rw/models_correct/normal/checkpoint.pt.latest"
	# model_l2     = "/p/adversarialml/as9rw/models_correct/l2/checkpoint.pt.latest"
	# model_linf   = "/p/adversarialml/as9rw/models_correct/linf/checkpoint.pt.latest"
	# scale_nat    = "/p/adversarialml/as9rw/binary_stats/nat/stats"
	# scale_l2     = "/p/adversarialml/as9rw/binary_stats/l2/stats"
	# scale_linf   = "/p/adversarialml/as9rw/binary_stats/linf/stats"

	# model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_10000.000000_16_0.010000_1/checkpoint.pt.best"
	# scale_nat = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"
	model_custom = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"
	scale_custom = "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/"
	model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt"
	scale_nat = "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/"
	# model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt"
	# scale_nat = "/p/adversarialml/as9rw/cifar10_vgg_stats/l2/stats/"
	model_linf = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt"
	scale_linf = "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/"

	ds = CIFAR()

	from matplotlib import rc
	rc('text', usetex=True)

	# Load model
	def populate(model_path, scale_path, colors, prefix):
		model_kwargs = {
			# 'arch': 'resnet50',
			'arch': 'vgg19',
			'dataset': ds,
			'resume_path': model_path
		}
		model, _ = make_and_restore_model(**model_kwargs)
		model.eval()

		# Get scaled activations values
		(mean, std) = utils.get_stats(scale_path)
		indices = np.argsort(mean)
		mean = mean[indices]
		std  = std[indices]
		x = list(range(mean.shape[0]))

		# Normalize means to be in [0,1] range
		mean -= mean.min()
		mean /= mean.max()

		# plt.errorbar(x, mean, yerr=std, ecolor=colors[0][0], color=colors[0][1], label=prefix)
		# plt.errorbar(x, mean, color=colors[0][1], label=prefix)
		plt.errorbar(x, mean, label=prefix)

	
		# Extract final weights matrix from model
		with ch.no_grad():
			weights = model.state_dict().get("module.model.classifier.weight") # n_classes * n_features vector

		weights_mean, weights_std = ch.mean(weights, dim=0), ch.std(weights, dim=0)
		weights_mean = weights_mean.cpu().numpy()
		weights_std  = weights_std.cpu().numpy()
		weights_mean, weights_std = weights_mean[indices], weights_std[indices]

		# plt.errorbar(x, weights_std, yerr=weights_std, ecolor=colors[1][0], color=colors[1][1], label=prefix + ' weights')


	populate(model_nat, scale_nat, (('turquoise', 'blue'), ('lightcoral', 'red')), 'standard')
	populate(model_linf, scale_linf, (('thistle', 'violet'), ('lightsteelblue', 'cornflowerblue')), 'adversarial training')
	populate(model_custom, scale_custom, (('lightyellow', 'yellow'), ('lightgreen', 'green')), 'sensitivity training')
	# plt.title('CIFAR-10 (binary)')
	# plt.title('CIFAR-10 : Feature Activation Values')

	# labelLines(plt.gca().get_lines(), align=False, fontsize=13, backgroundcolor=(1.0, 1.0, 1.0, 0.75))

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
