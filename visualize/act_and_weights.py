import torch as ch
import numpy as np
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import dill

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import utils


if __name__ == "__main__":

	dataset_path = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
	# model_nat    = "/p/adversarialml/as9rw/models_cifar10/cifar_nat.pt"
	# model_l2     = "/p/adversarialml/as9rw/models_cifar10/cifar_l2_0_5.pt"
	# model_linf   = "/p/adversarialml/as9rw/models_cifar10/cifar_linf_8.pt"
	# scale_nat    = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
	# scale_l2     = "/p/adversarialml/as9rw/cifar10_stats/l2/stats"
	# scale_linf   = "/p/adversarialml/as9rw/cifar10_stats/linf/stats"

	model_nat    = "/p/adversarialml/as9rw/models_correct/normal/checkpoint.pt.latest"
	model_l2     = "/p/adversarialml/as9rw/models_correct/l2/checkpoint.pt.latest"
	model_linf   = "/p/adversarialml/as9rw/models_correct/linf/checkpoint.pt.latest"
	scale_nat    = "/p/adversarialml/as9rw/binary_stats/nat/stats"
	scale_l2     = "/p/adversarialml/as9rw/binary_stats/l2/stats"
	scale_linf   = "/p/adversarialml/as9rw/binary_stats/linf/stats"

	ds = GenericBinary(dataset_path)
	# ds = CIFAR()

	# Load model
	def populate(model_path, scale_path, colors, prefix):
		model_kwargs = {
			'arch': 'resnet50',
			'dataset': ds,
			'resume_path': model_path
		}
		model, _ = make_and_restore_model(**model_kwargs)
		model.eval()

		# Get scaled delta values
		(mean, std) = utils.get_stats(scale_path)
		indices = np.argsort(mean)
		mean = mean[indices]
		std  = std[indices]
		x = list(range(mean.shape[0]))

		plt.errorbar(x, mean, yerr=std, ecolor=colors[0][0], color=colors[0][1], label=prefix + ' activations')
	
		# Extract final weights matrix from model
		with ch.no_grad():
			weights = model.state_dict().get("module.model.linear.weight") # n_classes * n_features vector

		weights_mean, weights_std = ch.mean(weights, dim=0), ch.std(weights, dim=0)
		weights_mean = weights_mean.cpu().numpy()
		weights_std  = weights_std.cpu().numpy()
		weights_mean, weights_std = weights_mean[indices], weights_std[indices]

		plt.errorbar(x, weights_std, yerr=weights_std, ecolor=colors[1][0], color=colors[1][1], label=prefix + ' weights')


	populate(model_nat, scale_nat, (('turquoise', 'blue'), ('lightcoral', 'red')), 'nat')
	populate(model_l2, scale_l2, (('lightyellow', 'yellow'), ('lightgreen', 'green')), 'l2')
	populate(model_linf, scale_linf, (('thistle', 'violet'), ('lightsteelblue', 'cornflowerblue')), 'linf')
	plt.title('CIFAR-10 (binary)')
	plt.legend()
	plt.savefig("acts_with_weights.png")
