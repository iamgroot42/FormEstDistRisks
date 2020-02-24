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

	# model_nat    = "/p/adversarialml/as9rw/models_cifar10/cifar_nat.pt"
	# scale_nat    = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
	# model_nat    = "/p/adversarialml/as9rw/models_cifar10/cifar_l2_0_5.pt"
	# scale_nat    = "/p/adversarialml/as9rw/cifar10_stats/l2/stats"
	# model_nat    = "/p/adversarialml/as9rw/models_cifar10/cifar_linf_8.pt"
	# scale_nat    = "/p/adversarialml/as9rw/cifar10_stats/linf/stats"
	# model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt"
	# scale_nat = "/p/adversarialml/as9rw/cifar10_vgg_stats/l2/stats/"
	# model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt"
	# scale_nat = "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/"
	model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt"
	scale_nat = "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/"
	# model_nat = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_10000.000000_16_0.010000_1/checkpoint.pt.best"
	# scale_nat = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"

	constants = utils.CIFAR10()
	ds = constants.get_dataset()

	# Load model
	def populate(model_path, scale_path, colors, prefix):
		model_kwargs = {
			'arch': 'vgg19',
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

		# Extract final weights matrix from model
		with ch.no_grad():
			# weights = model.state_dict().get("module.model.linear.weight") # n_classes * n_features vector
			weights = model.state_dict().get("module.model.classifier.weight") # n_classes * n_features vector

		for i in range(weights.shape[0]):
			plt.scatter(x, weights[i][indices].cpu().numpy(), label='class ' + str(i), alpha=0.7, s=1)

	populate(model_nat, scale_nat, (('turquoise', 'blue'), ('lightcoral', 'red')), 'nat')
	plt.title('CIFAR-10 (Linf)')
	# plt.legend()
	plt.savefig("just_weights.png")
