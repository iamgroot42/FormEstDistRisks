import torch as ch
import numpy as np
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import dill
import sys

import utils


if __name__ == "__main__":

	model_path   = "/p/adversarialml/as9rw/models_cifar10/delta_model.pt"
	sense_path   = "/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt"
	scale_path   = "/p/adversarialml/as9rw/cifar10_stats/nat/stats/"

	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}

	# Scale down worst N delta values by 1/factor
	factor = sys.argv[1]
	if factor == 'inf':
		factor = 0
	else:
		factor = 1 / float(factor)
	N = int(sys.argv[2])

	# Load model
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# Get scaled delta values
	senses   = utils.get_sensitivities(sense_path)
	(mean, std) = utils.get_stats(scale_path)

	# 99.7% interval
	threshold = mean + 3 * std

	# Only consider neurons with any hopes of attackking (delta within some sensible range)
	chuck_these = np.min(np.abs(senses), 1) < threshold
	print("%d candidates identified" % (int(np.sum(chuck_these))))
	senses[np.logical_not(chuck_these)] = np.inf

	senses = utils.scaled_values(senses, mean, std)
	senses = np.mean(senses, axis=1)
	worst_n = np.argsort(np.abs(senses))[:N]

	# Random weight drop-out
	worst_n = np.random.permutation(senses.shape[0])[:N]

	# Extract final weights matrix from model
	with ch.no_grad():
		model.state_dict().get("module.model.linear.weight")[:, worst_n] *= factor

	# Save modified model
	sd_info = {
		'model':model.state_dict(),
		# 'optimizer':opt.state_dict(),
		'epoch': 1
      }
	ch.save(sd_info, model_path, pickle_module=dill)

	# Obtain feature representations, if needed
	# (_, image_rep), _  = model(im, with_latent=1)
	# print(image_rep.shape)
