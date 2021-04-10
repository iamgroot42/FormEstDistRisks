import torch as ch
import numpy as np
from robustness.model_utils import make_and_restore_model
from robustness.tools.helpers import save_checkpoint
import sys

import utils


def chuck_inf_means(senses):
	chucked = []
	for i in range(senses.shape[0]):
		x = senses[i]
		chucked.append(np.mean(x[x != np.inf]))
	return np.array(chucked)


if __name__ == "__main__":

	model_path   = "/p/adversarialml/as9rw/models_cifar10_vgg/delta_model.pt"
	
	m_type    = "linf"
	arch_type = "vgg19"

	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	model_kwargs = {
		'arch': arch_type,
		'dataset': ds,
		'resume_path': model_path
	}

	# Get scaled delta values
	senses = constants.get_deltas(m_type, arch_type)
	(mean, std) = constants.get_stats(m_type, arch_type)

	# Scale down worst N delta values by 1/factor
	random_drop = False
	factor = sys.argv[1]
	if factor == 'inf':
		factor = 0
	else:
		factor = float(factor)
		if factor < 0:
			random_drop = True
			factor = 0
		else:
			factor = 1 / factor
	N = int(sys.argv[2])

	# Load model
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()
	
	print("Dropping %d out of %d neurons" % (N, senses.shape[0]))

	# Random weight drop-out if negative factor
	if random_drop:
		print("Random drop-out! : %d out of %d" % (N, senses.shape[0]))
		worst_n = np.random.permutation(senses.shape[0])[:N]
	else:
		# 99.7% interval
		threshold = mean + 3 * std

		# Only consider neurons with any hopes of attackking (delta within some sensible range)
		chuck_these = np.min(np.abs(senses), 1) < threshold
		print("%d candidates identified" % (int(np.sum(chuck_these))))
		senses = utils.scaled_values(senses, mean, std)
		inf_counts = np.sum(senses == np.inf, 1)
		senses = chuck_inf_means(senses)

		# senses[np.logical_not(chuck_these)] = np.inf
		# 1/30 experiments use this:
		worst_n = np.argsort(np.abs(senses))[:N]
		# 2/6 experiments use this (reverse of above order):
		# worst_n = np.argsort(-np.abs(senses))[:N]
		# Neurons seemingly easiest to fool : drop them first
		# worst_n = np.argsort(inf_counts)[:N]
		# Neurons seemingly easiest to fool : drop them last
		# worst_n = np.argsort(-inf_counts)[:N]
		# Sliding window (on top of 2/6)
		# left = 0
		# print("Window : [%d,%d)" % (left, left + N))
		# worst_n = np.argsort(-np.abs(senses))[left: left + N]
		# Drop N from left AND right
		# worst_n = np.concatenate((np.argsort(-np.abs(senses))[:290], np.argsort(-np.abs(senses))[310:]))
		# worst_n = sorted(list(set(worst_n)))

	# Extract final weights matrix from model
	with ch.no_grad():
		# model.state_dict().get("module.model.classifier.weight")[:, worst_n] *= factor
		model.state_dict().get("module.model.classifier.weight")[:, worst_n] *= factor
		# No need for following call : weights are shared b/w model and classifier
		# model.state_dict().get("module.attacker.model.linear.weight")[:, worst_n] *= factor

	# Save modified model
	sd_info = {
		'model': model.state_dict(),
		# 'optimizer': opt.state_dict(),
		'epoch': 1
	}
	save_checkpoint(sd_info, False, model_path)

	# Obtain feature representations, if needed
	# (_, image_rep), _  = model(im, with_latent=1)
	# print(image_rep.shape)
