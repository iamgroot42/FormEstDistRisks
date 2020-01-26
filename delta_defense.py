import torch as ch
import numpy as np
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
import dill

import utils


if __name__ == "__main__":

	model_path   = "/p/adversarialml/as9rw/models_correct/edit_this.pt"
	dataset_path = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
	sense_path   = "/p/adversarialml/as9rw/binary_stats/nat/deltas_nat.txt"
	scale_path   = "/p/adversarialml/as9rw/binary_stats/nat/stats/"

	ds = GenericBinary(dataset_path)
	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}

	# Load model
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# Get scaled delta values
	senses   = utils.get_sensitivities(sense_path)
	(mean, std) = utils.get_stats(scale_path)
	senses = utils.scaled_values(senses, mean, std)
	senses = np.mean(senses, axis=1)

	# Scale down worst N delta values by 1/2
	# factor = 4
	# factor = 1 / factor
	factor = 0
	N = 256
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
	# z = model(image_rep, with_latent=-=)
	# (_, image_rep), _  = model(im, with_latent=1)
	# print(image_rep.shape)
	# (_, image_rep), _  = model(im, with_latent=2)
	# print(image_rep.shape)
	# (_, image_rep), _  = model(im, with_latent=True)
	# print(image_rep.shape)
