# Read images
# Read sensitivities
# For each attack image, see which neurons actually surpassed the given delta values
import numpy as np
import torch as ch
import utils
from tqdm import tqdm


if __name__ == "__main__":
	import sys
	model_arch  = sys.argv[1]
	model_type  = sys.argv[2]
	adv_ex_path = sys.argv[3]
	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	model = constants.get_model(model_type , model_arch)
	prefix = "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3"
	senses = utils.get_sensitivities(prefix + ".txt")
	# senses = constants.get_deltas(model_type, model_arch)

	attack_images = np.load(adv_ex_path)
	n = 8
	batch_size = 64
	
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)
	iterator = tqdm(test_loader)
	i = 0
	satisfy_neuron = []
	for (image, _) in iterator:
		with ch.no_grad():
			# Get original image latent rep
			(_, image_rep), _  = model(image.cuda(), with_latent=True)
			these_perturbed = ch.from_numpy(attack_images[i * n * batch_size: (i + 1) * n * batch_size])
			# Get latent rep for images with adv noise
			(_, preturbed_reps), _ = model(these_perturbed.cuda(), with_latent=True)
			# Look at delta vector
			deltas_for_them = senses[:, i * batch_size: (i + 1) *  batch_size]
			# Per image, take note of neurons which actually satisfy this condition
			for j in range(image_rep.shape[0]):
				satisfy = []
				for k in range(n):
					delta_diff = (preturbed_reps[j * n + k] - image_rep[j]).cpu().numpy()
					satisfied = np.nonzero(delta_diff >= deltas_for_them[:, j])[0]
					satisfy.append(satisfied)
				satisfy_neuron.append(satisfy)
			i += 1

		tsat = [sum([len(y) for y in x]) for x in satisfy_neuron]
		tsat = sum([x > 0 for x in tsat])
		iterator.set_description('Satisfied (for any delta) rate : %f' % (100.0 * tsat / len(satisfy_neuron)))
