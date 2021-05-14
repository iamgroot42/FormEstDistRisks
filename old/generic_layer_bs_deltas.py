import torch as ch
import utils
import numpy as np
from tqdm import tqdm


def delta_checks(delta_vec, intermediate_outputs, label, injection_index, mid):
	delta_vec[:, injection_index] = mid
	delta_vec_shaped              = delta_vec.view_as(intermediate_outputs)
	perturbed_input               = intermediate_outputs + delta_vec_shaped

	with ch.no_grad():
		# Pass on modified input (intermediate output + noise) to rest of model
		perturbed_pred, _ = model(perturbed_input, this_layer_input=injection_layer + 1)
		perturbed_pred    = ch.argmax(perturbed_pred, 1)
		unchanged_indices = (perturbed_pred == label)
	return unchanged_indices


# Assuming single flipping point for prediction change for delta-addition to specific neuron's output
def delta_binary_search(model, intermediate_outputs, image, label, injection_point, n_steps, injection_range, low, high, fast=False, positive_range=True):
	injection_layer, injection_index = injection_point
	eps = 1e-1

	low       = ch.zeros(image.shape[0]).cuda() + low
	high      = ch.zeros(image.shape[0]).cuda() + high

	with ch.no_grad():
		delta_vec = ch.zeros((intermediate_outputs.shape[0], injection_range), dtype=high.dtype).cuda()

		# Perform binary search
		for _ in range(n_steps):
			mid = (high + low) / 2

			# Construct delta vector (adding noise to output of specific layer's output)
			unchanged_indices = delta_checks(delta_vec, intermediate_outputs, label, injection_index, mid)

			# Proceed with binary search based on predictions
			if positive_range:
				low[unchanged_indices]   = mid[unchanged_indices]  + eps
				high[~unchanged_indices] = mid[~unchanged_indices] - eps
			else:
				high[unchanged_indices] = mid[unchanged_indices]   - eps
				low[~unchanged_indices] = mid[~unchanged_indices]  + eps

	actually_mid  = mid
	if positive_range:
		actually_mid[unchanged_indices] = high[unchanged_indices]
	else:
		actually_mid[unchanged_indices] = low[unchanged_indices]

	unchanged_indices = delta_checks(delta_vec, intermediate_outputs, label, injection_index, actually_mid)

	actually_mid    = actually_mid
	changed_indices = (~unchanged_indices)
	return (actually_mid, changed_indices)


def get_sensitivities(model, data_loader, injection_layer, n_steps, fast):
	sensitivities = {}
	upper_caps = [-1e10, 1e10]
	if fast: upper_caps = [-1e4, 1e4]
	# if fast: upper_caps = [-1e3, 1e3]
	# Get batches of data
	for (im, label) in data_loader:
		im, label = im.cuda(), label.cuda()
		total, gotcha = 0, 0
		# Get intermediate outputs of model (caching)
		with ch.no_grad():
			intermediate_outputs, _ = model(im, this_layer_output=injection_layer, just_latent=True)
			iss = intermediate_outputs.shape
			injection_range = iss[1] * iss[2] * iss[3]
			upper_caps[0] = - ch.max(intermediate_outputs)
			print("Range for search : [%.2f,%.2f]" % (upper_caps[0], upper_caps[1]))
		iterator  = tqdm(range(injection_range))
		for i in iterator:
			# Delta value when adding
			(delta_value, id_1)          = delta_binary_search(model, intermediate_outputs, im, label, (injection_layer, i), n_steps,
																injection_range, 0.0, upper_caps[1], fast=fast)
			(negative_delta_value, id_2) = delta_binary_search(model, intermediate_outputs, im, label, (injection_layer, i), 2,
																injection_range, upper_caps[0], 0.0, fast=fast, positive_range=False)
			# For cases where both directions succeed, pick the one that has lowest magnitude
			delta_value_l = ch.zeros_like(delta_value) + np.inf
			# If only negative direction worked
			delta_value_l[id_2] = negative_delta_value[id_2]
			# If only positive direction
			delta_value_l[id_1] = delta_value[id_1]
			# If both directions worked
			both_worked = id_1 & id_2
			delta_value_l[both_worked] = ch.where(delta_value[both_worked] > ch.abs(negative_delta_value[both_worked]),
													negative_delta_value[both_worked],
													delta_value[both_worked])
			# Take not of sensitivities
			sensitivities[i] = sensitivities.get(i, []) + list(delta_value_l.cpu().numpy())

			gotcha += ch.sum(id_1 | id_2)
			total  += id_1.shape[0]

			iterator.set_description("Delta Success Rate : %.2f" % (100 * gotcha/ total))
	return sensitivities, injection_range


if __name__ == "__main__":
	import sys
	filename        = sys.argv[1]
	model_arch      = sys.argv[2]
	model_type      = sys.argv[3]
	dataset         = sys.argv[4]
	injection_layer = int(sys.argv[5])
	fast_mode       = sys.argv[6]

	fast = 0
	if fast_mode == "yes": fast = 1

	if dataset == "cifar":
		dx = utils.CIFAR10()
	elif dataset == "imagenet":
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds    = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	# batch_size = 10000
	batch_size = 8000
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	# VGG-19 specific parameters
	# Layer 48, 45, 42:  512 * 2 * 2
	# Layer 38, 35, 32, 29 : 512 * 4 * 4 
	# Layer 25, 22, 19, 16: 256 * 8 * 8
	# Layer 12, 9: 128 * 16 * 16
	# Layer 5, 2: 64 * 32 * 32
	n_steps = 50
	if fast: n_steps = 7#10
	sensitivities, injection_range = get_sensitivities(model, test_loader, injection_layer, n_steps, fast)

	with open("%s.txt" % filename, 'w') as f:
		for i in range(injection_range):
			floats_to_string = ",".join([str(x) for x in sensitivities[i]])
			f.write(floats_to_string + "\n")
