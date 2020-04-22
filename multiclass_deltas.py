import torch as ch
import utils
import numpy as np
from tqdm import tqdm


# Extract final weights matrix from model
def get_these_params(model, identifier):
	for name, param in model.state_dict().items():
		if name == identifier:
			return param
	return None

def classwise_closed_form_solutions(logits, weights):
	# Iterate through all possible classes, calculate flip probabilities
	actual_label = ch.argmax(logits)
	delta_values = logits[actual_label] - logits
	delta_values /= weights - weights[actual_label]
	delta_values[actual_label] = np.inf
	return delta_values


def get_sensitivities(model, data_loader, weights, bias, validity_check_exit=False):
	n_features = weights.shape[1]
	sensitivities = {}
	# Get batches of data
	for (im, label) in data_loader:
		with ch.no_grad():
			(logits, features), _ = model(im.cuda(), with_latent=True)
		# For each data point in batch
		for j, logit in tqdm(enumerate(logits)):
			# For each feature
			for i in range(n_features):
				specific_weights = weights[:, i]
				# Get sensitivity values across classes
				sensitivity = classwise_closed_form_solutions(logit, specific_weights)
				if validity_check_exit:
					print()
					# # Check wx+b before and after delta noise
					unsqueezed_features = features[j].unsqueeze(1)
					before = ch.mm(weights, unsqueezed_features).squeeze(1) + bias
					print(before)
					feature_modified = unsqueezed_features.clone()
					feature_modified[i] += sensitivity[8]
					after = ch.mm(weights, feature_modified).squeeze(1) + bias
					print(after)
					print(sensitivity)
					print()
					exit()

				# Only consider delta values that correspond to valud ReLU range, register others as 'inf'
				valid_sensitivity = sensitivity[features[j][i] + sensitivity >= 0]
				best_delta = ch.argmin(ch.abs(valid_sensitivity))
				best_sensitivity = valid_sensitivity[best_delta]
				best_sensitivity = best_sensitivity.cpu().numpy()
				sensitivities[i] = sensitivities.get(i, []) + [best_sensitivity]
	return sensitivities


if __name__ == "__main__":
	import sys
	filename   = sys.argv[1]
	model_arch = sys.argv[2]
	model_type = sys.argv[3]
	dataset    = sys.argv[4]

	if dataset == "cifar":
		dx = utils.CIFAR10()
	elif dataset == "imagenet":
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	batch_size = 1024
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	weight_name = utils.get_logits_layer_name(model_arch)
	weights = get_these_params(model, weight_name)
	bias    = get_these_params(model, weight_name.rsplit(".", 1)[0] + "bias")
	sensitivities = get_sensitivities(model, test_loader, weights, bias)

	with open("%s.txt" % filename, 'w') as f:
		for i in range(n_features):
			floats_to_string = ",".join([str(x) for x in sensitivities[i]])
			f.write(floats_to_string + "\n")
