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

def classwise_closed_form_solutions(logits, weights, actual_label):
	# Iterate through all possible classes, calculate flip probabilities
	actual_label = ch.argmax(logits)
	delta_values = logits[actual_label] - logits
	delta_values /= weights - weights[actual_label]
	delta_values[actual_label] = np.inf
	return delta_values


def get_sensitivities(model, data_loader, weights, bias, features_list):
	n_features = weights.shape[1]
	sensitivities = {}
	# Get batches of data
	for (im, label) in data_loader:
		with ch.no_grad():
			(logits, features), _ = model(im.cuda(), with_latent=True)
		# For each data point in batch
		for j, logit in tqdm(enumerate(logits)):
			# For each feature-cluster
			for i, fl in enumerate(features_list):
				specific_weights = weights.t()[fl].t()
				specific_weights = weights.sum(1)
				# Get sensitivity values across classes
				sensitivity = classwise_closed_form_solutions(logit, specific_weights, label[j])

				# Only consider delta values that correspond to valud ReLU range, register others as 'inf'
				specific_features = features.t()[fl].t()
				valid_sensitivity = sensitivity[sensitivity + specific_features[j].min() >= 0]
				best_delta = ch.argmin(ch.abs(valid_sensitivity))
				best_sensitivity = valid_sensitivity[best_delta]
				best_sensitivity = best_sensitivity.cpu().numpy()
				sensitivities[i] = sensitivities.get(i, []) + [best_sensitivity]
	return sensitivities


if __name__ == "__main__":
	import sys
	clusters_f = sys.argv[1]
	model_arch = sys.argv[2]
	model_type = sys.argv[3]
	dataset    = sys.argv[4]
	filename   = sys.argv[5]

	if dataset == "cifar":
		dx = utils.CIFAR10()
	elif dataset == "imagenet":
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	batch_size = 10000
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	weight_name = utils.get_logits_layer_name(model_arch)
	weights = get_these_params(model, weight_name)
	bias    = get_these_params(model, weight_name.rsplit(".", 1)[0] + "bias")

	# Read files of clusters of neurons
	clusters = []
	with open(clusters_f, 'r') as f:
		for line in f:
			clusters.append([int(i) for i in line.rstrip('\n').split(',')])

	sensitivities = get_sensitivities(model, test_loader, weights, bias, clusters)

	with open("%s.txt" % filename, 'w') as f:
		for i in range(len(clusters)):
			floats_to_string = ",".join([str(x) for x in sensitivities[i]])
			f.write(floats_to_string + "\n")
