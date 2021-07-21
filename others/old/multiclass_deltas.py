import torch as ch
import utils
import numpy as np
from tqdm import tqdm


def get_sensitivities(model, data_loader, weights, bias, validity_check_exit=False, want_all=False):
	n_features = weights.shape[1]
	sensitivities = [] if want_all else {}
	# Get batches of data
	for (im, label) in data_loader:
		with ch.no_grad():
			(logits, features), _ = model(im.cuda(), with_latent=True)
		# For each data point in batch
		for j, logit in tqdm(enumerate(logits)):
			# For each feature
			all_together = utils.compute_delta_values(logit, weights, label[j])

			focus_features = features[j].unsqueeze(0).repeat(all_together.shape[0], 1)
			
			# Only consider delta values that correspond to valud ReLU range, register others as 'inf'
			invalid_deltas = all_together + focus_features < 0
			all_together[invalid_deltas] = np.inf

			if want_all:
				all_together = all_together.cpu().numpy().T
				sensitivities.append(all_together)
			else:
				best_chosen = ch.argmin(ch.abs(all_together), 0)
				all_together = all_together.cpu().numpy()

				for i, x in enumerate(best_chosen):
					sensitivities[i] = sensitivities.get(i, []) + [all_together[x][i]]


			# sensitivities[i] = wanted_deltas.cpu().numpy()

			# for i in range(n_features):
			# 	specific_weights = weights[:, i]
			# 	# Get sensitivity values across classes
			# 	sensitivity = utils.compute_delta_values(logit, specific_weights, label[j])
			# 	if validity_check_exit:
			# 		print()
			# 		# # Check wx+b before and after delta noise
			# 		unsqueezed_features = features[j].unsqueeze(1)
			# 		before = ch.mm(weights, unsqueezed_features).squeeze(1) + bias
			# 		print(before)
			# 		feature_modified = unsqueezed_features.clone()
			# 		feature_modified[i] += sensitivity[8]
			# 		after = ch.mm(weights, feature_modified).squeeze(1) + bias
			# 		print(after)
			# 		print(sensitivity)
			# 		print()
			# 		exit()

			# 	# Only consider delta values that correspond to valud ReLU range, register others as 'inf'
			# 	valid_sensitivity = sensitivity[features[j][i] + sensitivity >= 0]
			# 	best_delta = ch.argmin(ch.abs(valid_sensitivity))
			# 	best_sensitivity = valid_sensitivity[best_delta]
			# 	best_sensitivity = best_sensitivity.cpu().numpy()
			# 	sensitivities[i] = sensitivities.get(i, []) + [best_sensitivity]

	return sensitivities


if __name__ == "__main__":
	import sys
	filename   = sys.argv[1]
	model_arch = sys.argv[2]
	model_type = sys.argv[3]
	dataset    = sys.argv[4]

	batch_size = 10000
	if dataset == "cifar":
		dx = utils.CIFAR10()
	elif dataset == "imagenet":
		dx = utils.ImageNet1000()
		batch_size = 512
	elif dataset == "svhn":
		dx = utils.SVHN10()
	elif dataset == 'binary':
		# dx = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary_nodog/")
		dx = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary/")
		# dx = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary_10pdog/")
		# dx = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary_50pdog/")
	else:
		raise ValueError("Dataset not supported")

	model = dx.get_model(model_type, model_arch, parallel=True)
	# model = ch.nn.DataParallel(model)
	
	# Use CIFAR10 test set (balanced) when generating
	dx = utils.CIFAR10()
	ds = dx.get_dataset()

	want_all = False#True#False
	_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
	# data_loader, _ = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_train=False, data_aug=False)

	weight_name   = utils.get_logits_layer_name(model_arch)
	weights       = utils.get_these_params(model, weight_name)
	bias          = utils.get_these_params(model, weight_name.rsplit(".", 1)[0] + "bias")
	sensitivities = get_sensitivities(model, data_loader, weights, bias, want_all=want_all)

	if want_all:
		sensitivities = np.array(sensitivities)
		print(sensitivities.shape)
		np.save(filename, sensitivities)
	else:
		with open("%s.txt" % filename, 'w') as f:
			for i in range(weights.shape[1]):
				floats_to_string = ",".join([str(x) for x in sensitivities[i]])
				f.write(floats_to_string + "\n")
