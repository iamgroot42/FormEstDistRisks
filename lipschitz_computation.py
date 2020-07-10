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


def get_lipschitz(model, data_loader, weights, bias, validity_check_exit=False):
	n_features = weights.shape[1]
	distances = []

	weight_sums = []
	for i in range(weights.shape[0]): weight_sums.append(ch.sum(ch.abs(weights[i])).cpu().numpy())

	# Get batches of data
	for (im, label) in tqdm(data_loader):
		with ch.no_grad():
			logits, _ = model(im.cuda())
			predicted_labels = ch.argmax(logits, 1)

			deltas   = ch.zeros_like(logits)
			for j in range(logits.shape[0]):
				these_class_distances = []
				for i in range(logits.shape[1]):
					if i == predicted_labels[j]:
						continue
					logits_y     = logits[j, predicted_labels[j]]
					logits_y_cap = logits[j, i]
					logits_diff  = (logits_y - logits_y_cap).cpu().numpy()
					these_class_distances.append(logits_diff / weight_sums[i])
				distances.append(np.min(these_class_distances))

	return distances


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

	ds    = dx.get_dataset()
	model = dx.get_model(model_type, model_arch)

	batch_size = 512
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	weight_name = utils.get_logits_layer_name(model_arch)
	weights = get_these_params(model, weight_name)
	bias    = get_these_params(model, weight_name.rsplit(".", 1)[0] + "bias")
	lipschitz = get_lipschitz(model, test_loader, weights, bias)

	with open("%s.txt" % filename, 'w') as f:
		for i in range(weights.shape[1]):
			f.write(str(lipschitz[i]) + "\n")
