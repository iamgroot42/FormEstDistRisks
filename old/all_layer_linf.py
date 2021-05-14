import torch as ch
import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


# Extract final weights matrix from model
def get_these_params(model, identifier):
	for name, param in model.state_dict().items():
		if name == identifier:
			return param
	return None

def classwise_closed_form_solutions(logits, weights, actual_label):
	numerator   = logits[actual_label] - logits
	denominator = ch.sum(ch.abs(weights[actual_label].unsqueeze(0) -  weights), 1)
	delta_values = numerator / denominator
	delta_values[actual_label] = np.inf
	return delta_values


def get_sensitivities(model, data_loader, weights, bias, validity_check_exit=False):
	n_features = weights.shape[1]
	sensitivities = []
	# Get batches of data
	for (im, label) in data_loader:
		with ch.no_grad():
			(logits, features), _ = model(im.cuda(), with_latent=True)
			model_preds = ch.argmax(logits, 1)
		# For each data point in batch
		for j, logit in tqdm(enumerate(logits)):
			sensitivity = classwise_closed_form_solutions(logit, weights, model_preds[j])
			sensitivities.append(ch.min(sensitivity).cpu().numpy())
	
	return np.array(sensitivities)


if __name__ == "__main__":
	model_types =["nat", "linf"]

	for model_type in model_types:
		dx = utils.CIFAR10()
		ds = dx.get_dataset()
		model = dx.get_model(model_type, "vgg19")

		batch_size = 10000
		_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

		weight_name = utils.get_logits_layer_name("vgg19")
		weights = get_these_params(model, weight_name)
		bias    = get_these_params(model, weight_name.rsplit(".", 1)[0] + "bias")
		sensitivities = get_sensitivities(model, test_loader, weights, bias)

		plt.plot(list(range(batch_size)), sorted(sensitivities), label=model_type)

	plt.legend()
	
	plt.savefig("visualize/overall_linf.png")
