import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

ds = CIFAR()

model_path = sys.argv[1]
filename = sys.argv[2]

model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': model_path
}

model, _ = make_and_restore_model(**model_kwargs)
model.eval()

batch_size = 1024
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

# Extract final weights matrix from model
weights = None
for name, param in model.state_dict().items():
	if name == "module.model.linear.weight":
		weights = param
		break

# Extract bias (just because)
bias = None
for name, param in model.state_dict().items():
	if name == "module.model.linear.bias":
		bias = param
		break

def classwise_closed_form_solutions(logits, weights):
	# Iterate through all possible classes, calculate flip probabilities
	actual_label = ch.argmax(logits)
	delta_values = logits[actual_label] - logits
	delta_values /= weights - weights[actual_label]
	delta_values[actual_label] = np.inf
	return delta_values


n_features = weights.shape[1]
sensitivities = {}
# Get batches of data
for (im, label) in test_loader:
	with ch.no_grad():
		(logits, features), _ = model(im.cuda(), with_latent=True)
	# For each data point in batch
	for j, logit in tqdm(enumerate(logits)):
		# For each feature
		for i in range(n_features):
			specific_weights = weights[:, i]
			# Get sensitivity values across classes
			sensitivity = classwise_closed_form_solutions(logit, specific_weights)
			# print()
			# Check wx+b before and after delta noise
			# unsqueezed_features = features[j].unsqueeze(1)
			# before = ch.mm(weights, unsqueezed_features).squeeze(1) + bias
			# print(before)
			# feature_modified = unsqueezed_features.clone()
			# feature_modified[i] += sensitivity[7]
			# after = ch.mm(weights, feature_modified).squeeze(1) + bias
			# print(after)
			# print(sensitivity)
			# print()

			# Only consider delta values that correspond to valud ReLU range, register others as 'inf'
			valid_sensitivity = sensitivity[features[j][i] + sensitivity >= 0]
			best_delta = ch.argmin(ch.abs(valid_sensitivity))
			best_sensitivity = valid_sensitivity[best_delta]
			best_sensitivity = best_sensitivity.cpu().numpy()
			sensitivities[i] = sensitivities.get(i, []) + [best_sensitivity]

with open("%s.txt" % filename, 'w') as f:
	for i in range(n_features):
		floats_to_string = ",".join([str(x) for x in sensitivities[i]])
		f.write(floats_to_string + "\n")
