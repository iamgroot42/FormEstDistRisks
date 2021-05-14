import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

ds_path    = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
ds = GenericBinary(ds_path)

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


def binary_closed_form_solution(logits, weights):
	# If same weights, class can never be changed
	if weights[0] == weights[1]:
		return np.inf

	delta = (logits[0] - logits[1]) / (weights[1] - weights[0])
	return delta


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
			sensitivity = binary_closed_form_solution(logit, specific_weights)
			# print()
			# Check wx+b before and after delta noise
			# unsqueezed_features = features[j].unsqueeze(1)
			# before = ch.mm(weights, unsqueezed_features).squeeze(1) + bias
			# print(before)
			# feature_modified = unsqueezed_features.clone()
			# feature_modified[i] += sensitivity
			# after = ch.mm(weights, feature_modified).squeeze(1) + bias
			# print(after)
			# print(sensitivity)
			# print()

			# If new value after perturbed violates ReLU range, register as 'inf'
			if features[j][i] + sensitivity < 0:
				sensitivity = np.inf
			elif sensitivity != np.inf:
				sensitivity = sensitivity.cpu().numpy()
			sensitivities[i] = sensitivities.get(i, []) + [sensitivity]

with open("%s.txt" % filename, 'w') as f:
	for i in range(n_features):
		floats_to_string = ",".join([str(x) for x in sensitivities[i]])
		f.write(floats_to_string + "\n")
