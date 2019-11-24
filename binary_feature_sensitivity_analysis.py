import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

ds_path    = "./datasets/cifar_binary/animal_vehicle/"
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
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True)

# Extract final weights matrix from model
weights = None
for name, param in model.state_dict().items():
	if name == "module.model.linear.weight":
		weights = param
		break

def closed_form_solution(logits, weights):
	num_clases = logits.shape[0]

	predicted_class = ch.argmax(logits)
	pos_delta_values = [np.inf]
	neg_delta_values = [-np.inf]

	for i in range(num_clases):
		if i == predicted_class:
			continue
		if (weights[i] - weights[predicted_class]) >= 0:
			pos_delta_values.append((logits[predicted_class] - logits[i])/(weights[i] - weights[predicted_class]))
		else:
			neg_delta_values.append((logits[predicted_class] - logits[i])/(weights[i] - weights[predicted_class]))
	return (np.min(pos_delta_values), np.max(neg_delta_values))


n_features = weights.shape[1]
sensitivities = {}
# Get batches of data
for (im, label) in test_loader:
	with ch.no_grad():
		logits, _ = model(im.cuda())
	# For each data point in batch
	for logit in tqdm(logits):
		# For each feature
		for i in range(n_features):
			specific_weights = weights[:, i]
			(x, y) = closed_form_solution(logit, specific_weights)
			sensitivity = float(min(x, -y).cpu().numpy())
			sensitivities[i] = sensitivities.get(i, []) + [sensitivity]

# Dump sensitivity values (mean, std) for each feature
with open("%s.txt" % filename, 'w') as f:
	for i in range(n_features):
		floats_to_string = ",".join([str(x) for x in sensitivities[i]])
		f.write(floats_to_string + "\n")
