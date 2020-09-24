import torch as ch
import torch.nn as nn

import utils


def extract_wb(m):
	combined = []
	features  = model.module.model.features
	classifier = model.module.model.classifier

	# Extract feature params
	for layer in features:
		if hasattr(layer, 'weight'):
			w, b = layer.weight.detach(), layer.bias.detach()
			w_, b_ = w.view(w.shape[0], -1), b.view(b.shape[0], -1)
			combined.append(ch.cat((w_, b_), -1))

	# Extract classifier params
	w, b = classifier.weight.detach(), classifier.bias.detach()
	w_, b_ = w.view(w.shape[0], -1), b.view(b.shape[0], -1)
	combined.append(ch.cat((w_, b_), -1))

	return combined


def get_PIN_representations(params, phis):
	# Extract all weights, biases in model
	# Use the same approach as in property inference to combine weight/biases and generate a single
	# Feature representation for the model that is permutation invariant=
	# Should have a phi function for every layer
	assert len(phis) == len(params)

	layer_reps = []
	prev_node_rep = None
	for phi, c in zip(phis, params):
		combined_c = c
		if prev_node_rep is not None:
			prev_nodes = ch.transpose(prev_node_rep.repeat(1, combined_c.shape[0]), 0, 1)
			# print(combined_c.shape)
			combined_c = ch.cat((combined_c, prev_nodes), -1)
			# print(combined_c.shape)
		node_rep = phi(combined_c.cuda())
		# Compute and keep track of layer-wise reps
		layer_rep = ch.sum(node_rep, 0)
		layer_reps.append(layer_rep)
		# Remember previous node reps
		prev_node_rep = node_rep

	model_rep = ch.cat(layer_reps)
	return model_rep


if __name__ == "__main__":
	constants = utils.CIFAR10()
	model = constants.get_model("nat" , "vgg19", parallel=True)
	params = extract_wb(model)

	# Get phi-functions ready
	phi_models = []
	for i, param in enumerate(params):
		rs = param.shape[1]
		if i > 0: rs += params[i-1].shape[0]
		phi_models.append(nn.Sequential(nn.Linear(rs, 1), nn.ReLU()).cuda())

	# Classifier on top of PIN
	rho = nn.Sequential(
		nn.Linear(33, 8),
		nn.ReLU(),
		nn.Linear(8, 1),
		nn.Sigmoid()
		).cuda()

	model_rep = get_PIN_representations(params, phi_models)
	
	# Use a fewer-parameters rho (a decision tree, perhaps)
	# Train using the 22 training points
	# Quite low, but something like clustering and considering
	# just binary <=50 classification should give decent results?

	model_rep = rho(model_rep.unsqueeze_(0))

	print(model_rep)
