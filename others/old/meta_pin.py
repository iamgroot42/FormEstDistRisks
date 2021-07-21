import torch as ch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

# Custom module imports
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
		node_rep = phi(combined_c.cuda()).clone().detach()
		# Compute and keep track of layer-wise reps
		layer_rep = ch.sum(node_rep, 0)
		layer_reps.append(layer_rep)
		# Remember previous node reps
		prev_node_rep = node_rep

	model_rep = ch.cat(layer_reps)
	return model_rep


if __name__ == "__main__":
	# Read all models and store their representations
	reps = []
	labels  = [
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		1, 1,
		1, 1,
		1, 1,
		1, 1,
		1, 1,
		1, 1
	]
	paths = [
		"0p_linf", "0p_linf_2",
		"10p_linf", "10p_linf_2",
		"20p_linf", "20p_linf_2",
		"30p_linf", "30p_linf_2",
		"40p_linf", "40p_linf_2",
		"50p_linf", "50p_linf_2",
		"60p_linf", "60p_linf_2",
		"70p_linf", "70p_linf_2",
		"80p_linf", "80p_linf_2",
		"90p_linf", "90p_linf_2",
		"100p_linf", "100p_linf_2"
	]

	# Use a dummy model to get required dimensionalities
	constants = utils.BinaryCIFAR(None)
	model = constants.get_model(None , "vgg19", parallel=True)
	params = extract_wb(model)

	# Get phi-functions ready
	phi_models = []
	for i, param in enumerate(params):
		rs = param.shape[1]
		if i > 0: rs += params[i-1].shape[0]
		phi_models.append(nn.Sequential(nn.Linear(rs, 1), nn.ReLU()).cuda())

	# Generate PI (permutation invariant) model representations
	prefix = "/p/adversarialml/as9rw/new_exp_models/small/"
	suffix = "checkpoint.pt.best"
	for path in tqdm(paths):
		model = constants.get_model(os.path.join(prefix, path, suffix) , "vgg19", parallel=True)
		params = extract_wb(model)

		model_rep = get_PIN_representations(params, phi_models)
		reps.append(model_rep.cpu().numpy())

		if len(params) == 2: break

	reps = np.array(reps)
	print(reps.shape)

	np.save("model_reps", reps)

	# Classifier on top of PIN
	# rho = nn.Sequential(
	# 	nn.Linear(33, 8),
	# 	nn.ReLU(),
	# 	nn.Linear(8, 1),
	# 	nn.Sigmoid()
	# 	).cuda()

	
	# Use a fewer-parameters rho (a decision tree, perhaps)
	# Train using the 22 training points
	# Quite low, but something like clustering and considering
	# just binary <=50 classification should give decent results?

	# model_rep = rho(model_rep.unsqueeze_(0))
