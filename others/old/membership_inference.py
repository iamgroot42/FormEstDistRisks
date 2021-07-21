import torch as ch
from robustness.datasets import CIFAR
import utils
import numpy as np
from scipy import spatial


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def custom_optimization(model, inp_og, target_rep, indices, iters=100):
	inp =  inp_og.clone().detach().requires_grad_(True)
	optimizer = ch.optim.Adamax([inp], lr=0.1)
	iterator = range(iters)
	targ_ = target_rep.view(target_rep.shape[0], -1)
	# use_best behavior
	best_loss, best_x = (ch.zeros(inp.shape[0], dtype=ch.float64) + np.inf).cuda(), ch.zeros(inp.shape).cuda()

	indices_mask = ch.zeros_like(targ_).cuda()
	for i, x in enumerate(indices):
		indices_mask[i][x] = 1

	for i in iterator:

		# Get image rep
		rep, _ = model(inp, with_latent=True, fake_relu=False, just_latent=True)
		rep_  = rep.view(rep.shape[0], -1)

		# Old Loss Term
		# loss = ch.div(ch.norm(rep_ - targ_, dim=1), ch.norm(targ_, dim=1))
		loss = ch.norm(rep_ - targ_, dim=1)

		
		aux_loss = ch.sum(ch.abs((rep_ - targ_) * indices_mask), dim=1)
		# aux_loss = ch.div(aux_loss, ch.norm(targ_ * indices_mask, dim=1))
		loss = loss + aux_loss

		# Back-prop loss
		optimizer.zero_grad()
		loss.backward(ch.ones_like(loss), retain_graph=True)
		optimizer.step()

		# Clamp back
		with ch.no_grad():
			inp.data  = ch.clamp(inp.data, 0, 1)
		
		# iterator.set_description('Loss : %f' % loss.mean())

	return ch.clamp(inp.clone().detach(), 0, 1)


def find_impostors(model, delta_vec, real, real_rep, indices, iters=100):
	# Shift delta_vec to GPU
	delta_vec = ch.from_numpy(delta_vec).cuda()

	with ch.no_grad():
		target_rep  = real_rep + delta_vec

	real_ = real.clone().detach()

	impostors = custom_optimization(model, real_, target_rep, indices, iters=iters)

	return impostors


def process_delta_values(x_batch, deltas, easiest, std, specific_indices):
	senses  = np.zeros((x_batch.shape[0], deltas.shape[0]))
	indices = []
	active_condition = (x_batch > 0).cpu().numpy()
	
	for j, i in enumerate(specific_indices):
		easiest_wanted = easiest[:, i]
		# Filter out unwanted neurons, unachievable values, non-activated neurons
		condition = np.logical_and((deltas[easiest_wanted, i] != np.inf), (std != 0), active_condition[j])
		easiest_wanted = easiest_wanted[condition]
		# Pick top (k?)
		which_neuron = easiest_wanted[0]
		senses[j][which_neuron] = deltas[which_neuron][j]
		indices.append(which_neuron)

	return senses, np.array(indices)


def get_what_you_want(X, normalized_deltas, deltas, std, batch_size):
	# Get latent vector reps for test data
	criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()
	delta_rankings = np.argsort(normalized_deltas, 0)
	l2ds, csds, losses = [], [], []
	for i in tqdm(range(0, len(X), batch_size)):
		# Ready stuff
		x_batch = ch.from_numpy(X[i:i + batch_size]).cuda()
		with ch.no_grad():
			(labels, x_batch_rep), _ = model(x_batch, with_latent=True, fake_relu=False)
			labels = ch.argmax(labels, 1)
			delta_vec, indices = process_delta_values(x_batch_rep, deltas, delta_rankings, std, specific_indices=range(i, i+batch_size))
			
		# Get optimized image
		test_imps = find_impostors(model, delta_vec, x_batch, x_batch_rep, indices, iters=100)

		# Get its representation
		with ch.no_grad():
			(output, x_batch_rep_opt), _ = model(test_imps, with_latent=True, fake_relu=False)

		# Measure cosine distance, L2 distance
		l2_dist = ch.norm(x_batch_rep_opt - x_batch_rep, p=2, dim=1).cpu().numpy()
		csd     = np.array([spatial.distance.cosine(x, y) for (x,y) in zip(x_batch_rep.cpu().numpy(), x_batch_rep_opt.cpu().numpy())])
		loss    = criterion(output, labels).cpu().numpy()

		# Collect this data
		l2ds.append(l2_dist)
		csds.append(csd)
		losses.append(loss)

		# if i > 1000:
		# 	break

	return np.concatenate(l2ds), np.concatenate(csds), np.concatenate(losses)


if __name__ == "__main__":

	# Ready data loaders
	constants = utils.CIFAR10()
	ds = constants.get_dataset()

	# Get loaders
	n_classes = len(ds.class_names)
	train_loader, test_loader = ds.make_loaders(workers=10, batch_size=256, data_aug=False)

	# Read train, test datasets
	X_train, Y_train = utils.load_all_loader_data(train_loader)
	X_test, Y_test   = utils.load_all_loader_data(test_loader)
	X_train, Y_train = X_train.numpy(), Y_train.numpy()
	X_test, Y_test   = X_test.numpy(), Y_test.numpy()

	# Load stats, delta values
	delta_train  = utils.get_sensitivities("./deltas_train_cifar10_linf.npy", numpy=True)
	delta_test   = constants.get_deltas("linf", "vgg19", numpy=True)
	(mean, std)  = constants.get_stats("linf", "vgg19")

	# Normalize delta values
	normalised_delta_train = np.abs((delta_train - np.expand_dims(mean, 1)) / np.expand_dims(std, 1))
	normalised_delta_test  = np.abs((delta_test  - np.expand_dims(mean, 1)) / np.expand_dims(std, 1))

	# Load model
	model = constants.get_model("linf", "vgg19")

	# Get latent vector reps for test data
	batch_size = 1000
	test_l2_stats, test_cosine_stats, test_losses = get_what_you_want(X_test, normalised_delta_test, delta_test, std, batch_size)

	# Run multiple runs to capture variance
	num_runs = 5
	for _ in range(num_runs):
		# Randomly sample from training data
		permuted_indices = np.random.permutation(len(X_train))[:len(X_test)]
		X_train_p, Y_train_p =X_train[permuted_indices], Y_train[permuted_indices]

		# Get corresponding delta values
		normalised_delta_train_p = normalised_delta_train[:, permuted_indices]
		delta_train_p = delta_train[:, permuted_indices]

		train_l2_stats, train_cosine_stats, train_losses = get_what_you_want(X_train_p, normalised_delta_train_p, delta_train_p, std, batch_size)

		# Split into train-test at 50:50
		train_data = train_losses
		test_data  = test_losses
		# train_data = train_cosine_stats
		# test_data  = test_cosine_stats
		# train_data = train_l2_stats
		# test_data  = test_l2_stats

		permute_1 = np.random.permutation(len(train_data))
		permute_2 = np.random.permutation(len(test_data))
		X_train = np.concatenate([train_data[permute_1[:len(train_data) // 2]], test_data[permute_2[:len(test_data) // 2]]])
		X_test  = np.concatenate([train_data[permute_1[len(train_data) // 2:]], test_data[permute_2[len(test_data) // 2:]]])
		Y       = np.concatenate([np.ones((len(train_data) // 2)), np.zeros((len(test_data) // 2))])

		X_train = X_train.reshape(-1, 1)
		X_test  = X_test.reshape(-1, 1)

		from sklearn.ensemble import RandomForestClassifier
		from sklearn import metrics
		clf = RandomForestClassifier(max_depth=4, random_state=0)
		print(X_train.shape, X_test.shape, Y.shape)
		clf.fit(X_train, Y)
		scores = clf.predict_proba(X_test)[:, 1]

		fpr, tpr, thresholds = metrics.roc_curve(Y, scores)

		plt.plot(fpr, tpr, label='ROC curve')
		plt.xlim([0.0, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver operating characteristic example')
		# plt.legend(loc="lower right")
		# plt.show()

		# train_l2_stats = sorted(train_l2_stats)
		# test_l2_stats  = sorted(test_l2_stats)
		# plt.plot(range(len(train_l2_stats)), train_l2_stats, label='train-l2')
		# plt.plot(range(len(test_l2_stats)), test_l2_stats, label='test-l2')

		# train_cosine_stats = sorted(train_cosine_stats)
		# test_cosine_stats  = sorted(test_cosine_stats)
		# plt.plot(range(len(train_cosine_stats)), train_cosine_stats, label='train-cosine')
		# plt.plot(range(len(test_cosine_stats)), test_cosine_stats, label='test-cosine')
	
		plt.grid(True)
		plt.legend()
		plt.savefig("./visualize/threshold_membership_try.png")

		exit(0)
