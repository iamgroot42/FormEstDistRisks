# Search for instances from the training set that are closest to test inputs in latent space (via various layers)
# Also check for ones that correspond to delta values
import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def get_representations_and_labels(data_loader):
	# Iterate through data loader, cache latent space representations of all images
	representations, labels = [], []
	for (im, label) in tqdm(data_loader):
		with ch.no_grad():
			latent, _ = model(im.cuda(), with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
			representations.append(latent.cpu())
			labels.append(label)

	representations = ch.cat(representations, 0)
	labels          = ch.cat(labels, 0)

	return representations, labels


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='nat', help='type of model (nat/l2/linf)')
	parser.add_argument('--inject', type=int, default=None, help='index of layers, to the output of which delta is to be added')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	fake_relu       = (model_arch != 'vgg19')
	inject          = args.inject
	batch_size      = 1000

	constants = utils.CIFAR10()
	ds = constants.get_dataset()

	# Load model
	model = constants.get_model(model_type , model_arch)
	# Get stats for neuron activations
	if inject:
		senses_raw  = utils.get_sensitivities("./generic_deltas_%s/%d.txt" % (model_type, inject))
		(mean, std) = utils.get_stats("./generic_stats/%s/%d/" % (model_type, inject))
	else:
		senses_raw  = constants.get_deltas(model_type, model_arch)
		(mean, std) = constants.get_stats(model_type, model_arch)
	
	print("==> Loaded delta vectors")

	# Process and make senses
	if inject:
		mean = mean.flatten()
		std  = std.flatten()

	target_image_index = 6
	easiest = np.argsort(np.abs(senses_raw) / (np.expand_dims(std, 1)), axis=0)
	easiest = easiest[:, target_image_index]

	delta_target_index = 10
	print(easiest[delta_target_index], " culprit")
	delta_vector = np.zeros_like(easiest)
	delta_vector[easiest[delta_target_index]] = senses_raw[target_image_index, easiest[delta_target_index]]
	
	train_loader, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, fixed_train_order=True, fixed_test_order=True, data_aug=False)

	# Compute latent space representations
	test_reps, test_labels   = get_representations_and_labels(test_loader)
	train_reps, train_labels = get_representations_and_labels(train_loader)
	
	# Get data loaders again
	train_loader, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, fixed_train_order=True, fixed_test_order=True, data_aug=False)	

	test_latent, test_label = test_reps[target_image_index], test_labels[target_image_index]

	i = 0
	# Store image which you selected
	for (images, labels) in test_loader:
		for im in images:
			if i == target_image_index:
				starting_test_image = im
				break
			i += 1
		if i == target_image_index:
			break

	# Then, pass throuh train set, compute latent space representations, find top-X closest matches and display them
	indices, distances = [], []
	for i, x in tqdm(enumerate(train_reps)):
		# if train_labels[i] == 9:
		# if train_labels[i] != test_label:
		if 1 == 1:
			indices.append(i)
			# distances.append(ch.norm(test_latent - x, 2))
			distances.append(ch.norm((test_latent + delta_vector - x)[easiest[delta_target_index]], 2))
	# distances = [ch.norm(x - test_latent, 2) for x in train_reps]
	show_size = 100
	best_matches = np.argsort(distances)[:show_size]
	best_matches = np.array(indices)[best_matches]
	best_images, best_labels = [], []
	i = 0
	for (images, labels) in train_loader:
		for im, lab in zip(images, labels):
			if i in best_matches:
				best_images.append(im)
				best_labels.append(lab)
			i += 1
		if len(best_images) == len(best_matches):
				break


	# i_want_this = 0
	
	# iterator = tqdm(enumerate(test_loader))
	# for num, (image, label) in iterator:
	# 	if num < i_want_this:
	# 		continue
	# 	image, label = image.cuda(), label.cuda()

	# 	for j in range(image.shape[0]):
	# 		image[j] = image[index_focus]
	# 		label[j] = label[index_focus]

	# 	asr        += ch.sum(succeeded).float()
	# 	index_base += len(image)


	# 	bad_labels = ch.argmax(model(impostors)[0], 1)
	best_images.append(starting_test_image)
	best_labels.append(test_label)
	best_images = ch.stack(best_images, 0)

	best_labels = [x.item() for x in best_labels]

	# 	image_labels = [label.cpu().numpy(), bad_labels.cpu().numpy()]
	# 	# image_labels[0] = [ds.class_names[i] for i in image_labels[0]]
	# 	# image_labels[1] = [ds.class_names[i] for i in image_labels[1]]

	show_image_row([best_images],
				["Images from Train Set"],
				tlist=[best_labels],
				fontsize=22,
				filename="./visualize/delta_nn_matches.png")


	# 	if inject:
	# 		delta_actual = (latent_pert - latent).cpu().view(latent.shape[0], -1).numpy().T
	# 	else:
	# 		delta_actual = (latent_pert - latent).cpu().numpy().T

	# 	achieved = delta_actual >= senses_raw[:, index_base: index_base + len(image)]
	# 	print(np.sum(achieved, 0))

	# 	satisfied = []
	# 	for ii, kk in enumerate(easiest[0, index_base: index_base + len(image)]):
	# 		satisfied.append(1 * (delta_actual[kk, ii] >= senses_raw[kk, ii]))

	# 	print(satisfied)

	# 	exit(0)
