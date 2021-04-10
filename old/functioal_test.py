import os
import torch as ch
from robustness.tools.misc import log_statement
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image

import utils


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='linf', help='type of model (nat/l2/linf)')
	parser.add_argument('--bs', type=int, default=512, help='batch size while performing attack')
	parser.add_argument('--dataset', type=str, default='binary', help='dataset: one of [binary, cifar10, imagenet, robustcifar]')
	parser.add_argument('--sample_ratio', type=float, default=-1, help='sample ratio for f()?')
	parser.add_argument('--sample_size', type=int, default=0, help='how many samples total to be used?')
	
	args = parser.parse_args()
	utils.flash_utils(args)
	
	model_arch   = args.model_arch
	model_type   = args.model_type
	batch_size   = args.bs
	fake_relu    = ('vgg' not in model_arch)
	sample_ratio = args.sample_ratio
	sample_size  = args.sample_size

	# Load model
	img_side, num_feat = 32, 512
	mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
		img_side, num_feat = 224, 2048
		mappinf = [str(x) for x in range(1000)]
	elif args.dataset == 'binary':
		constants = utils.BinaryCIFAR(None)
		# mappinf = [str(x) for x in range(2)]
	else:
		print("Invalid Dataset Specified")

	# Load model
	model = constants.get_model(model_type , model_arch, parallel=True)

	# Define and load data loaders
	ds = utils.CIFAR10().get_dataset()
	_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_val=True, only_val=True)

	# If sampling requested, sub-select data (class balanced)
	if sample_size > 0:
		images, labels = utils.load_all_loader_data(data_loader)

		# Negative sample ratio means all classes balanced
		if sample_ratio < 0:
			images, labels = [], []
			for i in range(10):
				satisfy_indices = np.nonzero(labels == i)[:, 0]
				np.random.shuffle(satisfy_indices)
				satisfy_size = int(sample_size / 10)
				images.append(images[satisfy_indices[:satisfy_size]])
				labels.append(labels[satisfy_indices[:satisfy_size]])
			images = ch.cat([satisfy_images, nosatisfy_images])
			labels = ch.cat([satisfy_labels, nosatisfy_labels])

		else:
			# Pick indices corresponding to f(x)=1 ,ie dog
			satisfy_indices = np.nonzero(labels == 5)[:,0]
			np.random.shuffle(satisfy_indices) 
			# Pick indices corresponding to f(x)=0, ie not dog
			nosatisfy_indices = np.nonzero(labels != 5)[:,0]
			np.random.shuffle(nosatisfy_indices)

			satisfy_size = int(sample_size * sample_ratio)
			satisfy_images   = images[satisfy_indices[:satisfy_size]]
			satisfy_labels   = labels[satisfy_indices[:satisfy_size]]
			nosatisfy_images = images[nosatisfy_indices[:sample_size - satisfy_size]]
			nosatisfy_labels = labels[nosatisfy_indices[:sample_size - satisfy_size]]

			print("Mixing %d and %d samples" % (len(satisfy_images), len(nosatisfy_images)))
			images = ch.cat([satisfy_images, nosatisfy_images])
			labels = ch.cat([satisfy_labels, nosatisfy_labels])

		# Redefine data loaders, using ratio-data
		use_ds = utils.BasicDataset(images, labels)
		data_loader = DataLoader(use_ds, batch_size=batch_size, shuffle=True, num_workers=8)

	# correct, total = 0., 0.
	n_known = 10
	correct, total = np.zeros(n_known), np.zeros(n_known)
	mapping = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
	for (images, labels) in tqdm(data_loader, total=len(data_loader)):
		# Get model prediction
		logits, _ = model(images.cuda(), fake_relu=fake_relu)
		preds = ch.argmax(logits, 1)

		# Record class-wise accuracies
		for j in range(n_known):
			picked_indices = labels == j
			labels_ = labels[picked_indices]
			labels_[:] = mapping[j]

			correct[j] += (labels_.cuda() == preds[picked_indices]).sum()
			total[j]   += labels_.shape[0]

		# Compare with ground-truth labels
		# for i, l in enumerate(labels):
		# 	labels[i] = mapping[l]
		# labels = labels.cuda()

		# correct += (labels == preds).sum()
		# total   += labels.shape[0]

				
	for j in range(n_known):
		print("Class %s : %.2f%%" % (mappinf[j], 100 * correct[j] / total[j]))
	# print("Accuracy on ratio %.2f is %.2f%%" % (sample_ratio, 100 * correct / total))
