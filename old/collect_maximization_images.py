import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
from robustness.tools.misc import log_statement
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

import utils


def custom_optimization(model, inp_og, iters=100,
	fake_relu=True, indices=None,
	clip_min=0, clip_max=1, lr=0.01):
	inp =  inp_og.clone().detach().requires_grad_(True)
	optimizer = ch.optim.Adamax([inp], lr=lr)
	iterator = range(iters)

	for i in iterator:

		# Get image rep
		rep, _ = model(inp, with_latent=True, fake_relu=fake_relu, just_latent=True)
		rep_  = rep.view(rep.shape[0], -1)

		ch_indices = ch.from_numpy(np.array(indices)).cuda()
		loss = 1.1 ** (-rep_.gather(1, ch_indices.view(-1, 1))[:, 0])

		# Back-prop loss
		optimizer.zero_grad()
		loss.backward(ch.ones_like(loss), retain_graph=True)
		optimizer.step()

		# Clamp back to norm-ball
		with ch.no_grad():
			inp.data  = ch.clamp(inp.data, clip_min, clip_max)

	return ch.clamp(inp.clone().detach(), clip_min, clip_max)


def find_impostors(model, real, iters=200,
	fake_relu=True, indices=None, lr=0.01):

	with ch.no_grad():
		real_rep, _ = model(real, with_latent=True, fake_relu=fake_relu, just_latent=True)

		# Take note of indices where neuron is activated
		activated_indices = []
		for i, x in enumerate(indices):
			if real_rep[0][x] > 0:
			# if real_rep_[len(activated_indices)][x] > 0:
				activated_indices.append(i)

		if len(activated_indices) == 0: return None

		# If too many activated indices, keep only the ones that batch-size allows
		activated_indices = activated_indices[:real_rep.shape[0]]

	# Map indices accordingly
	indices_ = [indices[i] for i in activated_indices[:real.shape[0]]]
	indices = indices_[:]

	# Keep only activated indices
	real_ = real[:len(activated_indices)]
	return custom_optimization(model, real_, iters=iters,
		fake_relu=fake_relu, indices=indices, lr=lr)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='linf', help='type of model (nat/l2/linf)')
	parser.add_argument('--iters', type=int, default=250, help='number of iterations')
	parser.add_argument('--bs', type=int, default=250, help='batch size while performing attack')
	parser.add_argument('--lr', type=float, default=0.01, help='lr for optimizer')
	parser.add_argument('--dataset', type=str, default='binary', help='dataset: one of [binary, cifar10, imagenet, robustcifar]')
	parser.add_argument('--save_path', type=str, default='/p/adversarialml/as9rw/generated_images_binary/', help='path to save generated images')
	parser.add_argument('--seed_mode_normal', type=bool, default=False, help='use normal images as seeds instead of gray images?')
	parser.add_argument('--sample_ratio', type=float, default=1.0, help='how much of the test set (class balanced) to be used when generating images?')
	
	args = parser.parse_args()
	utils.flash_utils(args)
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	batch_size      = args.bs
	iters           = args.iters
	fake_relu       = ('vgg' not in model_arch)
	save_path       = args.save_path
	lr              = args.lr
	gray_mode       = not args.seed_mode_normal
	sample_ratio    = args.sample_ratio

	# Load model
	img_side, num_feat, n_classes = 32, 512, 10
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
		mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
		img_side, num_feat, n_classes = 224, 2048, 1000
		mappinf = [str(x) for x in range(1000)]
	elif args.dataset == 'binary':
		# n_classes = 2
		constants = utils.BinaryCIFAR(None)
		mappinf = [str(x) for x in range(2)]
	else:
		print("Invalid Dataset Specified")

	# Load model
	model = constants.get_model(model_type , model_arch, parallel=True)

	# For extraction of data satisfying property
	# test_data, test_labels = utils.load_all_loader_data(data_loader)
	# Get cat images
	# which_one = 369
	# dog_image = test_data[test_labels == 6][which_one:which_one+1].numpy()

	glob_counter = 1
	if gray_mode:
		gray_vales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
		for gv in gray_vales:
			for i in tqdm(range(0, num_feat, batch_size)):
	
				gray_image = ch.zeros((batch_size, 3, img_side, img_side)).cuda() + gv

				# Use own image
				# loaded_image = np.asarray(Image.open("./visualize/sky_animal_grass_template.jpg")).astype('float32') / 255
				# loaded_image = np.expand_dims(np.transpose(loaded_image, (2, 0, 1)), 0)
				# loaded_image = dog_image
				# loaded_image = np.tile(loaded_image, (batch_size, 1, 1, 1))
				# gray_image = ch.from_numpy(loaded_image).cuda()

				indices = list(range(i, min(i + batch_size, num_feat)))
				impostors = find_impostors(model, gray_image, iters=iters,
										fake_relu=fake_relu, indices=indices, lr=lr)

				if impostors is None: continue
				impostors = impostors.cpu().numpy().transpose(0, 2, 3, 1)

				# Save to appropriate folders
				for k, impostor in enumerate(impostors):
					im = Image.fromarray((impostor * 255).astype(np.uint8))
					im.save(os.path.join(save_path, str(glob_counter) + ".png"))
					glob_counter += 1

	else:
		# Define and load data loaders
		ds = utils.CIFAR10().get_dataset()
		_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_val=False, only_val=True)
		mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

		# Create class directories
		for mm in mappinf:
			os.mkdir(os.path.join(save_path, mm))

		# If sampling requested, sub-select data (class balanced)
		if sample_ratio < 1:
			images, labels = utils.load_all_loader_data(data_loader)
			images_, labels_ = [], []
			for i in range(n_classes):
				eligible_indices = np.nonzero(labels == i)[:,0]
				np.random.shuffle(eligible_indices) 
				# Pick according to ratio
				picked_indices = eligible_indices[:int(len(eligible_indices) * sample_ratio)]
				images_.append(images[picked_indices])
				labels_.append(labels[picked_indices])
			images = ch.cat(images_)
			labels = ch.cat(labels_)

			use_ds = utils.BasicDataset(images, labels)
			data_loader = DataLoader(use_ds, batch_size=batch_size, shuffle=False, num_workers=8)

		for (images, labels) in tqdm(data_loader, total=len(data_loader)):

			# Per example
			for j in range(images.shape[0]):

				# Batches of neurons
				for i in range(0, num_feat, batch_size):
					image_template = ch.empty_like(images)
					for k in range(images.shape[0]): image_template[k] = images[j].clone()
					image_template = image_template.cuda()

					indices = list(range(i, min(i + batch_size, num_feat)))
					impostors = find_impostors(model, image_template, iters=iters,
												fake_relu=fake_relu, indices=indices, lr=lr)

					if impostors is None: continue
					impostors = impostors.cpu().numpy().transpose(0, 2, 3, 1)

					# Save to appropriate folders
					for impostor in impostors:
						im = Image.fromarray((impostor * 255).astype(np.uint8))
						im.save(os.path.join(save_path, mappinf[labels[j]], str(glob_counter) + ".png"))
						glob_counter += 1
