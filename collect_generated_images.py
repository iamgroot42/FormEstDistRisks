import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
from robustness.tools.misc import log_statement
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def custom_optimization(model, inp_og, target_rep, iters=100,
	fake_relu=True, inject=None, retain_images=False, indices=None,
	clip_min=0, clip_max=1):
	inp =  inp_og.clone().detach().requires_grad_(True)
	optimizer = ch.optim.Adamax([inp], lr=0.01)
	iterator = range(iters)
	targ_ = target_rep.view(target_rep.shape[0], -1)
	# retain images
	retained = None
	if retain_images:
		retained = []

	indices_mask = None
	if indices:
		indices_mask = ch.zeros_like(targ_).cuda()
		for i, x in enumerate(indices):
			indices_mask[i][x] = 1

	for i in iterator:

		# Keep track of images for GIF
		if retain_images:
			np_image = inp.data.detach().cpu().numpy()
			np_image = np_image.transpose(0, 2, 3, 1)
			retained.append(np.concatenate(np_image, 0))

		# Get image rep
		rep, _ = model(inp, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		rep_  = rep.view(rep.shape[0], -1)
		# loss = ch.norm(rep_ - targ_, dim=1)

		# Add aux loss if indices provided
		if indices_mask is not None:
			aux_loss = ch.sum(ch.abs((rep_ - targ_) * indices_mask), dim=1)
			loss = aux_loss

		# Back-prop loss
		optimizer.zero_grad()
		loss.backward(ch.ones_like(loss), retain_graph=True)
		optimizer.step()

		# Clamp back to norm-ball
		with ch.no_grad():
			inp.data  = ch.clamp(inp.data, clip_min, clip_max)

	return ch.clamp(inp.clone().detach(), clip_min, clip_max), retained


def find_impostors(model, delta_vec, ds, real,
	iters=200, fake_relu=True, inject=None,
	retain_images=False, indices=None):

	# Shift delta_vec to GPU
	delta_vec = ch.from_numpy(delta_vec).cuda()

	# real_ = (ch.zeros_like(real) + 0.5).cuda()
	real_ = real.clone()

	with ch.no_grad():
		real_rep, _ = model(real, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		real_rep_, _ = model(real_, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		
		# Take note of indices where neuron is activated
		activated_indices = []
		for i, x in enumerate(indices):
			if real_rep_[0][x] > 0:
				activated_indices.append(i)
		delta_vec = delta_vec[activated_indices]
		# Consider the best k-delta values out of these
		delta_vec = delta_vec[:real_.shape[0]]

		target_rep  = real_rep + delta_vec

	# Map indices accordingly
	indices_ = [indices[i] for i in activated_indices[:real_.shape[0]]]
	indices = indices_[:]

	# Print activations at indices; are they really not activated?
	acts = real_rep[0][indices].cpu().numpy()
	# print(ch.sum(real_rep[0] == 0).item(), "/", real_rep[0].shape[0], "neurons not activated")

	impostors, retained = custom_optimization(model, real_, target_rep,
		iters=iters, fake_relu=fake_relu,
		inject=inject, retain_images=retain_images, indices=indices)

	with ch.no_grad():
		labels_    = ch.argmax(model(real_)[0], dim=1)
		pred, _    = model(impostors)
		label_pred = ch.argmax(pred, dim=1)
	
		flatten = (impostors - real.cuda()).view(impostors.shape[0], -1)
		dist_l2   = ch.norm(flatten, p=2, dim=-1)
		dist_linf = ch.max(ch.abs(flatten), dim=-1)[0]

		succeeded = (label_pred != labels_)

	return (impostors, succeeded, dist_l2, dist_linf, retained)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='linf', help='type of model (nat/l2/linf)')
	parser.add_argument('--iters', type=int, default=300, help='number of iterations')
	parser.add_argument('--bs', type=int, default=50, help='batch size while performing attack')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet, robustcifar]')
	parser.add_argument('--inject', type=int, default=None, help='index of layers, to the output of which delta is to be added')
	parser.add_argument('--save_gif', type=bool, default=False, help='save animation of optimization procedure?')
	parser.add_argument('--work_with_train', type=bool, default=False, help='operate on train dataset or test')
	parser.add_argument('--save_path', type=str, default='/p/adversarialml/as9rw/generated_images_notgray/', help='path to save generated images')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	batch_size      = args.bs
	iters           = args.iters
	fake_relu       = ('vgg' not in model_arch)
	inject          = args.inject
	work_with_train = args.work_with_train
	save_path       = args.save_path

	# Load model
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
	elif args.dataset == 'svhn':
		constants = utils.SVHN10()
	# elif args.dataset == 'robustcifar':
	# 	data_path = 
	# 	constants = utils.RobustCIFAR10("")
	else:
		print("Invalid Dataset Specified")
	ds = constants.get_dataset()

	# Load model
	model = constants.get_model(model_type , model_arch)
	# Get stats for neuron activations
	if inject:
		senses_raw  = utils.get_sensitivities("./generic_deltas_%s/%d.txt" %( model_type, inject))
		(mean, std) = utils.get_stats("./generic_stats/%s/%d/" % (model_type, inject))
	else:
		senses_raw  = constants.get_deltas(model_type, model_arch, numpy=True)
		(mean, std) = constants.get_stats(model_type, model_arch)

	# Process and make senses
	if inject:
		mean = mean.flatten()
		std  = std.flatten()
	
	easiest = np.argsort(np.abs((senses_raw - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)), axis=0)

	if work_with_train:
		data_loader, _ = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_train=False, data_aug=False)
	else:
		_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_val=False)

	print(len(data_loader))
	exit(0)

	index_focus = 0
	glob_counter = 1
	# mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	mappinf = [str(x) for x in range(1000)]
	for mm in mappinf:
		os.mkdir(os.path.join(save_path, mm))

	iterator = tqdm(enumerate(data_loader), total=10000 // batch_size)
	for num, (images, labels) in iterator:

		# Per example
		for j in range(images.shape[0]):

			senses  = np.zeros((std.shape[0], std.shape[0]))
			easiest_wanted = easiest[:, index_focus]
			condition = np.logical_and((senses_raw[easiest_wanted, index_focus] != np.inf), (std != 0))
			easiest_wanted = easiest_wanted[condition]

			indices   = []
			i = 0
			for ew in easiest_wanted:
				# Do not deal with INF delta values
				if senses_raw[ew, index_focus] == np.inf:
					continue
				senses[i][ew] = senses_raw[ew, index_focus]
				indices.append(ew)
				i += 1
			# Reshape senses to reflect the number of neurons we are considering
			senses = senses[:i]

			image_template = ch.empty_like(images)
			for k in range(batch_size): image_template[k] = images[j].clone()
			image_template = image_template.cuda()
		
			(impostors, _, _, _, _) = find_impostors(model,senses, ds, image_template, iters=iters,
													fake_relu=fake_relu, inject=inject, indices=indices)
		
			# Impostor labels
			bad_labels = ch.argmax(model(impostors)[0], 1)
			impostors = impostors.cpu().numpy().transpose(0, 2, 3, 1)

			# Save to appropriate folders
			for k, impostor in enumerate(impostors):
				from PIL import Image
				im = Image.fromarray((impostor * 255).astype(np.uint8))
				im.save(os.path.join(save_path, mappinf[bad_labels[k]], str(glob_counter) + ".png"))
				glob_counter += 1

			index_focus += 1
