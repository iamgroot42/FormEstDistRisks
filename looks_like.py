import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
from robustness.tools.misc import log_statement
import numpy as np
import sys
from PIL import Image
import imageio
from tqdm import tqdm
from torch.autograd import Variable

import utils


def custom_optimization(model, inp_og, target_rep, eps, iters=100,
	p='unconstrained', fake_relu=True, inject=None, retain_images=False, indices=None,
	clip_min=0, clip_max=1, lr=0.01, maximize_mode=False, border_erase=0):

	# Define the variable that the optimizer will optimize
	inp =  inp_og.clone().detach().requires_grad_(True)
	# Define optimize
	optimizer = ch.optim.Adamax([inp], lr=lr)
	iterator = tqdm(range(iters))

	if not maximize_mode:
		# Reshape target representation according to how delta values are stored
		targ_ = target_rep.view(target_rep.shape[0], -1)

		# Focus on only one neuron at a time
		indices_mask = None
		if indices:
			indices_mask = ch.zeros_like(targ_).cuda()
			for i, x in enumerate(indices):
				indices_mask[i][x] = 1

	# Track best loss image across all iterations
	best_loss, best_x = float('inf'), None

	# Retain images for creating GIF
	retained = None
	if retain_images: retained = []

	for i in iterator:

		# Add black border if specified:
		if border_erase > 0:
			# print(inp.data.shape)
			# exit(0)
			inp.data[:,:,:border_erase]   = 0
			inp.data[:,:,:,:border_erase:]  = 0
			inp.data[:,:,-border_erase:]  = 0
			inp.data[:,:,:,-border_erase:] = 0

		# Keep track of images for GIF
		if retain_images:
			np_image = inp.data.detach().cpu().numpy()
			np_image = np_image.transpose(0, 2, 3, 1)
			retained.append(np.concatenate(np_image, 0))

		# Get latent representation of image (at current stage of optimization)
		rep, _ = model(inp, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		rep_   = rep.view(rep.shape[0], -1)

		# Simply maximize
		if maximize_mode:
			ch_indices = ch.from_numpy(np.array(indices)).cuda()
			# Construct loss such that the optimizer making loss 0 correspondings to maximizing neuron activations, which is what we want
			loss = 1.1 ** (-rep_.gather(1, ch_indices.view(-1, 1))[:, 0])
		else:
			# Old Loss Term
			loss = ch.norm(rep_ - targ_, dim=1)

			# Add aux loss if indices provided
			if indices_mask is not None:
				aux_loss = ch.sum(ch.abs((rep_ - targ_) * indices_mask), dim=1)
				# loss = loss + aux_loss
				# When seed is not start as same, don't use aux_loss
				loss = aux_loss

		# Zero-out accumulated gradients from previous iteration (useful when training model, not here)
		optimizer.zero_grad()
		# Back-propagate using the defined loss
		loss.backward(ch.ones_like(loss), retain_graph=True)
		# Using the computed gradients, change the image
		optimizer.step()

		# Clamp back to norm-ball, if constraint on perturbation budget
		with ch.no_grad():
			if p == 'inf':
				# Linf
				diff      = ch.clamp(inp.data - inp_og.data, -eps, eps)
				# Add back perturbation, clip
				inp.data  = ch.clamp(inp_og.data + diff, clip_min, clip_max)
			elif p == '2':
				# L2
				diff = inp.data - inp_og.data
				diff = diff.renorm(p=2, dim=0, maxnorm=eps)
				# Add back perturbation, clip
				inp.data  = ch.clamp(inp_og.data + diff, clip_min, clip_max)
			elif p == 'unconstrained':
				inp.data  = ch.clamp(inp.data, clip_min, clip_max)
			else:
				raise ValueError("Projection Currently Not Supported")
		
		# Keep track of loss (and how it changes) by printing it: just to see if things are working, and how well
		iterator.set_description('Loss : %f' % loss.mean())

		# Store best loss and x so far
		if best_loss > loss.mean():
			best_loss = loss.mean()
			best_x    = inp.clone().detach()

	return ch.clamp(best_x, clip_min, clip_max), retained


def neuron_optimization(model, delta_vec, real, labels,
	eps=2.0, iters=200, norm='2', fake_relu=True, inject=None,
	retain_images=False, indices=None, start_with=None, lr=0.01,
	target_class=None, maximize_mode=False, border_erase=0):

	if start_with is None:
		# If some seed image given explicitly to start with, use that
		real_ = real
		real_ = ch.empty_like(real).uniform_(0, 1).cuda()
	else:
		# Otherwise, use the image your're optimizing for as the seed image
		real_ = start_with
	
	with ch.no_grad():
		# Compute latent space representations
		real_rep, _  = model(real,  with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		real_rep_, _ = model(real_, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)

		# Take note of indices where neuron is activated
		activated_indices = []
		for i, x in enumerate(indices):
			# Break out when number of desired neurons (batch size) reached
			if len(activated_indices) == real_.shape[0]: break
			# Look only at neurons which are activated for a given image
			if real_rep_[0][x] > 0: activated_indices.append(i)

		# If none of the neurons are activated, tell the user there's something wrong!
		if len(activated_indices) == 0: raise ValueError("No target neuron identified. Sorry!")

		# If too many activated indices, keep only the ones that batch-size allows
		activated_indices = activated_indices[:real_rep.shape[0]]
		
		# Make sure you're looking at only batch-size many images
		real_rep = real_rep[:len(activated_indices)]

	if maximize_mode: target_rep = None
	else:
		with ch.no_grad():
			# Shift delta_vec to GPU
			delta_vec = ch.from_numpy(delta_vec).cuda()
			delta_vec = delta_vec[activated_indices]
			target_rep  = real_rep + delta_vec

	# Map indices accordingly
	indices_ = [indices[i] for i in activated_indices[:real_.shape[0]]]
	indices  = indices_[:]

	print("\nTargeting these Indices:", indices)
	log_statement("%d/%d neurons activated" % (ch.sum(real_rep_[0] > 0).item(), real_rep_[0].shape[0]))

	# Keep only activated indices
	real_ = real_[:len(activated_indices)]

	# Run optimization to get final image(s)
	impostors, retained = custom_optimization(model, real_, target_rep,
		eps=eps, iters=iters, p=norm, fake_relu=fake_relu,
		inject=inject, retain_images=retain_images, indices=indices, lr=lr,
		maximize_mode=maximize_mode, border_erase=border_erase)

	with ch.no_grad():
		labels_    = ch.argmax(model(real_)[0], dim=1)
		pred, _    = model(impostors)
		label_pred = ch.argmax(pred, dim=1)
	
		flatten = (impostors - real.cuda()).view(impostors.shape[0], -1)
		dist_l2   = ch.norm(flatten, p=2, dim=-1)
		dist_linf = ch.max(ch.abs(flatten), dim=-1)[0]

		# Track images which led to prediction flips after optimzation (attack success rate)
		succeeded = (label_pred != labels_) if target_class is None else (label_pred == target_class)

	return (impostors, succeeded, dist_l2, dist_linf, retained, activated_indices)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch',      type=str,   default='vgg19',         help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type',      type=str,   default='linf',          help='type of model (nat/l2/linf)')
	parser.add_argument('--dataset',         type=str,   default='cifar10',       help='dataset: one of [binarycifar10, cifar10, imagenet, robustcifar]')
	parser.add_argument('--norm',            type=str,   default='unconstrained', help='P-norm to limit budget of adversary')
	parser.add_argument('--load_gray_image', type=str,   default=None,            help='iif path provided, load this image and use as seed instead')
	parser.add_argument('--eps',             type=float, default=0.5,             help='epsilon-iter')
	parser.add_argument('--lr',              type=float, default=0.01,            help='lr for optimizer')
	parser.add_argument('--gray_image',      type=float, default=None,            help='if not None, use image filled with this value ([0, 1]) and use its deltas')
	parser.add_argument('--iters',           type=int,   default=500,             help='number of iterations')
	parser.add_argument('--bs',              type=int,   default=16,              help='batch size while performing attack')
	parser.add_argument('--inject',          type=int,   default=None,            help='index of layers, to the output of which delta is to be added')
	parser.add_argument('--index_focus',     type=int,   default=0,               help='which example to focus on?')
	parser.add_argument('--target_class',    type=int,   default=None,            help='which class to target (default: untargeted)')
	parser.add_argument('--border_erase',    type=int,   default=0,               help='zero out border of image?')
	parser.add_argument('--work_with_train', type=bool,  default=False,           help='operate on train dataset or test')
	parser.add_argument('--save_gif',        type=bool,  default=False,           help='save animation of optimization procedure?')
	parser.add_argument('--maximize_mode',   type=bool,  default=False,           help='maximize neuron activation instead of working with delta values')
	
	args = parser.parse_args()
	utils.flash_utils(args)
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	batch_size      = args.bs
	iters           = args.iters
	eps             = args.eps
	norm            = args.norm
	fake_relu       = ('vgg' not in model_arch)
	inject          = args.inject
	retain_images   = args.save_gif
	work_with_train = args.work_with_train
	index_focus     = args.index_focus
	lr              = args.lr
	target_class    = args.target_class
	maximize_mode   = args.maximize_mode
	is_gray_image   = args.gray_image
	load_gray_image = args.load_gray_image
	border_erase    = args.border_erase

	# Valid range check
	if is_gray_image is not None and (is_gray_image < 0 or is_gray_image > 1):
		raise ValueError("is_gray_image should be in [0,1]")

	# Pick the right dataset
	side_size  = 32  # Size of image
	n_features = 512 # For VGG-19 model
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
		side_size  = 224  # Imagenet images are 224x224
		n_features = 2048 # for Resnet-50 model
	elif args.dataset == 'svhn':
		constants = utils.SVHN10()
	elif args.dataset == 'binary':
		constants = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary_nodog/")
	else:
		print("Invalid Dataset Specified")
	ds = constants.get_dataset()

	# Load model
	model = constants.get_model(model_type , model_arch, parallel=True)

	# Get stats for neuron activations, if optimization mode is based on delta values instead of neuron activation maximization
	if maximize_mode:
		senses  = None
		indices = range(n_features)
	else:
		if inject:
			# inject means we're working with delta values of a layer deep inside the model, load those delt avalues
			senses_raw  = utils.get_sensitivities("./generic_deltas_%s/%d.txt" %( model_type, inject))
			(mean, std) = utils.get_stats("./generic_stats/%s/%d/" % (model_type, inject))
			mean, std   = mean.flatten(), std.flatten()
		else:
			# Pick target class, if given
			if target_class is None:
				senses_raw  = constants.get_deltas(model_type, model_arch, numpy=True)
			else:
				senses_raw  = utils.get_sensitivities("./cifar10_linf_allclass.npy", numpy=True)
				if target_class == -1:
					best_classes = np.argmin(senses_raw, 2)
					senses_picked = np.zeros((senses_raw.shape[0], senses_raw.shape[1]))
					for i in range(senses_raw.shape[0]):
						for j in range(senses_raw.shape[1]):
							senses_picked[i][j] = senses_raw[i][j][best_classes[i][j]]
					senses_picked = senses_picked.T
				else:
					senses_raw = senses_raw[:,:,target_class].T

			(mean, std) = constants.get_stats(model_type, model_arch)

			# prefix = "./npy_files/binary_deltas_linf"
			# (mean, std) = utils.get_stats(prefix + "/")
			# senses_raw  = utils.get_sensitivities(prefix + ".npy", numpy=True)

		if is_gray_image is not None:
			# Extract model weights for the given gray image, work with those delta values
			# Basically delta value computation on the fly
			weights     = utils.get_these_params(model, utils.get_logits_layer_name(model_arch))
			# Get delta values corresponding to gray image
			with ch.no_grad():
				gray_image    = ch.zeros((1, 3, side_size, side_size)).cuda() + is_gray_image
				logits, _     = model(gray_image)
				compute_delta_values = utils.compute_delta_values(logits[0], weights)
				compute_delta_values, _ = compute_delta_values.min(0)
				compute_delta_values = compute_delta_values.cpu().numpy()
				# See what happens
				compute_delta_values = (compute_delta_values * 0) + 1e1
				for i in range(senses_raw.shape[1]):
					senses_raw[:, i] = compute_delta_values

		# Use observed mean and std of neuron outputs to see which delta values are seemingly easy (scaled by mean snd std)
		easiest = np.argsort(np.abs((senses_raw - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)), axis=0)
		senses  = np.zeros((std.shape[0], std.shape[0]))
	
		log_statement("==> Example in focus : %d" % index_focus)
		easiest_wanted = easiest[:, index_focus]
		condition = np.logical_and((senses_raw[easiest_wanted, index_focus] != np.inf), (std != 0))
		easiest_wanted = easiest_wanted[condition]

		indices   = []
		i = 0
		for ew in easiest_wanted:
			# Do not deal with INF delta values (since they correspond to cases where class flip is not possible)
			if senses_raw[ew, index_focus] == np.inf: continue
			senses[i][ew] = senses_raw[ew, index_focus]
			indices.append(ew)
			i += 1
		# Reshape senses to reflect the number of neurons we are considering
		senses = senses[:i]

		# Use loaded image as gray image
		if load_gray_image is None :
			gray_image = ch.zeros((1, 3, side_size, side_size)).cuda() + is_gray_image
		else:
			# Use user-specified template image as seed instead of gray image, if path given
			log_statement("Using template image")
			loaded_image = np.asarray(Image.open(load_gray_image)).astype('float32') / 255
			loaded_image = np.expand_dims(np.transpose(loaded_image, (2, 0, 1)), 0)
			gray_image = ch.from_numpy(loaded_image).cuda()
	
	if work_with_train:
		# Work with images from the train dataset (for debugging, not the real usecase) if specified
		data_loader, _ = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_train=False, data_aug=False)
	else:
		# train_loader, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_val=False)
		_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_val=False, only_val=True)
		# pmeans, pstds = utils.classwise_pixelwise_stats(train_loader, classwise=True)

	index_base, asr = 0, 0
	l2_norms        = [np.inf, -1.0, 0]
	linf_norms      = [np.inf, -1.0, 0]
	norm_count      = 0
	
	iterator = enumerate(data_loader)
	for num, (image, label) in iterator:
		image, label = image.cuda(), label.cuda()

		start_with = None
		start_with = image.clone()
		for j in range(image.shape[0]):
			if is_gray_image is None:
				start_with[j] = image[index_focus]
			else:
				start_with[j] = gray_image[0]
				# start_with[j] = pmeans[j % 10]

		for j in range(image.shape[0]):
			image[j] = image[index_focus]
			if is_gray_image is None:
				label[j] = label[index_focus]
			else:
				image[j] = gray_image[0]
				# image[j] = pmeans[j % 10]

		# Main algorithm
		(impostors, succeeded, dist_l2, dist_linf, retained, activated_indices) = neuron_optimization(model,
															senses, image, label, eps=eps, iters=iters,
															norm=norm, fake_relu=fake_relu, inject=inject,
															retain_images=retain_images, indices=indices,
															start_with=start_with, lr=lr, target_class=target_class,
															maximize_mode=maximize_mode, border_erase=border_erase)
		asr        += ch.sum(succeeded).float()
		index_base += len(image)

		# Keep track of distance statistics
		l2_norms[0],   l2_norms[1]   = min(l2_norms[0],   ch.min(dist_l2).item()),   max(l2_norms[1],   ch.max(dist_l2).item())
		linf_norms[0], linf_norms[1] = min(linf_norms[0], ch.min(dist_linf).item()), max(linf_norms[1], ch.max(dist_linf).item())

		l2_norms[2]    += ch.sum(dist_l2)
		linf_norms[2]  += ch.sum(dist_linf)
		norm_count     += dist_l2.shape[0]
		dist_string    = "L2 norm: [%.2f, %.2f, %.2f], Linf norm: [%.1f/255, %.1f/255, %.1f/255]"  % (
			l2_norms[0], l2_norms[1], l2_norms[2] / norm_count,
			255 * linf_norms[0], 255 * linf_norms[1], 255 * linf_norms[2] / norm_count)

		print('Success Rate : %.2f | %s' % (100 * (asr/index_base),  dist_string))

		# To save as PNG (for editing in paint, or just looking at them specifically)
		# Out of all images in batch, pick one to look at in detail if you want
		want_this = 1
		image_in_sight = retained[-1][(want_this-1) * side_size:want_this * side_size]
		np.save("./visualize/closest_to_this", image_in_sight)
		log_statement("==> Saved as array")

		# For most case, ignore this image
		im = Image.fromarray((image_in_sight * 255).astype(np.uint8))
		im.save("./visualize/paint_this.png")
		log_statement("==> Saved as PNG")

		# Save GIF
		if retain_images:
			log_statement("==> Generating GIF")
			
			image = image[:len(activated_indices)]
			basic_image = np.concatenate(image.cpu().numpy().transpose(0, 2, 3, 1), 0)
			retained = [np.concatenate([x, basic_image], 1) for x in retained]

			# Scale to [0, 255] with uint8
			retained = [(255 * x).astype(np.uint8) for x in retained]
			imageio.mimsave('./visualize/basic_deltas.gif', retained)
			log_statement("==> Generated GIF")


		if not maximize_mode:
			with ch.no_grad():
				# Get latent reps of perturbed images
				latent, _      = model(image,    with_latent=True,  just_latent=True, this_layer_output=inject)
				latent_pert, _ = model(impostors, with_latent=True, just_latent=True, this_layer_output=inject)

			if inject:
				delta_actual = (latent_pert - latent).cpu().view(latent.shape[0], -1).numpy().T
			else:
				delta_actual = (latent_pert - latent).cpu().numpy().T

			achieved = delta_actual >= senses_raw[:, index_base: index_base + len(image)]
			print(np.sum(achieved, 0))

		exit(0)
