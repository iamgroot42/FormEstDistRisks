import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
from robustness.tools.misc import log_statement
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def custom_optimization(model, inp_og, target_rep, eps, iters=100,
	p='unconstrained', fake_relu=True, inject=None, retain_images=False, indices=None,
	clip_min=0, clip_max=1):
	# clip_min=0, clip_max=0.75):
	inp =  inp_og.clone().detach().requires_grad_(True)
	# optimizer = ch.optim.AdamW([inp], lr=1)
	optimizer = ch.optim.Adamax([inp], lr=0.1)
	# optimizer = ch.optim.Adamax([inp], lr=0.05)
	# optimizer = ch.optim.Adamax([inp], lr=0.01)
	# optimizer = ch.optim.Adamax([inp], lr=0.001)
	# scheduler = ch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)
	iterator = range(iters)
	iterator = tqdm(iterator)
	targ_ = target_rep.view(target_rep.shape[0], -1)
	# use_best behavior
	best_loss, best_x = float('inf'), None
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
		# (output, rep), _ = model(inp, with_latent=True, fake_relu=fake_relu, this_layer_output=inject)
		rep_  = rep.view(rep.shape[0], -1)
		# Get loss
		# loss  = ch.norm(rep_ - targ_, dim=1)
		# this_loss = loss.sum().item()

		# Old Loss Term
		# loss = ch.div(ch.norm(rep_ - targ_, dim=1), ch.norm(targ_, dim=1))
		# loss = ch.norm(rep_ - targ_, dim=1)

		# Add aux loss if indices provided
		if indices_mask is not None:
			aux_loss = ch.sum(ch.abs((rep_ - targ_) * indices_mask), dim=1)
			# aux_loss = ch.div(aux_loss, ch.norm(targ_ * indices_mask, dim=1))
			# loss = loss + aux_loss
			# When seed is not start as same, don't use aux_loss
			loss = aux_loss

		# Back-prop loss
		optimizer.zero_grad()
		loss.backward(ch.ones_like(loss), retain_graph=True)
		optimizer.step()
		# scheduler.step()
		# print(optimizer.param_groups[0]['lr'])

		# Clamp back to norm-ball
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
		
		iterator.set_description('Loss : %f' % loss.mean())

		# Store best loss and x so far
		if best_loss > loss.mean():
			best_loss = loss.mean()
			best_x    = inp.clone().detach()

	return ch.clamp(best_x, clip_min, clip_max), retained


def find_impostors(model, delta_vec, ds, real, labels,
	eps=2.0, iters=200, norm='2', fake_relu=True, inject=None,
	retain_images=False, indices=None, start_with=None):

	# Shift delta_vec to GPU
	delta_vec = ch.from_numpy(delta_vec).cuda()

	if start_with is None:
		real_ = real
		real_ = ch.empty_like(real).uniform_(0, 1).cuda()
	else:
		# real_ = ch.zeros_like(real).cuda() + 0.4
		# real_ = ch.empty_like(real).uniform_(0, 1).cuda()
		real_ = start_with
		# Add some random noise to alleviate 'no gradient' problems
		# noise = ch.empty_like(real).uniform_(-64/255, 64/255).cuda()
		# real_ = ch.clamp(real_ + noise, 0, 1)

	with ch.no_grad():
		real_rep, _ = model(real, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		real_rep_, _ = model(real_, with_latent=True, fake_relu=fake_relu, just_latent=True, this_layer_output=inject)
		
		# Take note of indices where neuron is activated
		activated_indices = []
		for i, x in enumerate(indices):
			# Break out when done
			if len(activated_indices) == real_.shape[0]:
				break
			# if real_rep_[0][x] > 0:
			if real_rep_[len(activated_indices)][x] > 0:
				activated_indices.append(i)
			# activated_indices.append(i)
		delta_vec = delta_vec[activated_indices]
		# Consider the best k-delta values out of these
		delta_vec = delta_vec[:real_.shape[0]]

		target_rep  = real_rep + delta_vec
		# target_rep  = real_rep

	# Map indices accordingly
	indices_ = [indices[i] for i in activated_indices[:real_.shape[0]]]
	indices = indices_[:]

	# Print activations at indices; are they really not activated?
	acts = real_rep[0][indices].cpu().numpy()
	# for i, act in enumerate(acts):
	# 	print(i, ":", act)
	print(ch.sum(real_rep_[0] == 0).item(), "/", real_rep_[0].shape[0], "neurons not activated")

	impostors, retained = custom_optimization(model, real_, target_rep,
		eps=eps, iters=iters, p=norm, fake_relu=fake_relu,
		inject=inject, retain_images=retain_images, indices=indices)

	with ch.no_grad():
		labels_    = ch.argmax(model(real_)[0], dim=1)
		pred, _    = model(impostors)
		label_pred = ch.argmax(pred, dim=1)
	
		flatten = (impostors - real.cuda()).view(impostors.shape[0], -1)
		dist_l2   = ch.norm(flatten, p=2, dim=-1)
		dist_linf = ch.max(ch.abs(flatten), dim=-1)[0]

		succeeded = (label_pred != labels_)
		mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
		# for ii in label_pred:
			# print(mappinf[ii])

	return (impostors, succeeded, dist_l2, dist_linf, retained)


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='linf', help='type of model (nat/l2/linf)')
	parser.add_argument('--eps', type=float, default=0.5, help='epsilon-iter')
	parser.add_argument('--iters', type=int, default=1000, help='number of iterations')
	parser.add_argument('--bs', type=int, default=16, help='batch size while performing attack')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet, robustcifar]')
	parser.add_argument('--norm', type=str, default='unconstrained', help='P-norm to limit budget of adversary')
	parser.add_argument('--inject', type=int, default=None, help='index of layers, to the output of which delta is to be added')
	parser.add_argument('--save_gif', type=bool, default=False, help='save animation of optimization procedure?')
	parser.add_argument('--work_with_train', type=bool, default=False, help='operate on train dataset or test')
	parser.add_argument('--index_focus', type=int, default=0, help='which example to focus on?')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
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
		# senses_raw  = utils.get_sensitivities("./deltas_train_cifar10_linf.txt")
	# senses = np.load("./cw_deltas_%s/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_%s/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_%s_ut/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_try_nat/%d.npy" % (inject))

	# Process and make senses
	if inject:
		mean = mean.flatten()
		std  = std.flatten()

	# Mess up with delta values BEFORE picking indices
	# scale=5e2
	# scale=1e1
	# senses_raw = np.random.normal(mean, scale*std, size=(senses_raw.shape[1], senses_raw.shape[0])).T
	# print("Totally random delta values")

	# easiest = np.argsort(np.abs(senses_raw) / (np.expand_dims(std, 1) + 1e-10), axis=0)
	# easiest = np.argsort(np.abs(senses_raw) / (np.expand_dims(std, 1)), axis=0)
	easiest = np.argsort(np.abs((senses_raw - np.expand_dims(mean, 1)) / np.expand_dims(std, 1)), axis=0)
	# easiest = np.argsort(np.abs(senses_raw), axis=0)
	# easiest = np.argsort(np.abs(senses_raw), axis=0)
	senses  = np.zeros((std.shape[0], std.shape[0]))
	
	# for i in range(easiest.shape[1]):
	# 	senses.append(senses_raw[easiest[0, i], i])
	# senses = np.array(senses)

	# index_focus  = 10#6#2
	log_statement("==> Example in focus : %d" % index_focus)
	easiest_wanted = easiest[:, index_focus]
	condition = np.logical_and((senses_raw[easiest_wanted, index_focus] != np.inf), (std != 0))
	easiest_wanted = easiest_wanted[condition]

	# Reverse
	# easiest_wanted = easiest_wanted[::-1]
	# Random
	# np.random.shuffle(easiest_wanted)

	# Totally random values
	# scale=1e2
	# # scale=1e1
	# senses_raw = np.random.normal(mean, scale*std, size=(senses_raw.shape[1], senses_raw.shape[0])).T
	# print("Totally random delta values")

	# valid_senses = senses_raw[easiest[:, index_focus], index_focus]
	# valid_senses = valid_senses[valid_senses != np.inf]

	# value_try = list(range(-batch_size // 2, batch_size // 2 + 1))
	# value_try = list(range(1, batch_size + 1))
	# indices   = None
	indices   = []
	
	# easiest_wanted = easiest_wanted[200:]
	# Override 
	# easiest_wanted = list(range(easiest_wanted.shape[0]))

	i = 0
	for ew in easiest_wanted:
		# DO not deal with INF delta values
		if senses_raw[ew, index_focus] == np.inf:
			continue
		senses[i][ew] = senses_raw[ew, index_focus]
		indices.append(ew)
		# senses[i] = value_try[i]
		# print(senses[i][ew])
		i += 1
		# if i == batch_size:
		# 	break
	# Reshape senses to reflect the number of neurons we are considering
	senses = senses[:i]

	# print(indices[:batch_size])

	# while i < batch_size:
	# 	if senses_raw[easiest_wanted[i], index_focus] == np.inf:
	# 		continue
	# 	senses[i] = senses_raw[easiest_wanted[i], index_focus]
	# 	print(senses_raw[easiest_wanted[i], index_focus])
	# 	print(senses[i])
	# 	exit(0)
	# 	i += 1
	
	if work_with_train:
		data_loader, _ = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_train=False, data_aug=False)
	else:
		# train_loader, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, shuffle_val=False)
		# _, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
		_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, shuffle_val=True)
		# pmean, pstd = utils.classwise_pixelwise_stats(train_loader)
		# pmeans, pstds = utils.classwise_pixelwise_stats(train_loader, classwise=True)

	index_base, asr = 0, 0
	l2_norms   = [np.inf, -1.0, 0]
	linf_norms = [np.inf, -1.0, 0]
	norm_count = 0

	i_want_this = 0
	
	iterator = tqdm(enumerate(data_loader))
	for num, (image, label) in iterator:
		if num < i_want_this:
			continue
		image, label = image.cuda(), label.cuda()

		start_with = None
		start_with = image.clone()
		for j in range(image.shape[0]):
			# distr = ch.distributions.normal.Normal(loc=pmean, scale=pstd)
			# start_with[j] = distr.sample()
			start_with[j] = image[index_focus]
			# start_with[j][0] = np.random.uniform(0.2, 0.8)
			# start_with[j][1] = np.random.uniform(0.2, 0.8)
			# start_with[j][2] = np.random.uniform(0.2, 0.8)
			# start_with[j] = pmean
			# start_with[j] = pmeans[j % 10]
			# start_with[j] = pmeans[7]
			# start_with[j] = image[j]
		# start_with = None

		for j in range(image.shape[0]):
			image[j] = image[index_focus]
			label[j] = label[index_focus]

		(impostors, succeeded, dist_l2, dist_linf, retained) = find_impostors(model,
															# senses[index_base: index_base + len(image)], ds,
															senses, ds,
															image, label, eps=eps, iters=iters,
															norm=norm, fake_relu=fake_relu, inject=inject,
															retain_images=retain_images, indices=indices,
															start_with=start_with)
		asr        += ch.sum(succeeded).float()
		index_base += len(image)

		# Keep track of distance statistics
		# print(ch.min(dist_linf).item(), ch.max(dist_linf).item())
		l2_norms[0], l2_norms[1]     = min(l2_norms[0], ch.min(dist_l2).item()), max(l2_norms[1], ch.max(dist_l2).item())
		linf_norms[0], linf_norms[1] = min(linf_norms[0], ch.min(dist_linf).item()), max(linf_norms[1], ch.max(dist_linf).item())

		l2_norms[2]    += ch.sum(dist_l2)
		linf_norms[2]  += ch.sum(dist_linf)
		norm_count += dist_l2.shape[0]
		dist_string = "L2 norm: [%.2f, %.2f, %.2f], Linf norm: [%.1f/255, %.1f/255, %.1f/255]"  % (
			l2_norms[0], l2_norms[1], l2_norms[2] / norm_count,
			255 * linf_norms[0], 255 * linf_norms[1], 255 * linf_norms[2] / norm_count)
		# Keep track of attack success rate
		name_it = "Success Rate"
		iterator.set_description('%s : %.2f | %s' % (name_it, 100 * (	asr/index_base),  dist_string))

		# Impostor labels
		bad_labels = ch.argmax(model(impostors)[0], 1)

		image_labels = [label.cpu().numpy(), bad_labels.cpu().numpy()]
		# image_labels[0] = [ds.class_names[i] for i in image_labels[0]]
		# image_labels[1] = [ds.class_names[i] for i in image_labels[1]]

		# Statistics for cases where model misclassified
		l2_norms    = ch.sum(dist_l2[succeeded])
		linf_norms  = ch.sum(dist_linf[succeeded])
		print("\nSucceeded images: L2 norm: %.3f, Linf norm: %.2f/255"  % (l2_norms / norm_count, 255 * linf_norms / norm_count))

		show_image_row([image.cpu(), impostors.cpu()],
					["Real Images", "Attack Images"],
					tlist=image_labels,
					fontsize=22,
					filename="./visualize/basic_deltas.png")

		want_this = 16
		image_in_sight = retained[-1][(want_this-1) * 32:want_this * 32]
		np.save("./visualize/closest_to_this", image_in_sight)
		log_statement("==> Saved as array")

		# Save as PNG (for editing in paint)
		from PIL import Image
		im = Image.fromarray((image_in_sight * 255).astype(np.uint8))
		im.save("./visualize/paint_this.png")
		log_statement("==> Saved as PNG")

		# exit(0)

		# Save GIF
		if retain_images:
			log_statement("==> Generating GIF")
			import imageio
			basic_image = np.concatenate(image.cpu().numpy().transpose(0, 2, 3, 1), 0)
			retained = [np.concatenate([x, basic_image], 1) for x in retained]

			# Add blurring
			# import cv2
			# # Do for each image
			# for k, x in enumerate(retained):
				# for i in range(0, x.shape[0], 32):
					# for j in range(0, x.shape[1], 32):
			# 			retained[k][i:i+32, j:j+32] = cv2.GaussianBlur(x[i:i+32, j:j+32], (3,3), 0) #Gaussian Blur
						# retained[k][i:i+32, j:j+32] = cv2.medianBlur(x[i:i+32, j:j+32], 3) # Median Blur
			# 			retained[k][i:i+32, j:j+32] = cv2.blur(x[i:i+32, j:j+32], (3, 3)) # Mean Blur


			# Scale to [0, 255] with uint8
			retained = [(255 * x).astype(np.uint8) for x in retained]
			imageio.mimsave('./visualize/basic_deltas.gif', retained)

		with ch.no_grad():
			# Get latent reps of perturbed images
			latent, _      = model(image, with_latent=True, just_latent=True, this_layer_output=inject)
			latent_pert, _ = model(impostors, with_latent=True, just_latent=True, this_layer_output=inject)

		if inject:
			delta_actual = (latent_pert - latent).cpu().view(latent.shape[0], -1).numpy().T
		else:
			delta_actual = (latent_pert - latent).cpu().numpy().T

		print(np.linalg.norm(delta_actual, 2, 0))

		achieved = delta_actual >= senses_raw[:, index_base: index_base + len(image)]
		print(np.sum(achieved, 0))

		satisfied = []
		for ii, kk in enumerate(easiest[0, index_base: index_base + len(image)]):
			satisfied.append(1 * (delta_actual[kk, ii] >= senses_raw[kk, ii]))

		print(satisfied)

		exit(0)
