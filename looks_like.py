import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def custom_optimization(model, inp_og, target_rep, eps, iters=100,
	p='2', fake_relu=True, inject=None, retain_images=False):
	inp =  inp_og.clone().detach().requires_grad_(True)
	# optimizer = ch.optim.AdamW([inp], lr=0.001)
	# optimizer = ch.optim.Adamax([inp], lr=0.1)
	optimizer = ch.optim.Adamax([inp], lr=0.1)
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
		loss  = ch.norm(rep_ - targ_, dim=1)
		# this_loss = loss.sum().item()

		# Old Loss Term
		# loss = ch.div(ch.norm(rep_ - targ_, dim=1), ch.norm(targ_, dim=1))

		# Back-prop loss
		optimizer.zero_grad()
		loss.backward(ch.ones_like(loss), retain_graph=True)
		optimizer.step()
		# scheduler.step()
		# print(optimizer.param_groups[0]['lr'])

		# Clamp back to norm-ball
		with ch.no_grad():
			inp.data  = ch.clamp(inp.data, 0, 1)
		
		iterator.set_description('Loss : %f' % loss.mean())

		# Store best loss and x so far
		if best_loss > loss.mean():
			best_loss = loss.mean()
			best_x    = inp.clone().detach()

	# print(best_loss, this_loss)

	return ch.clamp(best_x, 0, 1), retained


def find_impostors(model, delta_vec, ds, real, labels,
	eps=2.0, iters=200, norm='2', fake_relu=True, inject=None,
	retain_images=False):

	# Shift delta_vec to GPU
	delta_vec = ch.from_numpy(delta_vec).cuda()

	impostors, retained = custom_optimization(model, real, delta_vec,
		eps=eps, iters=iters, p=norm, fake_relu=fake_relu,
		inject=inject, retain_images=retain_images)

	with ch.no_grad():
		labels_    = ch.argmax(model(real)[0])
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
	parser.add_argument('--model_type', type=str, default='nat', help='type of model (nat/l2/linf)')
	parser.add_argument('--eps', type=float, default=0.5, help='epsilon-iter')
	parser.add_argument('--iters', type=int, default=250, help='number of iterations')
	parser.add_argument('--bs', type=int, default=32, help='batch size while performing attack')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet]')
	parser.add_argument('--norm', type=str, default='2', help='P-norm to limit budget of adversary')	
	parser.add_argument('--inject', type=int, default=None, help='index of layers, to the output of which delta is to be added')
	parser.add_argument('--save_gif', type=bool, default=False, help='save animation of optimization procedure?')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	batch_size      = args.bs
	iters           = args.iters
	eps             = args.eps
	norm            = args.norm
	fake_relu       = (model_arch != 'vgg19')
	inject          = args.inject
	retain_images   = args.save_gif

	# Load model
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
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
		senses_raw  = constants.get_deltas(model_type, model_arch)
		(mean, std) = constants.get_stats(model_type, model_arch)
	# senses = np.load("./cw_deltas_%s/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_%s/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_%s_ut/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_try_nat/%d.npy" % (inject))
	print("Loaded delta vectors")

	# Process and make senses
	if inject:
		mean = mean.flatten()
		std  = std.flatten()

	easiest = np.argsort(np.abs(senses_raw) / (np.expand_dims(std, 1) + 1e-10), axis=0)
	senses = np.zeros((batch_size, std.shape[0]))
	
	# for i in range(easiest.shape[1]):
	# 	senses.append(senses_raw[easiest[0, i], i])
	# senses = np.array(senses)

	index_focus  = 6#2#6
	easiest_wanted = easiest[:, index_focus]
	condition = np.logical_and((senses_raw[easiest_wanted, index_focus] != np.inf), (std != 0))
	easiest_wanted = easiest_wanted[condition]

	# Reverse
	# easiest_wanted = easiest_wanted[::-1]
	# Random
	# np.random.shuffle(easiest_wanted)

	# Totally random values
	scale=1e2
	senses_raw = np.random.normal(mean, scale*std, size=(senses_raw.shape[1], senses_raw.shape[0])).T

	# valid_senses = senses_raw[easiest[:, index_focus], index_focus]
	# valid_senses = valid_senses[valid_senses != np.inf]
	
	i = 0
	while i < batch_size:
		if senses_raw[easiest_wanted[i], index_focus] == np.inf:
			continue
		senses[i] = senses_raw[easiest_wanted[i], index_focus]
		i += 1
	
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	index_base, asr = 0, 0
	l2_norms   = [np.inf, -1.0, 0]
	linf_norms = [np.inf, -1.0, 0]
	norm_count = 0

	i_want_this = 0
	
	iterator = tqdm(enumerate(test_loader))
	for num, (image, label) in iterator:
		if num < i_want_this:
			continue
		image, label = image.cuda(), label.cuda()

		for j in range(image.shape[0]):
			image[j] = image[index_focus]
			label[j] = label[index_focus]

		(impostors, succeeded, dist_l2, dist_linf, retained) = find_impostors(model,
															senses[index_base: index_base + len(image)], ds,
															image, label, eps=eps, iters=iters,
															norm=norm, fake_relu=fake_relu, inject=inject,
															retain_images=retain_images)
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
					# filename="./visualize/cw_try.png")

		# Save GIF
		if retain_images:
			print("==> Generating GIF")
			import imageio
			basic_image = np.concatenate(image.cpu().numpy().transpose(0, 2, 3, 1), 0)
			retained = [np.concatenate([x, basic_image], 1) for x in retained]
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

		achieved = delta_actual >= senses_raw[:, index_base: index_base + len(image)]
		print(np.sum(achieved, 0))

		satisfied = []
		for ii, kk in enumerate(easiest[0, index_base: index_base + len(image)]):
			satisfied.append(1 * (delta_actual[kk, ii] >= senses_raw[kk, ii]))

		print(satisfied)

		exit(0)
