import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import utils


def custom_optimization(model, inp_og, target_rep, eps, random_restart_targets, iters=100,
	p='2', custom_best=False, fake_relu=True, random_restarts=0, inject=None):
	inp =  inp_og.clone().detach().requires_grad_(True)
	# optimizer = ch.optim.AdamW([inp], lr=0.001)
	optimizer = ch.optim.Adamax([inp], lr=0.1)
	# optimizer = ch.optim.Adamax([inp], lr=0.001)
	# scheduler = ch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1)
	iterator = range(iters)
	# iterator = tqdm(iterator)
	targ_ = target_rep.view(target_rep.shape[0], -1)
	# use_best behavior
	best_loss, best_x = float('inf'), None
	for i in iterator:
		# Get image rep
		(output, rep), _ = model(inp, with_latent=True, fake_relu=fake_relu, this_layer_output=inject)
		rep_  = rep.view(rep.shape[0], -1)
		# Get loss
		loss  = ch.norm(rep_ - targ_, dim=1)
		# this_loss = loss.sum().item()
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
				inp.data  = ch.clamp(inp_og.data + diff, 0, 1)
			elif p == '2':
				# L2
				diff = inp.data - inp_og.data
				diff = diff.renorm(p=2, dim=0, maxnorm=eps)
				# Add back perturbation, clip
				inp.data  = ch.clamp(inp_og.data + diff, 0, 1)
			elif p == 'unconstrained':
				inp.data  = ch.clamp(inp.data, 0, 1)
			else:
				raise ValueError("Projection Currently Not Supported")
		
		# Track CE loss to pick best adv example
		with ch.no_grad():
			this_loss = custom_best(None, inp).mean().item()

		# iterator.set_description('Loss : %f' % loss.mean())
		# iterator.set_description('Loss : %f' % this_loss)

		# Store best loss and x so far
		if best_loss > this_loss:
			best_loss = this_loss
			best_x    = inp.clone().detach()

	# print(best_loss, this_loss)
	return ch.clamp(best_x, 0, 1)


def madry_optimization(model, inp_og, target_rep, eps, random_restart_targets, iters=100,
	p='2', custom_best=False, fake_relu=True, random_restarts=0, inject=None):
	def custom_inversion_loss(m, inp, targ):
		output, rep = m(inp, with_latent=True, fake_relu=fake_relu, this_layer_output=inject)
		# Normalized L2 error w.r.t. the target representation
		rep_  = rep.view(rep.shape[0], -1)
		targ_ = targ.view(rep.shape[0], -1)
		loss  = ch.sum(ch.pow(rep_ - targ_, 2), dim=1) # Do not normalize
		# loss  = ch.div(loss, ch.norm(targ_, dim=1)) # Normalize
		# print(loss.mean())
		return loss, output

	kwargs = {
		'custom_loss': custom_inversion_loss,
		'constraint': p,
		'eps': eps,
		'step_size': 0.01, 
		# 'step_size': 2.5 * eps / iters,
		'iterations': iters,
		'targeted': True,
		'do_tqdm': True,
		'custom_best': custom_best,
		'random_restarts': random_restarts,
		'random_restart_targets': random_restart_targets
	}
	_, im_matched = model(inp_og, target_rep, make_adv=True, **kwargs)
	return im_matched


def find_impostors(model, delta_vec, ds, real, labels,
	eps=2.0, iters=200, norm='2', fake_relu=True,
	random_restarts=0, inject=None, goal=False):

	# Shift delta_vec to GPU
	delta_vec = ch.from_numpy(delta_vec).cuda()

	if goal:
		labels_ = labels
	else:
		labels_ = ch.argmax(model(real)[0], 1)

	impostors = parallel_impostor(model, delta_vec, real, labels_,
		eps, iters, norm, fake_relu, random_restarts, inject)

	with ch.no_grad():
		pred, _    = model(impostors)
		label_pred = ch.argmax(pred, dim=1)
	
		flatten = (impostors - real.cuda()).view(impostors.shape[0], -1)
		dist_l2   = ch.norm(flatten, p=2, dim=-1)
		dist_linf = ch.max(ch.abs(flatten), dim=-1)[0]

		succeeded = (label_pred != labels_)

	return (impostors, succeeded, dist_l2, dist_linf)


def parallel_impostor(model, delta_vec, im, true_labels, eps, iters, norm, fake_relu, random_restarts, inject):
	# Override custom_best, use cross-entropy on model instead
	criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()
	def ce_loss(loss, x):
		output, _ = model(x, fake_relu=fake_relu)
		# We want CE loss b/w new and old to be as high as possible
		return -criterion(output, true_labels)

	# Use Madry's optimization
	# im_matched = madry_optimization(model, im, delta_vec,
	im_matched = custom_optimization(model, im, delta_vec,
		random_restart_targets=true_labels, eps=eps, iters=iters,
		p=norm, custom_best=ce_loss, fake_relu=fake_relu,
		random_restarts=random_restarts, inject=inject)
	
	return im_matched


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='nat', help='type of model (nat/l2/linf)')
	parser.add_argument('--eps', type=float, default=0.5, help='epsilon-iter')
	parser.add_argument('--iters', type=int, default=250, help='number of iterations')
	parser.add_argument('--bs', type=int, default=1024, help='batch size while performing attack')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet]')
	parser.add_argument('--norm', type=str, default='2', help='P-norm to limit budget of adversary')	
	parser.add_argument('--random_restarts', type=int, default=0, help='how many random restarts? (0 -> False)')
	parser.add_argument('--inject', type=int, default=None, help='index of layers, to the output of which delta is to be added')
	parser.add_argument('--goal', type=bool, default=False, help='is goal to maximize misclassification (True) or flip model predictions (False, default)')
	
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
	random_restarts = args.random_restarts
	inject          = args.inject
	goal            = args.goal

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
	# senses = np.load("./cw_deltas_%s/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_%s/%d.npy" % (model_type, inject))
	# senses = np.load("./pgd_deltas_%s_ut/%d.npy" % (model_type, inject))
	senses = np.load("./pgd_deltas_try_nat/%d.npy" % (inject))
	(mean, std) = utils.get_stats("./generic_stats/%s/%d/" % (model_type, inject))
	print("Loaded delta vectors")

	# Use totally random values
	scale=1e2
	senses_random = []
	for i in tqdm(range(senses.shape[0])):
		senses_random.append(np.random.normal(mean, scale*std))
	senses = np.array(senses_random)
	print("Replaced with random delta vectors")
	
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	index_base, asr = 0, 0
	l2_norms   = [np.inf, -1.0, 0]
	linf_norms = [np.inf, -1.0, 0]
	norm_count = 0
	
	iterator = tqdm(test_loader)
	for (image, label) in iterator:
		image, label = image.cuda(), label.cuda()
		(impostors, succeeded, dist_l2, dist_linf) = find_impostors(model,
															senses[index_base: index_base + len(image)], ds,
															image, label, eps=eps, iters=iters,
															norm=norm, fake_relu=fake_relu,
															random_restarts=random_restarts,
															inject=inject, goal=goal)
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
		if goal:
			name_it = "Misclassification Rate"
		else:
			name_it = "Success Rate"
		iterator.set_description('%s : %.2f | %s' % (name_it, 100 * (	asr/index_base),  dist_string))

		# Impostor labels
		# bad_labels = ch.argmax(model(impostors)[0], 1)

		# image_labels = [label.cpu().numpy(), bad_labels.cpu().numpy()]
		# image_labels[0] = [ds.class_names[i] for i in image_labels[0]]
		# image_labels[1] = [ds.class_names[i] for i in image_labels[1]]

		# # Statistics for cases where model misclassified
		# l2_norms    = ch.sum(dist_l2[succeeded])
		# linf_norms  = ch.sum(dist_linf[succeeded])
		# print("Succeeded images: L2 norm: %.3f, Linf norm: %.2f/255"  % (l2_norms / norm_count, 255 * linf_norms / norm_count))

		# show_image_row([image.cpu(), impostors.cpu()],
		# 			["Real Images", "Attack Images"],
		# 			tlist=image_labels,
		# 			fontsize=22,
		# 			filename="./visualize/cw_try.png")
