import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import optimize, utils


def find_impostors(model, delta_values, ds, images, labels, mean, std,
	optim_type='custom', verbose=True, n=4, eps=2.0, iters=200,
	binary=True, norm='2', save_attack=False, custom_best=False,
	fake_relu=True, analysis_start=0, random_restarts=0, 
	delta_analysis=False, corr_analysis=False, dist_stats=False,
	goal=False, active_only=False):
	image_, labels_ = [], []
	# Get target images
	for i, image in enumerate(images):
		targ_img = image.unsqueeze(0)
		real = targ_img.repeat(n, 1, 1, 1)
		image_.append(real)
		for j in range(n):
			labels_.append(labels[i])
	real = ch.cat(image_, 0)

	# Get scaled senses
	scaled_delta_values = utils.scaled_values(delta_values, mean, std, eps=0)
	# Replace inf values with largest non-inf values
	delta_values[delta_values == np.inf] = delta_values[delta_values != np.inf].max()

	# print("Targeting :", easiest[analysis_start : analysis_start + n, 0])

	# Use totally random values
	# scale=1e2
	# delta_values = np.random.normal(mean, scale*std, size=(delta_values.shape[1], delta_values.shape[0])).T

	# Get loss coefficients using these delta values
	loss_coeffs = 1 / np.abs(scaled_delta_values)
	loss_coeffs /= np.sum(loss_coeffs, axis=0)

	# Get feature representation of current image
	with ch.no_grad():
		(_, image_rep), _  = model(real.cuda(), with_latent=True)

	# Consider only active neurons when targeting for attack
	if active_only:
		inactive_neurons = (image_rep[::n] == 0).cpu().numpy()
		# Transpose for ease
		delta_values = delta_values.T
		for i in range(inactive_neurons.shape[0]):
			delta_values[i][inactive_neurons[i]] = np.inf
		# Transpose back
		delta_values = delta_values.T

	if corr_analysis:
		easiest = np.arange(delta_values.shape[0])
		easiest = np.repeat(np.expand_dims(easiest, 1), len(images), axis=1)
	else:
		# Pick easiest-to-attack neurons per image
		easiest = np.argsort(np.abs(delta_values) / np.expand_dims(std, 1), axis=0)

	# Construct delta vector and indices mask
	delta_vec = ch.zeros_like(image_rep)
	indices_mask = ch.zeros_like(image_rep)
	for j in range(len(images)):
		for i, x in enumerate(easiest[analysis_start : analysis_start + n, j]):
			delta_vec[i + j * n, x] = delta_values[x, j]
			indices_mask[i + j * n, x] = 1

	# Construct delta vector and indices mask, totally haywire way
	# try_these = list(range(-n // 2, n // 2 + 1))
	# delta_vec = ch.zeros_like(image_rep)
	# indices_mask = ch.ones_like(image_rep)
	# for j in range(len(images)):
	# 	# for i, x in enumerate(easiest[analysis_start : analysis_start + n, j]):
	# 	for i in range(n):
	# 		# delta_vec[i + j * n, x] = delta_values[x, j]
	# 		delta_vec[i + j * n] = try_these[i]
	# 		# indices_mask[i + j * n, x] = 1

	# Use model's outputs or actual ground-truth labels?
	if goal:
		labels_ = ch.from_numpy(np.array(labels_)).cuda()
	else:
		labels_ = ch.argmax(model(real)[0], 1)

	impostors = parallel_impostor(model, delta_vec, real, labels_, indices_mask, loss_coeffs, optim_type,
		verbose, eps, iters, norm, custom_best, fake_relu, random_restarts)

	with ch.no_grad():
		if save_attack or delta_analysis or corr_analysis:
			(pred, latent), _ = model(impostors, with_latent=True)
		else:
			pred, _ = model(impostors)
			latent = None

	if dist_stats:
		flatten = (impostors - real.cuda()).view(impostors.shape[0], -1)
		dist_l2   = ch.norm(flatten, p=2, dim=-1).cpu().numpy()
		dist_linf = ch.max(ch.abs(flatten), dim=-1)[0].cpu().numpy()
	else:
		dist_l2, dist_linf = None, None

	label_pred = ch.argmax(pred, dim=1)

	clean_pred, _ = model(real)
	clean_pred = ch.argmax(clean_pred, dim=1)

	clean_preds = clean_pred.cpu().numpy()
	preds       = label_pred.cpu().numpy()

	succeeded = [[] for _ in range(len(images))]
	neuronwise_bincounts = np.zeros((n, delta_values.shape[0]), dtype=np.int32)
	if delta_analysis:
		delta_succeeded = [[] for _ in range(len(images))]
	for i in range(len(images)):
		for j in range(n):
			succeeded[i].append(preds[i * n + j] != clean_preds[i * n + j])
			if delta_analysis or corr_analysis:
				analysis_index = easiest[analysis_start : analysis_start + n, i][j]
				success_criterion = (latent[i * n + j] >= (image_rep[i * n + j] + delta_vec[i * n + j]))
				if delta_analysis:
					delta_succeeded[i].append(success_criterion[analysis_index].cpu().item())
				if corr_analysis:
					neuronwise_bincounts[j] += success_criterion.cpu().numpy()

	succeeded = np.array(succeeded)
	if delta_analysis:
		delta_succeeded = np.array(delta_succeeded, 'float')
	image_labels = [clean_preds, preds]

	if not delta_analysis:
		delta_succeeded = None

	if save_attack:
		return (real, impostors, image_labels, succeeded, latent.cpu().numpy(), delta_succeeded, dist_l2, dist_linf)
	return (real, impostors, image_labels, succeeded, None, delta_succeeded, neuronwise_bincounts, dist_l2, dist_linf)


def parallel_impostor(model, delta_vec, im, labels, indices_mask, l_c, optim_type, verbose, eps,
	iters, norm, custom_best, fake_relu, random_restarts):
	# Get feature representation of current image
	with ch.no_grad():
		(_, image_rep), _  = model(im.cuda(), with_latent=True, fake_relu=fake_relu)

	# Get target feature rep
	target_rep = image_rep + delta_vec

	# Construct loss coefficients
	loss_coeffs = np.tile(l_c, (im.shape[0], 1))
	loss_coeffs = ch.from_numpy(loss_coeffs).float().cuda()

	# Override custom_best, use cross-entropy on model instead
	criterion = ch.nn.CrossEntropyLoss(reduction='none').cuda()
	def ce_loss(loss, x):
		output, _ = model(x, fake_relu=fake_relu)
		# We want CE loss b/w new and old to be as high as possible
		return -criterion(output, labels)
	# Use CE loss
	if custom_best: custom_best = ce_loss

	if optim_type == 'madry':
		# Use Madry's optimization
		# Custom-Best (if True, look at i^th perturbation, not care about overall loss)
		im_matched = optimize.madry_optimization(model, im, target_rep, indices_mask,
			random_restart_targets=labels, eps=eps, iters=iters, verbose=verbose,
			p=norm, reg_weight=1e1, custom_best=custom_best, fake_relu=fake_relu,
			random_restarts=random_restarts)
	elif optim_type == 'natural':
		# Use natural gradient descent
		im_matched = optimize.natural_gradient_optimization(model, im, target_rep, indices_mask,
			eps=eps, iters=iters, #1e-2, 100
			reg_weight=1e0, verbose=verbose, p=norm)
	elif optim_type == 'custom':
		# Use custom optimization loop
		im_matched = optimize.custom_optimization(model, im, target_rep, indices_mask,
			eps=eps, iters=iters, #2.0, 200
			reg_weight=1e1, verbose=verbose, p=norm)
	else:
		print("Invalid optimization strategy. Exiting")
		exit(0)

	return im_matched


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='linf', help='type of model (nat/l2/linf)')
	parser.add_argument('--eps', type=float, default=0.031372549019608,  help='epsilon-iter')
	parser.add_argument('--iters', type=int, default=50, help='number of iterations')
	parser.add_argument('--n', type=int, default=16, help='number of neurons per image')
	parser.add_argument('--bs', type=int, default=64, help='batch size while performing attack')
	parser.add_argument('--longrun', type=bool, default=True, help='whether experiment is long running or for visualization (default)')
	parser.add_argument('--custom_best', type=bool, default=True, help='look at absoltue loss or perturbation for best-loss criteria')
	parser.add_argument('--image', type=str, default='visualize', help='name of file with visualizations (if enabled)')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet]')
	parser.add_argument('--norm', type=str, default='inf', help='P-norm to limit budget of adversary')
	parser.add_argument('--technique', type=str, default='madry', help='optimization strategy while searching for examples')
	parser.add_argument('--save_attack', type=str, default=None, help='path to save attack statistics (default: None, ie, do not save)')
	parser.add_argument('--analysis', type=bool, default=False, help='report neuron-wise attack success rates?')
	parser.add_argument('--delta_analysis', type=bool, default=False, help='report neuron-wise delta-achieve rates?')
	parser.add_argument('--corr_analysis', type=bool, default=False, help='log neuron-wise correlation statistics?')
	parser.add_argument('--random_restarts', type=int, default=0, help='how many random restarts? (0 -> False)')
	parser.add_argument('--analysis_start', type=int, default=0, help='index to start from (to capture n)')
	parser.add_argument('--distortion_statistics', type=bool, default=False, help='distortion statistics needed?')
	parser.add_argument('--goal', type=bool, default=False, help='is goal to maximize misclassification (True, default) or flip model predictions (False)')
	parser.add_argument('--active_only', type=bool, default=False, help='target only neurons that are already activated')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	image_save_name = args.image
	batch_size      = args.bs
	iters           = args.iters
	eps             = args.eps
	n               = args.n
	binary          = args.dataset == 'binarycifar10'
	norm            = args.norm
	opt_type        = args.technique
	save_attack     = args.save_attack
	custom_best     = args.custom_best
	fake_relu       = (model_arch != 'vgg19')
	analysis        = args.analysis
	delta_analysis  = args.delta_analysis
	analysis_start  = args.analysis_start
	random_restarts = args.random_restarts
	corr_analysis   = args.corr_analysis
	dist_stats      = args.distortion_statistics
	goal            = args.goal
	active_only     = args.active_only

	# Load model
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
	elif args.dataset == 'binarycifar10':
		constants = utils.BinaryCIFAR()
	else:
		print("Invalid Dataset Specified")
	ds = constants.get_dataset()

	# Load model
	model = constants.get_model(model_type , model_arch)
	# Get stats for neuron activations
	senses = constants.get_deltas(model_type, model_arch)

	senses_2 = utils.get_sensitivities("./deltas_train_cifar10_nat.txt")

	print(senses == senses_2)
	print(senses[-1])
	print(senses_2[-1])
	print(senses.shape)
	exit(0)

	(mean, std) = constants.get_stats(model_type, model_arch)
	# prefix = "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3"
	# print(prefix)
	# senses = utils.get_sensitivities(prefix + ".txt")
	# (mean, std) = utils.get_stats(prefix)

	if args.longrun:
		_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

		index_base, avg_successes = 0, 0
		attack_rates = [0, 0, 0, 0]
		l2_norm, linf_norm = 0, 0
		norm_count = 0
		impostors_latents = []
		all_impostors = []
		neuron_wise_success = []
		delta_wise_success  = []
		iterator = tqdm(test_loader)
		succcess_histograms = np.zeros((n, senses.shape[0]), np.int32)
		for (image, labels) in iterator:
			picked_indices = list(range(index_base, index_base + len(image)))
			(real, impostors, image_labels, succeeded, impostors_latent,
				delta_succeeded, neuronwise_bincounts, dist_l2, dist_linf) = find_impostors(model,
																senses[:, picked_indices], ds,
																image, labels, mean, std, n=n, binary=binary,
																verbose=False, eps=eps, iters=iters,
																optim_type=opt_type, norm=norm,
																save_attack=(save_attack != None),
																custom_best=custom_best, fake_relu=fake_relu,
																analysis_start=analysis_start, random_restarts=random_restarts,
																delta_analysis=delta_analysis, corr_analysis=corr_analysis,
																dist_stats=dist_stats, goal=goal, active_only=active_only)

			attack_rates[0] += np.sum(np.sum(succeeded[:, :1], axis=1) > 0)
			attack_rates[1] += np.sum(np.sum(succeeded[:, :8], axis=1) > 0)
			attack_rates[2] += np.sum(np.sum(succeeded[:, :16], axis=1) > 0)
			num_flips       = np.sum(succeeded, axis=1)
			attack_rates[3] += np.sum(num_flips > 0)
			avg_successes   += np.sum(num_flips)
			index_base      += len(image)
			if save_attack:
				all_impostors.append(impostors.cpu().numpy())
				impostors_latents.append(impostors_latent)
			if corr_analysis:
				succcess_histograms += neuronwise_bincounts
			# Keep track of distance statistics if asked
			if dist_stats:
				l2_norm   += np.sum(dist_l2)
				linf_norm += np.sum(dist_linf)
				norm_count += dist_l2.shape[0]
				dist_string = "L2 norm: %.3f, Linf norm: %.2f/255"  % (l2_norm / norm_count, 255 * linf_norm / norm_count)
			else:
				dist_string = ""
			# Keep track of attack success rate
			if goal:
				name_it = "Misclassification Rates"
			else:
				name_it = "Success Rates"
			iterator.set_description('(n=1,8,16,%d) %s : (%.2f, %.2f, %.2f, %.2f) | Flips/Image : %.2f/%d | %s' \
				% (n, name_it, 100 * attack_rates[0]/index_base,
					100 * attack_rates[1]/index_base,
					100 * attack_rates[2]/index_base,
					100 * attack_rates[3]/index_base,
					avg_successes / index_base, n, 
					dist_string))
			# Keep track of neuron-wise attack success rate
			if analysis:
				neuron_wise_success.append(succeeded)
			if delta_analysis:
				delta_wise_success.append(delta_succeeded)

		if analysis:
			neuron_wise_success = np.concatenate(neuron_wise_success, 0)
			neuron_wise_success = np.mean(neuron_wise_success, 0)
			for i in range(neuron_wise_success.shape[0]):
				print("Neuron %d attack success rate : %f %%" % (i + analysis_start, 100 * neuron_wise_success[i]))
			print()

		if delta_analysis:
			delta_wise_success = np.concatenate(delta_wise_success, 0)
			delta_wise_success = np.mean(delta_wise_success, 0)
			for i in range(delta_wise_success.shape[0]):
				print("Neuron %d acheiving-delta success rate : %f %%" % (i + analysis_start, 100 * delta_wise_success[i]))
			print()

		if corr_analysis:
			with open("%d_%d.txt" % (analysis_start, analysis_start + n), 'w') as f:
				for i in range(n):
					f.write("%d:%s\n" % (analysis_start + i, succcess_histograms[i].tolist()))
			print("Dumped correlation histograms for delta values in [%d,%d)" % (analysis_start, analysis_start + n))

		print("Attack success rate : %f %%" % (100 * attack_rates[-1]/index_base))
		print("Average flips per image : %f/%d" % (avg_successes / index_base, n))
		if save_attack:
			all_impostors     = np.concatenate(all_impostors, 0)
			impostors_latents = np.concatenate(impostors_latents, 0)
			impostors_latents_mean, impostors_latents_std = np.mean(impostors_latents, 0), np.std(impostors_latents, 0)
			np.save(save_attack + "_mean", impostors_latents_mean)
			np.save(save_attack + "_std", impostors_latents_std)
			np.save(save_attack + "_images", all_impostors)
			print("Saved activation statistics for adversarial inputs at %s" % save_attack)

	else:
		# Load all data
		all_data = utils.load_all_data(ds)

		# Visualize attack images
		picked_indices = list(range(batch_size))
		picked_images = [all_data[0][i] for i in picked_indices]
		picked_labels = [all_data[1][i] for i in picked_indices]
		(real, impostors, image_labels, succeeded, _, _, _, _, _) = find_impostors(model, senses[:, picked_indices], ds, picked_images, picked_labels, 
																mean, std, n=n, verbose=True, optim_type=opt_type, save_attack=(save_attack != None),
																eps=eps, iters=iters, binary=binary, norm=norm, custom_best=custom_best,
																fake_relu=fake_relu, analysis_start=analysis_start, random_restarts=random_restarts,
																delta_analysis=delta_analysis, corr_analysis=corr_analysis, dist_stats=dist_stats, goal=goal,
																active_only=active_only)

		show_image_row([real.cpu(), impostors.cpu()],
					["Real Images", "Attack Images"],
					tlist=image_labels,
					fontsize=22,
					filename="./visualize/%s.png" % image_save_name)
