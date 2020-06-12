import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable

import optimize, utils


def find_impostors(model, delta_values, ds, images, mean, std,
	optim_type='custom', verbose=True, n=4, eps=2.0, iters=200,
	binary=True, norm='2', save_attack=False, custom_best=False,
	fake_relu=True, analysis_start=0, random_restarts=0, 
	delta_analysis=False):
	image_ = []
	# Get target images
	for image in images:
		targ_img = image.unsqueeze(0)
		real = targ_img.repeat(n, 1, 1, 1)
		image_.append(real)
	real = ch.cat(image_, 0)

	# Get scaled senses
	scaled_delta_values = utils.scaled_values(delta_values, mean, std, eps=0)
	# Replace inf values with largest non-inf values
	delta_values[delta_values == np.inf] = delta_values[delta_values != np.inf].max()

	# Pick easiest-to-attack neurons per image
	# easiest = np.argsort(scaled_delta_values, axis=0)
	easiest = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 17, 18, 19, 22, 23, 24, 26, 29, 32, 33, 35, 39, 45, 50, 51, 57, 62, 67, 68, 69, 73, 75, 78, 81, 85, 88, 95, 101, 106, 109, 110, 114, 118, 119, 120, 123, 125, 136, 137, 146, 148, 149, 150, 151, 157, 161, 172, 180, 188, 196, 199, 201, 203, 214, 225, 226, 231, 241, 242, 244, 245, 251, 269, 276, 286, 297, 307, 314, 315, 316, 317, 319, 320, 323, 325, 330, 332, 337, 344, 347, 348, 349, 351, 360, 381, 382, 386, 395, 396, 407, 415, 417, 420, 429, 437, 440, 451, 460, 474, 476, 478, 490, 495, 500, 504, 505, 508])
	easiest = np.repeat(np.expand_dims(easiest, 1), len(images), axis=1)

	# Get loss coefficients using these delta values
	loss_coeffs = 1 / np.abs(scaled_delta_values)
	loss_coeffs /= np.sum(loss_coeffs, axis=0)

	# Get feature representation of current image
	with ch.no_grad():
		(_, image_rep), _  = model(real.cuda(), with_latent=True)

	# Construct delta vector and indices mask
	delta_vec = ch.zeros_like(image_rep)
	indices_mask = ch.zeros_like(image_rep)
	for j in range(len(images)):
		for i, x in enumerate(easiest[analysis_start : analysis_start + n, j]):
			delta_vec[i + j * n, x] = delta_values[x, j]
			indices_mask[i + j * n, x] = 1		

	impostors = parallel_impostor(model, delta_vec, real, indices_mask, loss_coeffs, optim_type,
		verbose, eps, iters, norm, custom_best, fake_relu, random_restarts)

	with ch.no_grad():
		if save_attack or delta_analysis:
			(pred, latent), _ = model(impostors, with_latent=True)
		else:
			pred, _ = model(impostors)
			latent = None
	label_pred = ch.argmax(pred, dim=1)

	clean_pred, _ = model(real)
	clean_pred = ch.argmax(clean_pred, dim=1)

	clean_preds = clean_pred.cpu().numpy()
	preds       = label_pred.cpu().numpy()

	succeeded = [[] for _ in range(len(images))]
	if delta_analysis:
		delta_succeeded = [[] for _ in range(len(images))]
	for i in range(len(images)):
		for j in range(n):
			succeeded[i].append(preds[i * n + j] != clean_preds[i * n + j])
			if delta_analysis:
				analysis_index = easiest[analysis_start : analysis_start + n, i][j]
				success_criterion = (latent[i * n + j] >= (image_rep[i * n + j] + delta_vec[i * n + j]))
				delta_succeeded[i].append(success_criterion[analysis_index].cpu().item())
	succeeded = np.array(succeeded)
	if delta_analysis:
		delta_succeeded = np.array(delta_succeeded, 'float')
	image_labels = [clean_preds, preds]

	if not delta_analysis:
		delta_succeeded = None

	if save_attack:
		return (real, impostors, image_labels, succeeded, latent.cpu().numpy(), delta_succeeded)
	return (real, impostors, image_labels, succeeded, None, delta_succeeded)


def parallel_impostor(model, delta_vec, im, indices_mask, l_c, optim_type, verbose, eps,
	iters, norm, custom_best, fake_relu, random_restarts):
	# Get feature representation of current image
	with ch.no_grad():
		(target_logits, image_rep), _  = model(im.cuda(), with_latent=True, fake_relu=fake_relu)
		target_logits = ch.argmax(target_logits, dim=1)

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
		return -criterion(output, target_logits)
	# Use CE loss
	if custom_best: custom_best = ce_loss

	if optim_type == 'madry':
		# Use Madry's optimization
		# Custom-Best (if True, look at i^th perturbation, not care about overall loss)
		im_matched = optimize.madry_optimization(model, im, target_rep, indices_mask,
			random_restart_targets=target_logits, eps=eps, iters=iters, verbose=verbose,
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
	parser.add_argument('--model_type', type=str, default='nat', help='type of model (nat/l2/linf)')
	parser.add_argument('--eps', type=float, help='epsilon-iter')
	parser.add_argument('--iters', type=int, default=50, help='number of iterations')
	parser.add_argument('--n', type=int, default=8, help='number of neurons per image')
	parser.add_argument('--bs', type=int, default=8, help='batch size while performing attack')
	parser.add_argument('--longrun', type=bool, default=False, help='whether experiment is long running or for visualization (default)')
	parser.add_argument('--custom_best', type=bool, default=False, help='look at absoltue loss or perturbation for best-loss criteria')
	parser.add_argument('--image', type=str, default='visualize', help='name of file with visualizations (if enabled)')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet]')
	parser.add_argument('--norm', type=str, default='2', help='P-norm to limit budget of adversary')
	parser.add_argument('--technique', type=str, default='madry', help='optimization strategy while searching for examples')
	parser.add_argument('--save_attack', type=str, default=None, help='path to save attack statistics (default: None, ie, do not save)')
	parser.add_argument('--analysis', type=bool, default=False, help='report neuron-wise attack success rates?')
	parser.add_argument('--delta_analysis', type=bool, default=False, help='report neuron-wise delta-achieve rates?')
	parser.add_argument('--random_restarts', type=int, default=0, help='how many random restarts? (0 -> False)')
	parser.add_argument('--analysis_start', type=int, default=0, help='index to start from (to capture n). used only when analysis flag is set')
	
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
	(mean, std) = constants.get_stats(model_type, model_arch)
	# prefix = "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3"
	# prefix = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1"
	# print(prefix)
	# senses = utils.get_sensitivities(prefix + ".txt")
	# (mean, std) = utils.get_stats(prefix)

	if args.longrun:
		_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

		index_base, avg_successes = 0, 0
		attack_rates = [0, 0, 0, 0]
		impostors_latents = []
		all_impostors = []
		neuron_wise_success = []
		delta_wise_success  = []
		iterator = tqdm(test_loader)
		for (image, _) in iterator:
			picked_indices = list(range(index_base, index_base + len(image)))
			(real, impostors, image_labels, succeeded, impostors_latent, delta_succeeded) = find_impostors(model, senses[:, picked_indices], ds,
																image.cpu(), mean, std, n=n, binary=binary,
																verbose=False, eps=eps, iters=iters,
																optim_type=opt_type, norm=norm,
																save_attack=(save_attack != None),
																custom_best=custom_best, fake_relu=fake_relu,
																analysis_start=analysis_start, random_restarts=random_restarts,
																delta_analysis=delta_analysis)

			attack_rates[0] += np.sum(np.sum(succeeded[:, :1], axis=1) > 0)
			attack_rates[1] += np.sum(np.sum(succeeded[:, :4], axis=1) > 0)
			attack_rates[2] += np.sum(np.sum(succeeded[:, :8], axis=1) > 0)
			num_flips = np.sum(succeeded, axis=1)
			attack_rates[3] += np.sum(num_flips > 0)
			avg_successes += np.sum(num_flips)
			index_base += len(image)
			if save_attack:
				all_impostors.append(impostors.cpu().numpy())
				impostors_latents.append(impostors_latent)
			# Keep track of attack success rate
			iterator.set_description('(n=1,4,8,%d) Success rates : (%.2f, %.2f, %.2f, %.2f) | | Flips/Image : %.2f/%d' \
				% (n, 100 * attack_rates[0]/index_base,
					100 * attack_rates[1]/index_base,
					100 * attack_rates[2]/index_base,
					100 * attack_rates[3]/index_base,
					avg_successes / index_base, n))
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
		(real, impostors, image_labels, succeeded, _, delta_succeeded) = find_impostors(model, senses[:, picked_indices], ds, picked_images, mean, std,
																n=n, verbose=True, optim_type=opt_type, save_attack=(save_attack != None),
																eps=eps, iters=iters, binary=binary, norm=norm, custom_best=custom_best,
																fake_relu=fake_relu, analysis_start=analysis_start, random_restarts=random_restarts,
																delta_analysis=delta_analysis)

		show_image_row([real.cpu(), impostors.cpu()],
					["Real Images", "Attack Images"],
					tlist=image_labels,
					fontsize=22,
					filename="./visualize/%s.png" % image_save_name)
