import os
import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from torch.autograd import Variable

import optimize, utils


def find_impostors(model, delta_values, ds, images, mean, std, optim_type='custom', verbose=True, n=4, eps=2.0, iters=200):
	image_ = []
	# Get target images
	for image in images:
		targ_img = image.unsqueeze(0)
		real = targ_img.repeat(n, 1, 1, 1)
		image_.append(real)
	real = ch.cat(image_, 0)

	# Get scaled senses
	scaled_delta_values = utils.scaled_values(delta_values, mean, std)

	# Pick easiest-to-attack neurons per image
	easiest = np.argsort(scaled_delta_values, axis=0)

	# Get loss coefficients using these delta values
	loss_coeffs = 1 / np.abs(scaled_delta_values)
	loss_coeffs /= np.sum(loss_coeffs, axis=0)

	# Get feature representation of current image
	(_, image_rep), _  = model(real.cuda(), with_latent=True)

	# Construct delta vector and indices mask
	delta_vec = ch.zeros_like(image_rep)
	indices_mask = ch.zeros_like(image_rep)
	for j in range(len(images)):
		for i, x in enumerate(easiest[:n, j]):
			delta_vec[i + j * n, x] = delta_values[x, j]
			indices_mask[i + j * n, x] = 1		

	impostors = parallel_impostor(model, delta_vec, real, indices_mask, loss_coeffs, optim_type, verbose, eps, iters)

	pred, _ = model(impostors)
	label_pred = ch.argmax(pred, dim=1)

	clean_pred, _ = model(real)
	clean_pred = ch.argmax(clean_pred, dim=1)

	mapping = ["animal", "vehicle"]

	clean_preds = [mapping[x] for x in clean_pred.cpu().numpy()]
	preds       = [mapping[x] for x in label_pred.cpu().numpy()]

	succeeded = [[] for _ in range(len(images))]
	for i in range(len(images)):
		for j in range(n):
			succeeded[i].append(preds[i * n + j] != clean_preds[i * n + j])
	succeeded = np.array(succeeded)
	image_labels = [clean_preds, preds]

	return (real, impostors, image_labels, np.sum(succeeded, axis=1))


def parallel_impostor(model, delta_vec, im, indices_mask, l_c, optim_type, verbose, eps, iters):
	# Get feature representation of current image
	(_, image_rep), _  = model(im.cuda(), with_latent=True)

	# Get target feature rep
	target_rep = image_rep + delta_vec

	# Construct loss coefficients
	loss_coeffs = np.tile(l_c, (im.shape[0], 1))
	loss_coeffs = ch.from_numpy(loss_coeffs).float().cuda()

	if optim_type == 'madry':
		# Use Madry's optimization
		im_matched = optimize.madry_optimization(model, im, target_rep, indices_mask,
			eps=eps, iters=iters, verbose=verbose) #1.0, 200
	elif optim_type == 'natural':
		# Use natural gradient descent
		im_matched = optimize.natural_gradient_optimization(model, im, target_rep, indices_mask,
			eps=eps, iters=iters, #1e-2, 100
			reg_weight=1e0, verbose=verbose)
	elif optim_type == 'custom':
		# Use custom optimization loop
		im_matched = optimize.custom_optimization(model, im, target_rep, indices_mask,
			eps=eps, p='2', iters=iters, #2.0, 200
			reg_weight=1e0, verbose=verbose)
	else:
		print("Invalid optimization strategy. Exiting")
		exit(0)

	return im_matched


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--deltas', type=str, help='path to file storing delta values')
	parser.add_argument('--model', type=str, help='path to model checkpoint')
	parser.add_argument('--eps', type=float, help='epsilon-iter')
	parser.add_argument('--iters', type=int, help='number of iterations')
	parser.add_argument('--n', type=int, default=8, help='number of neurons per image')
	parser.add_argument('--bs', type=int, default=8, help='batch size while performing attack')
	parser.add_argument('--longrun', type=bool, default=False, help='whether experiment is long running or for visualization (default)')
	parser.add_argument('--image', type=str, default='visualize', help='name of file with visualizations (if enabled)')
	parser.add_argument('--stats', type=str, help='path to directory containing mean and std of features')
	
	args = parser.parse_args()
	
	deltas_filepath = args.deltas
	model_path      = args.model
	image_save_name = args.image
	stats_path      = args.stats
	batch_size      = args.bs
	iters           = args.iters
	eps             = args.eps
	n               = args.n

	senses = utils.get_sensitivities(deltas_filepath)
	# Pick image with lowest average delta-requirement
	# picked_image = utils.best_target_image(senses, 8223)

	# Load model
	ds_path    = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
	ds = GenericBinary(ds_path)

	# Load model
	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# Get stats for neuron activations
	(mean, std) = utils.get_stats(stats_path)

	if args.longrun:
		_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

		index_base = 0
		attack_rate, avg_successes = 0, 0
		for (image, _) in test_loader:
			picked_indices = list(range(index_base, index_base + len(image)))
			(real, impostors, image_labels, num_flips) = find_impostors(model, senses[:, picked_indices], ds,
																image.cpu(), mean, std, n=n,
																verbose=True, eps=eps, iters=iters)
			index_base += len(image)
			attack_rate += np.sum(num_flips > 0)
			avg_successes += np.sum(num_flips)
			print(num_flips, attack_rate, avg_successes)

		print("Attack success rate : %f 	%%" % (100 * attack_rate/index_base))
		print("Average flips per image : %f/%d" % (avg_successes / index_base, n))
	else:
		# Load all data
		all_data = utils.load_all_data(ds)

		# Visualize attack images
		picked_indices = list(range(8))
		picked_images = [all_data[0][i] for i in picked_indices]
		(real, impostors, image_labels, num_flips) = find_impostors(model, senses[:, picked_indices], ds, picked_images, mean, std,
																verbose=True, eps=eps, iters=iters)

		show_image_row([real.cpu(), impostors.cpu()],
					["Real Images", "Attack Images"],
					tlist=image_labels,
					fontsize=22,
					filename="%s.png" % image_save_name)
