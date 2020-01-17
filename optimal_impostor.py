import os
import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from torch.autograd import Variable
from tqdm import tqdm

import optimize


def get_sensitivities(path):
	features = []
	with open(path, 'r') as f:
		for line in tqdm(f):
			values = np.array([float(x) for x in line.rstrip('\n').split(',')])
			features.append(values)
	return np.array(features)


def load_all_data(ds):
	batch_size = 512
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	images, labels = [], []
	for (image, label) in test_loader:
		images.append(image)
		labels.append(label)
	labels = ch.cat(labels).cpu()
	images = ch.cat(images).cpu()
	return (images, labels)


def find_impostors(model, delta_values, ds, image_index, all_data, mean, std, n=8):
	(image, label) = all_data
	# Get target image
	targ_img = image[image_index].unsqueeze(0)
	real = targ_img.repeat(n, 1, 1, 1)
	
	# Get scaled senses
	scaled_delta_values = (delta_values - mean) / (std + 1e-10)

	# Pick easiest-to-attack neurons for this image
	easiest = np.argsort(scaled_delta_values)

	# Get loss coefficients using these delta values
	loss_coeffs = 1 / np.abs(scaled_delta_values)
	loss_coeffs /= np.sum(loss_coeffs)

	impostors = parallel_impostor(model, delta_values[easiest[:n]], real, easiest[:n], loss_coeffs)

	diff = (real.cpu() - impostors.cpu()).view(n, -1)

	pred, _ = model(impostors)
	label_pred = ch.argmax(pred, dim=1)

	clean_pred, _ = model(real)
	clean_pred = ch.argmax(clean_pred, dim=1)

	mapping = ["animal", "vehicle"]

	clean_preds = [mapping[x] for x in clean_pred.cpu().numpy()]
	preds       = [mapping[x] for x in label_pred.cpu().numpy()]

	success = 0
	succeeded = []
	for i in range(len(preds)):
		success += (clean_preds[i] != preds[i])
		if clean_preds[i] == preds[i]:
			succeeded.append(False)
		else:
			succeeded.append(True)

	# Calculate metrics (for cases where attack succeeded)
	diff_care = diff[succeeded]
	if len(diff_care) > 0:
		l1_norms   = ch.sum(ch.abs(diff_care), dim=1)
		l2_norms   = ch.norm(diff_care, dim=1)
		linf_norms = ch.max(ch.abs(diff_care), dim=1)[0]
		print("L-1   norms: [", ch.min(l1_norms), ",", ch.max(l1_norms), "]")
		print("L-2   norms: [", ch.min(l2_norms), ",", ch.max(l2_norms), "]")
		print("L-inf norms: [", ch.min(linf_norms), ",", ch.max(linf_norms), "]")
		print("Label flipped for %d/%d examples" % (success, len(preds)))
	else:
		print("Attack did not succeed")

	relative_num_flips = float(success) / len(preds)
	image_labels = [clean_preds, preds]

	return (real, impostors, image_labels, relative_num_flips)


def parallel_impostor(model, target_deltas, im, neuron_indices, l_c):
	# Get feature representation of current image
	(_, image_rep), _  = model(im.cuda(), with_latent=True)

	# Construct delta vector
	delta_vec = ch.zeros_like(image_rep)
	for i, x in enumerate(neuron_indices):
		delta_vec[i, x] = target_deltas[i]

	# Get target feature rep
	target_rep = image_rep + delta_vec
	indices_mask = ch.zeros_like(image_rep)
	for i in range(indices_mask.shape[0]):
		indices_mask[i][neuron_indices[i]] = 1

	# Construct loss coefficients
	loss_coeffs = np.tile(l_c, (im.shape[0], 1))
	loss_coeffs = ch.from_numpy(loss_coeffs).float().cuda()

	# Use Madry's optimization
	# im_matched = optimize.madry_optimization(model, im, target_rep, indices_mask,
	# 	eps=0.5, verbose=True)

	# Use custom optimization loop
	im_matched = optimize.custom_optimization(model, im, target_rep, indices_mask,
		eps=2.5, p='2', iters=200,
		reg_weight=1e0, verbose=True)

	# Use natural gradient descent
	# im_matched = optimize.natural_gradient_optimization(model, im, target_rep, indices_mask,
	# 	eps=1e-2, iters=100,
	# 	reg_weight=1e0, verbose=True)

	return im_matched


def best_target_image(mat, which=0):
	sum_m = []
	for i in range(mat.shape[1]):
		# print(mat[mat[:, i] != np.inf].shape)
		mat_interest = mat[mat[:, i] != np.inf, i]
		sum_m.append(np.average(np.abs(mat_interest)))
	best = np.argsort(sum_m)
	return best[which]


def get_stats(base_path):
	mean = np.load(os.path.join(base_path, "feature_mean.npy"))
	std  = np.load(os.path.join(base_path, "feature_std.npy"))
	return mean, std


if __name__ == "__main__":
	import sys
	deltas_filepath = sys.argv[1]
	model_path      = sys.argv[2]
	image_save_name = sys.argv[3]
	stats_path      = sys.argv[4]

	senses = get_sensitivities(deltas_filepath)
	# Pick image with lowest average delta-requirement
	picked_image = best_target_image(senses, 8223)

	# Load model
	ds_path    = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
	ds = GenericBinary(ds_path)

	# Load all data
	all_data = load_all_data(ds)

	# Load model
	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# Get stats for neuron activations
	(mean, std) = get_stats(stats_path)

	# Visualize attack images
	picked_image = 7233
	(real, impostors, image_labels, num_flips) = find_impostors(model, senses[:, picked_image], ds, picked_image, all_data, mean, std)

	show_image_row([real.cpu(), impostors.cpu()], 
				["Real Images", "Attack Images"],
				tlist=image_labels,
				fontsize=22,
				filename="%s.png" % image_save_name)
