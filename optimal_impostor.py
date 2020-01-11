
import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm


def get_sensitivities(path):
	features = []
	with open(path, 'r') as f:
		for line in tqdm(f):
			values = np.array([float(x) for x in line.rstrip('\n').split(',')])
			features.append(values)
	return np.array(features)


# Use model-inversion to calculate input that corresponds to desirec diff vector
def inversion_loss(model, inp, targ):
	_, rep = model(inp, with_latent=True, fake_relu=True)
	# Normalized L2 error w.r.t. the target representation
	loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
	return loss, None


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


def find_impostors(model, delta_values, ds, image_index, n=16):
	(image, label) = load_all_data(ds)

	# Get target image
	targ_img = image[picked_image].unsqueeze(0)
	real = targ_img.repeat(n, 1, 1, 1)
	
	# Pick easiest-to-attack neurons for this image
	easiest = np.argsort(delta_values)

	impostors = parallel_impostor(model, delta_values[easiest[:n]], real, easiest[:n])

	diff = (real.cpu() - impostors.cpu()).view(n, -1)
	l1_norms   = ch.sum(ch.abs(diff), dim=1)
	l2_norms   = ch.norm(diff, dim=1)
	linf_norms = ch.max(ch.abs(diff), dim=1)[0]
	print("L-1   norms: ", ch.mean(l1_norms), "+-", ch.std(l1_norms))
	print("L-2   norms: ", ch.mean(l2_norms), "+-", ch.std(l2_norms))
	print("L-inf norms: ", ch.mean(linf_norms), "+-", ch.std(linf_norms))

	pred, _ = model(impostors)
	label_pred = ch.argmax(pred, dim=1)

	clean_pred, _ = model(real)
	clean_pred = ch.argmax(clean_pred, dim=1)

	mapping = ["animal", "vehicle"]

	clean_preds = [mapping[x] for x in clean_pred.cpu().numpy()]
	preds       = [mapping[x] for x in label_pred.cpu().numpy()]

	success = 0
	for i in range(len(preds)):
		success += (clean_preds[i] != preds[i])

	print("Label flipped for %d/%d examples" % (success, len(preds)))
	image_labels = [clean_preds, preds]

	return (real, impostors, image_labels)


def parallel_impostor(model, target_deltas, im, neuron_indices):
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

	# Modified inversion loss that puts emphasis on non-matching neurons to have similar activations
	def custom_inversion_loss(model, inp, targ):
		_, rep = model(inp, with_latent=True, fake_relu=True)
		# Normalized L2 error w.r.t. the target representation
		loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
		# Extra loss term
		reg_weight = 1e0
		aux_loss = ch.sum(ch.abs((rep - targ) * indices_mask), dim=1)
		return loss + reg_weight * aux_loss, None

	# For now, consider [0,1] constrained images to see if any can be found
	kwargs = {
		# 'custom_loss': inversion_loss,
		'custom_loss': custom_inversion_loss,
		'constraint':'unconstrained',
		'eps': 1000,
		'step_size': 0.01,
		'iterations': 2000,
		'targeted': True,
		'do_tqdm': True
	}

	# Find image that minimizes this loss
	_, im_matched = model(im, target_rep, make_adv=True, **kwargs)

	# Return this image
	return im_matched


def best_target_image(mat, which=0):
	sum_m = []
	for i in range(mat.shape[1]):
		# print(mat[mat[:, i] != np.inf].shape)
		mat_interest = mat[mat[:, i] != np.inf, i]
		sum_m.append(np.average(np.abs(mat_interest)))
	best = np.argsort(sum_m)
	return best[which]


if __name__ == "__main__":
	import sys
	deltas_filepath = sys.argv[1]
	model_path = sys.argv[2]
	image_save_name = sys.argv[3]

	senses = get_sensitivities(deltas_filepath)
	# Pick image with lowest average delta-requirement
	picked_image = best_target_image(senses, 4)

	# Load model
	ds_path    = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
	ds = GenericBinary(ds_path)

	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# Visualize attack images
	(real, impostors, image_labels) = find_impostors(model, senses[:, picked_image], ds, picked_image)

	show_image_row([real.cpu(), impostors.cpu()], 
				["Real Images", "Attack Images"],
				tlist=image_labels,
				fontsize=22,
				filename="%s.png" % image_save_name)
