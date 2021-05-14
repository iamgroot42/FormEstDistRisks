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


def batched_impostor(model, target_deltas, im, neuron_index, labels, batch_size):
	attacks = []
	for i in range(0, len(im), batch_size):
		impostors = parallel_impostor(model, target_deltas[i:i+batch_size], im[i:i+batch_size], neuron_index)
		pred, _ = model(impostors)
		label_pred = ch.argmax(pred, dim=1).cpu()
		attacks.append(1*(label_pred != labels[i:i+batch_size]))
	return np.concatenate(attacks)


def comprehensive_matrix(model, delta_values, ds):
	(images, labels) = load_all_data(ds)

	attack_matrix = np.ones(delta_values.shape)

	for i in range(delta_values.shape[0]):
		# Find cases where delta values are not infinity
		these_deltas = np.argwhere(delta_values[i] != np.inf).squeeze(1)
		
		delta_these  = delta_values[i][these_deltas]
		these_images = images[these_deltas]
		these_labels = labels[these_deltas]

		# Get impostors
		matches = batched_impostor(model, delta_these, these_images, i, these_labels, 64)

		# Set values in matrix
		attack_matrix[these_deltas] = matches

	return attack_matrix


def find_impostors(model, delta_values, ds, neuron_index, n=16):
	(image, label) = load_all_data(ds)
	
	easiest = np.argsort(delta_values)

	impostors = parallel_impostor(model, delta_values[easiest[:n]], image[easiest[:n]], neuron_index)
	real = image[easiest[:n]]

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

	image_labels = [clean_preds, preds]

	return (real, impostors, image_labels)


def parallel_impostor(model, target_deltas, im, neuron_index):
	# Get feature representation of current image
	(_, image_rep), _  = model(im.cuda(), with_latent=True)

	# Construct delta vector
	delta_vec = ch.zeros_like(image_rep)
	for i in range(target_deltas.shape[0]):
		delta_vec[i, neuron_index] = target_deltas[i]

	# Get target feature rep
	target_rep = image_rep + delta_vec

	# Modified inversion loss that puts emphasis on non-matching neurons to have similar activations
	def custom_inversion_loss(model, inp, targ):
		_, rep = model(inp, with_latent=True, fake_relu=True)
		# Normalized L2 error w.r.t. the target representation
		loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
		# Extra loss term
		reg_weight = 1e0
		mask = ch.ones_like(targ)
		mask[:, neuron_index] = 0
		loss_2 = ch.div(ch.norm(mask * (rep - targ), dim=1), ch.norm(targ * mask, dim=1))
		loss_3 = ch.abs((rep - targ)[:, neuron_index])
		# return loss, None
		return loss + reg_weight * loss_3, None
		return loss + reg_weight * loss_2, None

	# For now, consider [0,1] constrained images to see if any can be found
	kwargs = {
		# 'custom_loss': inversion_loss,
		'custom_loss': custom_inversion_loss,
		'constraint':'unconstrained',
		'eps': 1000,
		'step_size': 0.01,
		'iterations': 1000,
		'targeted': True,
		'do_tqdm': True
	}

	# Find image that minimizes this loss
	_, im_matched = model(im, target_rep, make_adv=True, **kwargs)

	# Return this image
	return im_matched


def best_target_neuron(mat, which=0):
	sum_m = []
	for i in range(mat.shape[0]):
		inf_counts = np.sum(mat[i] == np.inf)
		mat_interest = mat[i][mat[i] != np.inf]
		sum_m.append(np.average(np.abs(mat_interest)))
	best = np.argsort(sum_m)
	return best[which]


if __name__ == "__main__":
	import sys
	deltas_filepath = sys.argv[1]
	model_path = sys.argv[2]
	image_save_name = sys.argv[3]

	senses = get_sensitivities(deltas_filepath)
	# Pick feature with lowest average delta-requirement
	picked_feature = best_target_neuron(senses, 1)

	# Load model
	ds_path    = "./datasets/cifar_binary/animal_vehicle/"
	ds = GenericBinary(ds_path)

	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# comprehensive_matrix(model, senses, ds)

	# Visualize attack images
	(real, impostors, image_labels) = find_impostors(model, senses[picked_feature], ds, picked_feature)

	show_image_row([real.cpu(), impostors.cpu()], 
				["Real Images", "Attack Images"],
				tlist=image_labels,
				fontsize=22,
				filename="%s.png" % image_save_name)
