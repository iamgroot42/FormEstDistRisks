import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm


def get_stats(base_path):
	mean = np.load(os.path.join(base_path, "feature_mean.npy"))
	std  = np.load(os.path.join(base_path, "feature_std.npy"))
	return (np.expand_dims(mean, 1), np.expand_dims(std, 1))


def scale_senses(senses, mean, std):
	return (senses - np.repeat(mean, senses.shape[1], axis=1)) / (std + np.finfo(float).eps)


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


def load_all_data(ds, specific=None):
	batch_size = 512
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	images, labels = [], []
	for (image, label) in test_loader:
		images.append(image)
		labels.append(label)
	labels = ch.cat(labels).cpu()
	images = ch.cat(images).cpu()
	if specific:
		return(images[specific], labels[specific])
	return (images, labels)


def find_impostors(model, delta_values, ds, neuron_index, n=10):
	(image, label) = load_all_data(ds)
	
	easiest = np.argsort(delta_values)
	real = []
	impostors = []
	labels = []
	preds = []
	for i in range(n):
		impostor = find_impostor(model, delta_values[easiest[i]], image[easiest[i]].unsqueeze(0), neuron_index)
		real.append(image[easiest[i]])
		impostors.append(impostor)
		# Get model prediction
		pred, _ = model(impostor.unsqueeze(0))
		label_pred = ch.argmax(pred, dim=1)[0]

		labels.append(str(label[easiest[i]].numpy()))
		preds.append(str(label_pred.cpu().numpy()))

	image_labels = [labels, preds]

	# Convert to tensors
	real = ch.stack(real)
	impostors = ch.stack(impostors)

	return (real, impostors, image_labels)


def find_impostor(model, target_delta, im, neuron_index):
	# Get feature representation of current image
	(_, image_rep), _  = model(im.cuda(), with_latent=True)

	# Construct delta vector
	delta_vec = ch.zeros_like(image_rep)
	delta_vec[0][neuron_index] = target_delta

	# Get target feature rep
	target_rep = image_rep + delta_vec

	# For now, consider [0,1] constrained images to see if any can be found
	kwargs = {
		'custom_loss': inversion_loss,
		'constraint':'unconstrained',
		'eps': 1000,
		'step_size': 0.1,
		'iterations': 1000,
		'targeted': True,
		'do_tqdm': True
	}

	# Find image that minimizes this loss
	_, im_matched = model(im, target_rep, make_adv=True, **kwargs)

	# Return this image
	return im_matched[0]


if __name__ == "__main__":
	import sys
	senses = get_sensitivities(sys.argv[1])
	picked_feature = 0

	# Load model
	ds_path    = "./datasets/cifar_binary/animal_vehicle/"
	ds = GenericBinary(ds_path)

	model_path = sys.argv[2]
	model_kwargs = {
		'arch': 'resnet50',
		'dataset': ds,
		'resume_path': model_path
	}
	model, _ = make_and_restore_model(**model_kwargs)
	model.eval()

	# Visualize attack images
	(real, impostors, image_labels) = find_impostors(model, senses[picked_feature], ds, picked_feature)

	show_image_row([real.cpu(), impostors.cpu()], 
				["Real Images", "Attack Images"],
				tlist=image_labels,
				fontsize=22,
				filename="impostors_linf.png")
