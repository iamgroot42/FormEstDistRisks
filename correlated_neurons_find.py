import torch as ch
import utils
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
	import sys

	model_type   = sys.argv[1]
	layer_index  = int(sys.argv[2])
	filepath     = sys.argv[3]
	dataset      = "cifar10"

	if dataset == 'cifar10':
		dx = utils.CIFAR10()
	elif dataset == 'imagenet':
		dx = utils.ImageNet1000()
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, "vgg19")

	batch_size = 128
	all_reps = []
	train_loader = None
	if dataset == 'cifar10':
		train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)
	else:
		_, val_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True)

	def get_reps(data_loader):
		with ch.no_grad():
			for (im, label) in tqdm(data_loader):
				rep, _ = model(im, this_layer_output=layer_index, just_latent=True)
				# rep, _ = model(im, with_latent=True, just_latent=True)
				all_reps.append(rep.cpu())

	def get_random_reps(bs, n_times):
		with ch.no_grad():
			for i in tqdm(range(n_times)):
				im = ch.rand(bs, 3, 32, 32)
				rep, _ = model(im, this_layer_output=layer_index, just_latent=True)
				# rep, _ = model(im, with_latent=True, just_latent=True)
				all_reps.append(rep.cpu())


	get_random_reps(600, 100)

	if train_loader:
		get_reps(train_loader)
	get_reps(val_loader)

	all_reps       = ch.cat(all_reps).numpy() 
	all_reps = all_reps.reshape(all_reps.shape[0], -1)

	print(all_reps.shape)

	correlations = np.corrcoef(all_reps, rowvar=False)

	clusters = {}
	threshold = 0.9
	not_covered = np.ones((correlations.shape[0]))
	for i in range(correlations.shape[0]):
		if not_covered[i]:
			clusters[i] = [i]
			for j in range(i + 1, correlations.shape[1]):
				if correlations[i][j] >= threshold and not_covered[j]:
					clusters[i] = clusters.get(i) + [j]
					not_covered[j] = 0
			not_covered[i] = 0

	with open("%s.txt" % filepath, 'w') as f:
		for k, v in clusters.items():
			v_ = [str(i) for i in v]
			f.write(",".join(v_))
			f.write("\n")
	# Dump mean, std vectors for later use:
	# np.save(filepath + "feature_mean", np_mean)
