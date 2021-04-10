# Look at feature activations for models across dataset and identify sets of highly-correlated features
# Perform this analysis for normal, PGD trained, and custom-reg trained models
import torch as ch
import numpy as np
import utils
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

from tqdm import tqdm

# First technique : PCA on test data
dx = utils.CIFAR10()
ds = dx.get_dataset()

model_arch = "vgg19"
# model_type = sys.argv[1]
# model_types = ["nat", "l2", "linf", "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"]
# model_type_names = ["normal", "L-2 adv", "L-inf adv", "Robust"]
model_types = ["nat"]
model_type_names = ["VGG-19 Model"]

for model_name, model_type in zip(model_type_names, model_types):
	model = dx.get_model(model_type, model_arch)

	batch_size = 1024
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	X = []
	for (im, label) in tqdm(test_loader):
		with ch.no_grad():
			(_, features), _ = model(im.cuda(), with_latent=True)
			X.append(features.cpu().numpy())

	X = np.concatenate(X, 0)
	pca = PCA()
	pca.fit(X)

	shortlist = pca.explained_variance_ratio_[:30]
	x = list(range(shortlist.shape[0]))
	plt.plot(x, shortlist, label=model_name)

plt.legend()
plt.title('Eigenvalue analysis (correlation : test data)')
plt.grid()
plt.savefig("visualize/eigenvalues.png")
