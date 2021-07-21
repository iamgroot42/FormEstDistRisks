import torch as ch
from robustness.datasets import RobustCIFAR, CIFAR
from robustness.model_utils import make_and_restore_model

import os
from PIL import Image
import numpy as np


# Save given image into D_robust
def add_to_dataset(ds, datum, suffix, already_image=False):
	if not os.path.exists(os.path.join(ds.data_path, suffix)):
		raise AssertionError("Folder does not exist: %s" % os.path.join(ds.data_path, suffix))
	(x, y) = datum
	class_name = ds.class_names[y]
	# Create class-name folder if it does not exist already
	desired_folder = os.path.join(os.path.join(ds.data_path, suffix), class_name)
	if not os.path.exists(desired_folder):
		os.makedirs(desired_folder)
		big_index = 0
	else:
		# Get biggest index file in here so far
		big_index = max([int(x.split('.')[0]) for x in os.listdir(desired_folder)])
	 # Permute to desired shape
	if already_image:
		image = x
	else:
		image = Image.fromarray((255 * np.transpose(x, (1, 2, 0))).astype('uint8'))
	image.save(os.path.join(desired_folder, str(big_index + 1) + ".png"))


# Add batches of images to R_robust
def add_batch_to_dataset(ds, data, suffix, already_image=False):
	for i in range(len(data[0])):
		add_to_dataset(ds, (data[0][i], data[1][i].item()), suffix, already_image=already_image)


# Custom loss for inversion
def inversion_loss(model, inp, targ):
	_, rep = model(inp, with_latent=True, fake_relu=True)
	loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
	return loss, None


ds = CIFAR()
ds_sample = CIFAR()
ds_robust = RobustCIFAR("./datasets/cifar_dr_linf")

model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': './models/cifar_linf_8.pt'
}

model, _ = make_and_restore_model(**model_kwargs)
model.eval()

train_loader, _        = ds.make_loaders(workers=10, batch_size=64, data_aug=False)
data_iterator          = enumerate(train_loader)

train_loader_sample, _ = ds_sample.make_loaders(workers=10, batch_size=64, data_aug=False)
data_iterator_sample   = enumerate(train_loader_sample)

# PGD parameters
kwargs = {
	'custom_loss': inversion_loss,
	'constraint':'2',
	'eps':1000,
	'step_size': 0.1,
	'iterations': 1000, 
	'do_tqdm': True,
	'targeted': True,
	'use_best': False
}

# Iterate through al of training data
for i, data in data_iterator:
	(im, targ) = data
	
	with ch.no_grad():
		(_, rep), _ = model(im.cuda(), with_latent=True) # Get Corresponding representation

	# Seed for inversion (x_0)
	_, (im_n, targ_n) = next(data_iterator_sample)

	# Procedure to generate Dr
	_, xadv = model(im_n, rep.clone(), make_adv=True, **kwargs)

	# Save set of attacked image
	add_batch_to_dataset(ds_robust, (xadv.cpu().numpy(), targ.numpy()), "train")
