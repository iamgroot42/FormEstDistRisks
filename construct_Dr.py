import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

ds = CIFAR()
model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': './models/cifar_nat.pt'
}

model, _ = make_and_restore_model(**model_kwargs)
model.eval()

train_loader, _ = ds.make_loaders(workers=10, batch_size=128, data_aug=False)
data_iterator = enumerate(train_loader)


# Save given image into D_robust
def add_to_dataset(ds, datum, suffix):
	if not os.path.exists(os.path.join(ds.data_path, suffix)):
		raise AssertionError("Folder does not exist: %s" % os.path.join(ds.data_path, suffix))
	(x, y) = datum
	class_name = ds.label_mapping[y.numpy().item()]
	# Create class-name folder if it does not exist already
	desired_folder = os.path.join(os.path.join(ds.data_path, suffix), class_name)
	if not os.path.exists(desired_folder):
		os.makedirs(desired_folder)
		big_index = 0
	else:
		# Get biggest index file in here so far
		big_index = max([int(x.split('.')[0]) for x in os.listdir(desired_folder)])
	 # Permute to desired shape
	x_cpu = x.cpu().numpy()
	image = Image.fromarray((255 * np.transpose(x_cpu, (1, 2, 0))).astype('uint8'))
	image.save(os.path.join(desired_folder, str(big_index + 1) + ".png"))


# Add batches of images to R_robust
def add_batch_to_dataset(ds, data, suffix):
	for i in tqdm(range(len(data[0]))):
		add_to_dataset(ds, (data[0][i], data[1][i]), suffix)


# Custom loss for inversion
def inversion_loss(model, inp, targ):
	_, rep = model(inp, with_latent=True, fake_relu=True)
	loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
	return loss, None

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

# Select images to invert (random samples from the test set)
_, (im, targ) = next(data_iterator) # Images to invert

ds_robust = CIFAR("./datasets/cifar_dr")

with ch.no_grad():
	(_, rep), _ = model(im.cuda(), with_latent=True) # Corresponding representation

im_n = ch.randn_like(im) / 20 + 0.5 # Seed for inversion (x_0)

_, xadv = model(im_n, rep.clone(), make_adv=True, **kwargs)

add_batch_to_dataset(ds_robust, (xadv, targ), "train")
