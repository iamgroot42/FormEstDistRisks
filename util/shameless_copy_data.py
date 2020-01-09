import torch as ch
from robustness.datasets import GenericBinary, CIFAR

import os
from PIL import Image
import numpy as np


# Save given image into D_robust
def add_to_dataset(ds, datum, suffix, class_names, already_image=False):
	if not os.path.exists(os.path.join(ds.data_path, suffix)):
		raise AssertionError("Folder does not exist: %s" % os.path.join(ds.data_path, suffix))
	(x, y) = datum
	class_name = class_names[y]
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
def add_batch_to_dataset(ds, data, suffix, class_names, already_image=False):
	for i in range(len(data[0])):
		add_to_dataset(ds, (data[0][i], data[1][i].item()), suffix, class_names, already_image=already_image)

ds = CIFAR()
ds_binary = GenericBinary("/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct")

class_names = ['vehicle', 'vehicle', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'vehicle', 'vehicle']

train_loader, test_loader = ds.make_loaders(workers=10, batch_size=64, data_aug=False)
train_data_iterator       = enumerate(train_loader)
test_data_iterator        = enumerate(test_loader)

# Iterate through al of training data
for i, data in train_data_iterator:
	(im, targ) = data
	add_batch_to_dataset(ds_binary, (im.cpu().numpy(), targ.numpy()), "train", class_names)

# Iterate through al of testing data
for i, data in test_data_iterator:
	(im, targ) = data
	add_batch_to_dataset(ds_binary, (im.cpu().numpy(), targ.numpy()), "test", class_names)
