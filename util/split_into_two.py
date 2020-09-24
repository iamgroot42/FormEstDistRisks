import torch as ch
from robustness.datasets import CIFAR
import utils
import os
from tqdm import tqdm
from PIL import Image
import numpy as np


# Save given image into D_robust
def add_to_dataset(prefix, X, Y, class_names):
	for i, x in tqdm(enumerate(X)):
		class_name = class_names[Y[i]]
		# Create class-name folder if it does not exist already
		desired_folder = os.path.join(os.path.join(prefix), class_name)
		
		# Get biggest index file in here so far
		if len(os.listdir(desired_folder)) > 0:
			big_index = max([int(x.split('.')[0]) for x in os.listdir(desired_folder)])
		else:
			big_index = 0

		# Convert to PIL image for saving
		image = Image.fromarray((255 * np.transpose(x, (1, 2, 0))).astype('uint8'))
		image.save(os.path.join(desired_folder, str(big_index + 1) + ".png"))


if __name__ == "__main__":
	import sys
	split_ratio = float(sys.argv[1])

	# Ready data loaders
	ds = CIFAR("/p/adversarialml/as9rw/datasets/cifar10")
	n_classes = len(ds.class_names)
	train_loader, test_loader = ds.make_loaders(workers=10, batch_size=256, data_aug=False)

	# Read train, test datasets
	X_train, Y_train = utils.load_all_loader_data(train_loader)
	X_test, Y_test   = utils.load_all_loader_data(test_loader)
	X_train, Y_train = X_train.numpy(), Y_train.numpy()
	X_test, Y_test   = X_test.numpy(), Y_test.numpy()

	# Get class-wise random indices, split into two
	indices_1, indices_2 = [], []
	for i in range(n_classes):
		permuted_indices = np.random.permutation(np.nonzero((Y_train == i)))[0]
		split_point = int(len(permuted_indices) * split_ratio)
		indices_1.append(permuted_indices[:split_point])
		indices_2.append(permuted_indices[split_point:])
	# Join all indices together
	indices_1 = np.concatenate(np.array(indices_1))
	indices_2 = np.concatenate(np.array(indices_2))

	# Folder paths
	folder_path_1 = "/p/adversarialml/as9rw/datasets/cifar10_split1"
	folder_path_2 = "/p/adversarialml/as9rw/datasets/cifar10_split2"

	# Create these folders
	if not os.path.exists(folder_path_1): os.mkdir(folder_path_1)
	if not os.path.exists(folder_path_2): os.mkdir(folder_path_2)

	# Make class-level folders
	for class_name in ds.class_names:
		os.makedirs(os.path.join(folder_path_1, "train", class_name))
		os.makedirs(os.path.join(folder_path_2, "train", class_name))
		os.makedirs(os.path.join(folder_path_1, "test", class_name))
		os.makedirs(os.path.join(folder_path_2, "test", class_name))

	# Copy train-data (splits) accordingly
	add_to_dataset(os.path.join(folder_path_1, "train"), X_train[indices_1], Y_train[indices_1], ds.class_names)
	add_to_dataset(os.path.join(folder_path_2, "train"), X_train[indices_2], Y_train[indices_2], ds.class_names)
	# Copy test-data (same) accordingly
	add_to_dataset(os.path.join(folder_path_1, "test"), X_test, Y_test, ds.class_names)
	add_to_dataset(os.path.join(folder_path_2, "test"), X_test, Y_test, ds.class_names)