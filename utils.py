import torch as ch
import numpy as np
from torchvision import transforms
from robustness.datasets import GenericBinary, CIFAR
from robustness.tools import folder
from tqdm import tqdm
import sys
import os


class DataPaths:
	def __init__(self, name, data_path):
		self.name      = name
		self.data_path = data_path
		self.dataset   = self.dataset_type(data_path)

	def get_dataset(self):
		return self.dataset

class BinaryCIFAR(DataPaths):
	def __init__(self):
		self.dataset_type = GenericBinary
		super(BinaryCIFAR, self).__init__('binary cifar10', "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct")


class CIFAR10(DataPaths):
	def __init__(self):
		self.dataset_type = CIFAR
		super(CIFAR10, self).__init__('cifar10', "/p/adversarialml/as9rw/datasets/cifar10")



def read_given_dataset(data_path):
	train_transform = transforms.Compose([])

	train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
	train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
	train_set = folder.TensorDataset(train_data, train_labels, transform=train_transform)

	X, Y = [], []
	for i in range(len(train_set)):
		X.append(train_set[i][0])
		Y.append(train_set[i][1].numpy())
	return (X, Y)


def scaled_values(val, mean, std):
	return (val - np.repeat(np.expand_dims(mean, 1), val.shape[1], axis=1)) / (np.expand_dims(std, 1) +  1e-10)


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


def get_sensitivities(path):
	features = []
	with open(path, 'r') as f:
		for line in tqdm(f):
			values = np.array([float(x) for x in line.rstrip('\n').split(',')])
			features.append(values)
	return np.array(features)


def best_target_image(mat, which=0):
	sum_m = []
	for i in range(mat.shape[1]):
		mat_interest = mat[mat[:, i] != np.inf, i]
		sum_m.append(np.average(np.abs(mat_interest)))
	best = np.argsort(sum_m)
	return best[which]


def get_statistics(diff):
	l1_norms   = ch.sum(ch.abs(diff), dim=1)
	l2_norms   = ch.norm(diff, dim=1)
	linf_norms = ch.max(ch.abs(diff), dim=1)[0]
	return (l1_norms, l2_norms, linf_norms)


def get_stats(base_path):
	mean = np.load(os.path.join(base_path, "feature_mean.npy"))
	std  = np.load(os.path.join(base_path, "feature_std.npy"))
	return mean, std
