import torch as ch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR
from robustness.tools import folder
from robustness.tools.misc import log_statement
from tqdm import tqdm
import sys
import os


class DataPaths:
	def __init__(self, name, data_path, stats_path):
		self.name      = name
		self.data_path = data_path
		self.dataset   = self.dataset_type(data_path)
		self.models = {'nat': None, 'l1': None, 'l2': None, 'temp': None, 'linf': None}
		self.model_prefix = {}
		self.stats_path = stats_path

	def get_dataset(self):
		return self.dataset

	def get_model(self, m_type, arch='resnet50'):
		model_path = self.models.get(m_type, None)
		if not model_path:
			model_path = m_type
		else:
			model_path = self.model_prefix[arch] + self.models[m_type]
		model_kwargs = {
			'arch': arch,
			'dataset': self.dataset,
			'resume_path': model_path
		}
		model, _ = make_and_restore_model(**model_kwargs)
		model.eval()
		return model

	def get_stats(self, m_type, arch='resnet50'):
		stats_path = os.path.join(self.stats_path, arch, m_type, "stats")
		return get_stats(stats_path)

	def get_deltas(self, m_type, arch='resnet50', numpy=False):
		ext = ".npy" if numpy else ".txt"
		deltas_path = os.path.join(self.stats_path, arch, m_type, "deltas" + ext)
		return get_sensitivities(deltas_path, numpy=numpy)


class BinaryCIFAR(DataPaths):
	def __init__(self):
		self.dataset_type = GenericBinary
		super(BinaryCIFAR, self).__init__('binary cifar10',
			"/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct",
			"/p/adversarialml/as9rw/binary_stats/")
		self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_correct/"
		self.models['nat']  = "normal/checkpoint.pt.latest"
		self.models['linf'] = "linf/checkpoint.pt.latest"
		self.models['l1']   = "l1/checkpoint.pt.latest"
		self.models['l2']   = "l2/checkpoint.pt.latest"


class CIFAR10(DataPaths):
	def __init__(self):
		self.dataset_type = CIFAR
		super(CIFAR10, self).__init__('cifar10',
			"/p/adversarialml/as9rw/datasets/cifar10",
			"/p/adversarialml/as9rw/cifar10_stats/")
		self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_cifar10/"
		self.model_prefix['densenet169'] = "/p/adversarialml/as9rw/models_cifar10_densenet/"
		self.model_prefix['vgg19'] = "/p/adversarialml/as9rw/models_cifar10_vgg/"
		self.models['nat']  = "cifar_nat.pt"
		self.models['linf'] = "cifar_linf_8.pt"
		self.models['l2']   = "cifar_l2_0_5.pt"


class RobustCIFAR10(DataPaths):
	def __init__(self, datapath, stats_prefix):
		self.dataset_type = RobustCIFAR
		super(RobustCIFAR10, self).__init__('robustcifar10',
			datapath, stats_prefix)


class SVHN10(DataPaths):
	def __init__(self):
		self.dataset_type = SVHN
		super(SVHN10, self).__init__('svhn',
			"/p/adversarialml/as9rw/datasets/svhn",
			"/p/adversarialml/as9rw/svhn_stats/")
		self.model_prefix['vgg16'] = "/p/adversarialml/as9rw/models_svhn_vgg/"
		self.models['nat']  = "svhn_nat.pt"
		self.models['linf'] = "svhn_linf_4.pt"
		self.models['l2']   = "svhn_l2_0_5.pt"


class ImageNet1000(DataPaths):
	def __init__(self):
		self.dataset_type = ImageNet
		super(ImageNet1000, self).__init__('imagenet1000',
			"/p/adversarialml/as9rw/datasets/imagenet/",
			"/p/adversarialml/as9rw/imagenet_stats/")
		self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_imagenet/"
		self.models['nat']  = "imagenet_nat.pt"
		self.models['l2']   = "imagenet_l2_3_0.pt"
		self.models['linf'] = "imagenet_linf_4.pt"


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


def scaled_values(val, mean, std, eps=1e-10):
	return (val - np.repeat(np.expand_dims(mean, 1), val.shape[1], axis=1)) / (np.expand_dims(std, 1) +  eps)


def load_all_loader_data(data_loader):
	images, labels = [], []
	for (image, label) in data_loader:
		images.append(image)
		labels.append(label)
	images = ch.cat(images)
	labels = ch.cat(labels)
	return (images, labels)


def load_all_data(ds):
	batch_size = 512
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)
	return load_all_loader_data(test_loader)


def get_sensitivities(path, numpy=False):
	log_statement("==> Loading Delta Values")
	# Directly load, if numpy array
	if numpy:
		return np.load(path)
	# Process, if text file
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


def get_logits_layer_name(arch):
	if "vgg" in arch:
		return "module.model.classifier.weight"
	elif "resnet" in arch:
		return "module.model.fc.weight"
	elif "densenet" in arch:
		return "module.model.linear.weight"
	return None


class SpecificLayerModel(ch.nn.Module):
	def __init__(self, model, layer_index):
		super(SpecificLayerModel, self).__init__()
		self.model = model
		self.layer_index = layer_index

	def forward(self, x):
		logits, _ = self.model(x, this_layer_input=self.layer_index)
		return logits


class MadryToNormal:
	def __init__(self, model, fake_relu=False):
		self.model = model
		self.fake_relu = fake_relu
		self.model.eval()

	def __call__(self, x):
		logits, _ = self.model(x, fake_relu=self.fake_relu)
		return logits

	def eval(self):
		return self.model.eval()

	def parameters(self):
		return self.model.parameters()

	def named_parameters(self):
		return self.model.named_parameters()


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		# Input size: [batch, n_features]
		# Output size: [batch, 3, 32, 32]
		# Expects 48, 4, 4
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
			nn.ReLU(),
			# nn.BatchNorm2d(24),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
			nn.ReLU(),
			# nn.BatchNorm2d(12),
			nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
			nn.Sigmoid(),
		)

	def forward(self, x):
		x_ = x.view(x.shape[0], 32, 4, 4)
		# for l in self.decoder:
			# x_ = l(x_)
			# print(x_.shape)
		# exit(0)
		# return x_
		return self.decoder(x_)


class BasicDataset(ch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.X, self.Y = X, Y

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, index):
		return self.X[index], self.Y[index]
