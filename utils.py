import torch as ch
import numpy as np
import torch.nn as nn
from torchvision import transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR, CelebA
from robustness.tools import folder
from robustness.tools.misc import log_statement
from facenet_pytorch import InceptionResnetV1

from tqdm import tqdm
import requests
import pandas as pd
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

	def get_model(self, m_type, arch='resnet50', parallel=False):
		model_path = self.models.get(m_type, None)
		if not model_path:
			model_path = m_type
		else:
			model_path = self.model_prefix[arch] + self.models[m_type]
		model_kwargs = {
			'arch': arch,
			'dataset': self.dataset,
			'resume_path': model_path,
			'parallel': parallel
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
	def __init__(self, path):
		self.dataset_type = GenericBinary
		super(BinaryCIFAR, self).__init__('binary_cifar10', path, None)


class CIFAR10(DataPaths):
	def __init__(self, data_path=None):
		self.dataset_type = CIFAR
		datapath = "/p/adversarialml/as9rw/datasets/cifar10" if data_path is None else data_path
		# print(datapath, "wtf?!")
		# exit(0)
		super(CIFAR10, self).__init__('cifar10',
			datapath,
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
	def __init__(self, data_path=None):
		self.dataset_type = ImageNet
		datapath = "/p/adversarialml/as9rw/datasets/imagenet/" if data_path is None else data_path
		super(ImageNet1000, self).__init__('imagenet1000',
			datapath,
			"/p/adversarialml/as9rw/imagenet_stats/")
		self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_imagenet/"
		self.models['nat']  = "imagenet_nat.pt"
		self.models['l2']   = "imagenet_l2_3_0.pt"
		self.models['linf'] = "imagenet_linf_4.pt"


class Celeb(DataPaths):
	def __init__(self, data_path=None):
		self.dataset_type = CelebA
		datapath = "/p/adversarialml/as9rw/datasets/celeba/" if data_path is None else data_path
		super(Celeb, self).__init__('celeb',
			datapath,
			"/p/adversarialml/as9rw/celeba_stats/")
		# self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_celeba/"


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
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
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


def classwise_pixelwise_stats(loader, num_classes=10, classwise=False):
	images, labels = load_all_loader_data(loader)
	if not classwise:
		return ch.mean(images, 0), ch.std(images, 0)
	means, stds = [], []
	for i in range(num_classes):
		specific_images = images[labels == i]
		means.append(ch.mean(specific_images, 0))
		stds.append(ch.std(specific_images, 0))
	return means, stds


class Decoder(nn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		# Input size: [batch, n_features]
		# Output size: [batch, 3, 32, 32]
		# Expects 48, 4, 4
		self.dnn = nn.Sequential(
			nn.Linear(512, 768),
			nn.BatchNorm1d(768),
			nn.ReLU())
		self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
			nn.BatchNorm2d(24),
			nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
			nn.BatchNorm2d(12),
			nn.ReLU(),
			nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
			nn.Sigmoid(),
		)

	def forward(self, x):
		x_ = self.dnn(x)
		x_ = x_.view(x_.shape[0], 48, 4, 4)
		return self.decoder(x_)


class BasicDataset(ch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.X, self.Y = X, Y

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, index):
		return self.X[index], self.Y[index]


def compute_delta_values(logits, weights, actual_label=None):
	# Iterate through all possible classes, calculate flip probabilities
	actual_label = ch.argmax(logits)
	numerator = (logits[actual_label] - logits).unsqueeze(1)
	denominator = weights - weights[actual_label]
	numerator = numerator.repeat(1, denominator.shape[1])
	delta_values = ch.div(numerator, denominator)
	delta_values[actual_label] = np.inf
	return delta_values


def get_these_params(model, identifier):
	for name, param in model.state_dict().items():
		if name == identifier:
			return param
	return None


def flash_utils(args):
	log_statement("==> Arguments:")
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))


# US Income dataset
class CensusIncome:
	def __init__(self, path):
		self.urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
		"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
		"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
		self.columns = ["age", "workClass", "fnlwgt", "education", "education-num",
			"marital-status", "occupation", "relationship",
			"race", "sex", "capital-gain", "capital-loss",
			"hours-per-week", "native-country", "income"]
		self.path = path
		self.download_dataset()

	def download_dataset(self):
		if not os.path.exists(self.path):
			log_statement("==> Downloading US Census Income dataset")
			os.mkdir(self.path)
			log_statement("==> Please modify test file to remove stray dot characters")

			for url in self.urls:
				data = requests.get(url).content
				filename = os.path.join(self.path, os.path.basename(url))
				with open(filename, "wb") as file:
					file.write(data)

	def process_df(self, df):
		df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

		def oneHotCatVars(x, colname):
			df_1 = df.drop(columns = colname, axis = 1)
			df_2 = pd.get_dummies(df[colname], prefix=colname, prefix_sep=':')
			return (pd.concat([df_1, df_2], axis=1, join='inner'))

		# colnames = ['workClass', 'education', 'occupation', 'race', 'sex', 'marital-status', 'relationship', 'native-country']
		colnames = ['workClass', 'occupation', 'race', 'sex', 'marital-status', 'relationship', 'native-country']
		# Drop education
		df = df.drop(columns = 'education', axis=1)
		for colname in colnames: df = oneHotCatVars(df, colname)
		return df


	def load_data(self, train_filter=None):
		train_data = pd.read_csv(os.path.join(self.path, 'adult.data'), names=self.columns,
			sep=' *, *', na_values='?', engine='python')
		test_data  = pd.read_csv(os.path.join(self.path, 'adult.test'), names=self.columns,
			sep=' *, *', skiprows=1, na_values='?', engine='python')

		# print(np.mean((test_data['income']=='<=50K').to_numpy()))
		# print(np.mean((train_data['income']=='<=50K').to_numpy()))

		# Add field to identify train/test, process together, split back
		train_data['is_train'] = 1
		test_data['is_train']  = 0
		df = pd.concat([train_data, test_data], axis=0)
		# print(df.info())
		df = self.process_df(df)

		train_df, test_df = df[df['is_train'] == 1], df[df['is_train'] == 0]

		# Apply filter to train data: efectively using different distribution to train it
		if train_filter is not None: train_df = train_filter(train_df)
		
		def get_x_y(P):
			Y = P['income'].to_numpy()
			X = P.drop(columns = 'income', axis = 1)
			cols = X.columns
			X = X.to_numpy()
			return (X.astype(float), np.expand_dims(Y, 1), cols)

		return get_x_y(train_df), get_x_y(test_df)


# Classifier on top of face features
class FaceModel(nn.Module):
	def __init__(self, n_feat):
		super(FaceModel, self).__init__()
		# self.feature_model = InceptionResnetV1(pretrained='vggface2').eval()
		self.feature_model = InceptionResnetV1(pretrained='casia-webface').eval()
		for param in self.feature_model.parameters(): param.requires_grad = False
		self.dnn = nn.Sequential(
			nn.Linear(n_feat, 64),
			nn.ReLU(),
			nn.Linear(64, 16),
			nn.ReLU(),
			nn.Linear(16, 1),
			nn.Sigmoid())

	def forward(self, x):
		# with ch.no_grad():
		x_ = self.feature_model(x)
		return self.dnn(x_)


class MNISTFlatModel(nn.Module):
	def __init__(self):
		super(MNISTFlatModel, self).__init__()
		self.dnn = nn.Sequential(
			nn.Linear(n_feat, 128),
			nn.ReLU(),
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Linear(32, 16),
			nn.ReLU(),
			nn.Linear(16, 10))

	def forward(self, x):
		x_ = x.view(x.shape[0], -1)
		return self.forward(x_)


def filter(df, condition, ratio):
	qi    = np.nonzero((condition(df)).to_numpy())[0]
	notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
	np.random.shuffle(notqualify)
	nqi = notqualify[:int(((1-ratio) * len(qi))/ratio)]
	return pd.concat([df.iloc[qi], df.iloc[nqi]])
