from keras.datasets import cifar10
import imgaug.augmenters as iaa
import keras

import numpy as np


class Dataset:
	def __init__(self, classes, shape, shuffle_data):
		self.classes = classes
		self.shuffle_data = shuffle_data
		self.sample_shape = shape
		(self.X_train, self.Y_train), (self.X_val, self.Y_val) = (None, None), (None, None)
		self.ready_data()

	def load_data(self):
		return

	def ready_data(self):
		self.load_data()
		return

	def get_data(self):
		return (self.X_train, self.Y_train), (self.X_val, self.Y_val)

	def normalize(self, X):
		return X

	def un_normalize(self, X):
		return X

	def get_augmentations(self):
		return None

	def shuffle(self, X, Y):
		assert len(X) == len(Y)
		indices = np.random.permutation(len(X))
		X, Y = X[indices], Y[indices]
		return X, Y

	def greatest_common_class(self, instances):
		classes = [type(x).mro() for x in instances]
		for x in classes[0]:
			if all(x in mro for mro in classes):
				return x


class CIFAR10(Dataset):
	def __init__(self, shuffle_data=True):
		self.name = "CIFAR10"
		super().__init__(classes=10, shape=(32, 32, 3), shuffle_data=shuffle_data)

	def load_data(self):
		(self.X_train, self.Y_train), (self.X_val, self.Y_val) = cifar10.load_data()

	def ready_data(self):
		self.load_data()
		self.X_train = self.normalize(self.X_train)
		self.X_val   = self.normalize(self.X_val)
		self.Y_train = keras.utils.to_categorical(self.Y_train, self.classes)
		self.Y_val   = keras.utils.to_categorical(self.Y_val,   self.classes)
		if self.shuffle_data:
			self.X_train, self.Y_train = self.shuffle(self.X_train, self.Y_train)

	def normalize(self, X):
		return X.astype('float32') / 255

	def un_normalize(self, X):
		return (X * 255).astype('uint8')

	def get_augmentations(self):
		seq = iaa.Sequential([
			iaa.Pad(4, keep_size=False),
			iaa.CropToFixedSize(32, 32),
			iaa.Fliplr(0.1),
			iaa.Affine(rotate=(-2, 2))
		])
		return seq

	
class RobustCIFAR10(CIFAR10):
	def __init__(self, path="./datasets/robust_cifar_data.npz", shuffle_data=True):
		self.path = path
		self.name = "RobustCIFAR10"
		super().__init__(shuffle_data=shuffle_data)

	def load_data(self):
		data = np.load(self.path, allow_pickle=True)

		def unroll(y):
			return np.concatenate(y, axis=0)

		self.X_train, self.X_val = unroll(data['X_train']), unroll(data['X_val'])
		self.Y_train, self.Y_val = data['Y_train'], data['Y_val']

	def ready_data(self):
		self.load_data()
		if self.shuffle_data:
			self.X_train, self.Y_train = self.shuffle(self.X_train, self.Y_train)


class CombinedDatasets(Dataset):
	def __init__(self, data_sets, shuffle_data=True, sample_ratios=None):
		if len(data_sets) < 2:
			raise ValueError("Only one dataset passed. Use this wrapper class for >=2 datasets")
		if self.greatest_common_class(data_sets) != type(data_sets[0]):
			raise TypeError("All datasets should be children/copies of the first dataset (variations)")
		if sample_ratios:
			assert len(sample_ratios) == len(data_sets), "Sample ratios should be given for all datasets, if default not used"
			if max(sample_ratios) > 1 or min(sample_ratios) < 0:
				raise ValueError("Sample ratios must be valid.")
		self.sample_ratios = sample_ratios
		self.name = data_sets[0].name
		self.data_sets = data_sets
		super().__init__(self.data_sets[0].classes, self.data_sets[0].sample_shape, shuffle_data)

	def sample_data(self, X, Y, sample_ratio):
		indices = np.random.permutation(len(X))[:int(sample_ratio * len(X))]
		return X[indices], Y[indices]

	def load_data(self):
		self.X_train, self.Y_train = [], []
		for i, dataset in enumerate(self.data_sets):
			(X, Y), _ = dataset.get_data()
			if self.sample_ratios:
				A, B = self.sample_data(X, Y, self.sample_ratios[i])
			else:
				A, B = X, Y
			self.X_train.append(A)
			self.Y_train.append(B)

		self.X_train = np.concatenate(self.X_train, axis=0)
		self.Y_train = np.concatenate(self.Y_train, axis=0)
		_, (self.X_val, self.Y_val) = self.data_sets[0].get_data()


# Update every time you add a new dataset
dataset_list = [CIFAR10, RobustCIFAR10]
