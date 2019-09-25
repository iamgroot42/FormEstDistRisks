from sklearn.utils import shuffle

from keras.datasets import cifar10
import imgaug.augmenters as iaa
import keras

import numpy as np


class Dataset:
	def __init__(self, classes, shape):
		self.classes = classes
		self.sample_shape = shape
		(self.X_train, self.Y_train), (self.X_val, self.Y_val) = (None, None), (None, None)

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


class CIFAR10(Dataset):
	def __init__(self):
		super(CIFAR10).__init__(classes=10, shape=(32, 32, 3))

	def load_data(self):
		(self.X_train, self.Y_train), (self.X_val, self.Y_val) = cifar10.load_data()

	def ready_data(self, shuffle=True):
		self.ready_data()
		self.X_train = self.normalize(self.X_train.astype('float32'))
		self.X_val   = self.normalize(self.X_val.astype('float32'))
		self.Y_train = keras.utils.to_categorical(self.Y_train, self.classes)
		self.Y_train = keras.utils.to_categorical(self.Y_train, self.classes)
		if shuffle:
			self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train)

	def normalize(self, X):
		return X / 255

	def un_normalize
		return X * 255

	def get_augmentations(self):
		seq = iaa.Sequential([
		iaa.Pad(4), # Pad all sides by 4
			iaa.CropToFixedSize(32, 32), # Crop to fixed size
			iaa.Fliplr(0.1), # horizontally flip 10% of the images
			iaa.Affine(rotate=(-2, 2)) # Rotate by 2 degrees
		])
		return seq

	
class RobustCIFAR10(CIFAR10):
	def __init__(self):
		super(RobustCIFAR10).__init__()

	def load_data(self, path):
		data = np.load(path, allow_pickle=True)

		def unroll(y):
			return np.concatenate(y, axis=0)

		self.X_train, self.X_val = unroll(data['X_train']), unroll(data['X_val'])
		(_, self.Y_train), (_, self.Y_val) = cifar10.load_data()

	def ready_data(self, path="./datasets/robust_cifar_data.npz", shuffle=True):
		self.path = path
		self.ready_data()
		self.Y_train = keras.utils.to_categorical(self.Y_train, self.classes)
		self.Y_train = keras.utils.to_categorical(self.Y_train, self.classes)
		if shuffle:
			self.X_train, self.Y_train = shuffle(self.X_train, self.Y_train)
