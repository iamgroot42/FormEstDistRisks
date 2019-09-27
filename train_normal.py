import keras
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint, LearningRateScheduler

import tensorflow as tf
import numpy as np
import argparse
import sys

import models, common, datasets


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-e','--nb_epochs', type=int, default=150, metavar='NUMBER', help='epochs(default: 200)')
parser.add_argument('-g','--save_here', type=str, default="./models/normally_trained", metavar='STRING', help='path where trained model should be saved')
parser.add_argument('-a','--augment', type=bool, default=False, metavar='BOOLEAN', help='use augmentation while training data')
parser.add_argument('-r','--robust_data', type=bool, default=False, metavar='BOOLEAN', help='use robust data?')
args = parser.parse_args()


def generator(X, Y, dataset, batch_size, aug):
	while True:
		for i in range(0, len(X), batch_size):
			x_clean, y_clean = X[i:i+batch_size], Y[i:i+batch_size]
			if aug is not None:
				x_aug = aug.augment_images(dataset.un_normalize(x_clean))
				x_aug = dataset.normalize(np.array(x_aug))
				x_clean_use = np.concatenate([x_clean, x_aug], axis=0)
				y_clean_use = np.concatenate([y_clean, y_clean], axis=0)
			else:
				x_clean_use, y_clean_use = x_clean, y_clean
			yield (x_clean_use, y_clean_use)


def train_model(dataset, batch_size, nb_epochs, augment, save_path):
	model, scheduler = models.ResNet50(input_shape=dataset.sample_shape, classes=dataset.classes)
	# Load data
	(X_train, Y_train), (X_val, Y_val) = dataset.get_data()

	def scheduler_wrap(epoch, current_lr):
		return scheduler(epoch)

	augmentor = None
	if augment:
		print(">> Using data augmentation")
		augmentor = dataset.get_augmentations()
		batch_size //= 2
	# Get image generator (with data augmentation, if flag enabled)
	gen = generator(X_train, Y_train, dataset, batch_size, augmentor)
	# Train model
	model.fit_generator(gen,
		steps_per_epoch=len(X_train) // batch_size,
		epochs=nb_epochs,
		callbacks=[ModelCheckpoint(filepath=save_path + "_{epoch:02d}-{val_loss:.2f}.hd5", period=10), 
			LearningRateScheduler(scheduler_wrap)],
		validation_data=(X_val, Y_val))

	model.save(save_path +  "_final.h5")


if __name__ == "__main__":
	common.conserve_gpu_memory()
	if args.robust_data:
		dataset = datasets.RobustCIFAR10()
	else:
		dataset = datasets.CIFAR10()
	train_model(dataset, args.batch_size, args.nb_epochs, args.augment, args.save_here)
