from sklearn.utils import shuffle

import keras
from keras.datasets import cifar10

import tensorflow as tf
import numpy as np
import argparse
import sys

import models, common, datasets
from construct_robust_dataset import load_robust_cifar


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-e','--nb_epochs', type=int, default=200, metavar='NUMBER', help='epochs(default: 200)')
parser.add_argument('-g','--save_here', type=str, default="./models/normally_trained", metavar='STRING', help='path where trained model should be saved')
parser.add_argument('-a','--augment', type=bool, default=False, metavar='BOOLEAN', help='use augmentation while training data')
parser.add_argument('-r','--robust_data', type=bool, default=True, metavar='BOOLEAN', help='use robust data?')
args = parser.parse_args()


def train_model(batch_size, nb_epochs, augment, save_path):
	model, scheduler = models.ResNet50(input_shape=dataset.sample_shape, classes=dataset.classes)

	session = keras.backend.get_session()
	init = tf.global_variables_initializer()
	session.run(init)

	(X_train, Y_train), (X_val, Y_val) = dataset.get_data()
	if augment:
		batch_size //= 2
		augmentor = dataset.get_augmentations()

	for i in range(nb_epochs):
		batch_no = 1
		train_loss, train_acc = 0, 0
		if scheduler:
			keras.backend.set_value(model.optimizer.lr, scheduler(i))
		for j in range(0, len(X_train), batch_size):
			x_clean, y_clean = X_train[j:j+batch_size], Y_train[j:j+batch_size]
			if augment:
				x_aug   = augmentor.augment_images(x_clean)
				x_batch = np.concatenate([x_clean, x_aug], axis=0)
				y_batch = np.concatenate([y_clean, y_clean], axis=0)
			else:
				x_batch, y_batch = x_clean, y_clean
			train_metrics = model.train_on_batch(x_batch, y_batch)
			train_loss   += train_metrics[0]
			train_acc    += train_metrics[1]
			sys.stdout.write("Epoch %d: %d / %d : Tr loss: %f, Tr acc: %f  \r" % (i+1, batch_no, 1 + len(X_train)//batch_size, train_loss/(batch_no), train_acc/(batch_no)))
			sys.stdout.flush()
			batch_no += 1
		val_metrics = model.evaluate(X_val, Y_val, batch_size=1024, verbose=0)
		print()
		print(">> Val loss: %f, Val acc: %f\n"% (val_metrics[0], val_metrics[1]))
		if i % 10 == 9:
			model.save(save_path + "_%d.h5" % (i + 1))

	model.save(save_path +  "_final.h5")


if __name__ == "__main__":
	common.conserve_gpu_memory()
	if args.robust_data:
		dataset = datasets.RobustCIFAR10()
	else:
		dataset = datasets.CIFAR10()
	train_model(dataset, args.batch_size, args.nb_epochs, args.augment, args.save_here)
