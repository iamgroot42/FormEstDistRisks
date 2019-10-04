from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

import keras

import tensorflow as tf
import numpy as np
import argparse
import sys

from sklearn.utils import shuffle
import models, common, datasets, attacks, helpers


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=32, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-e','--nb_epochs', type=int, default=200, metavar='NUMBER', help='epochs(default: 200)')
parser.add_argument('-g','--save_here', type=str, default="./models/adversarialy_trained", metavar='STRING', help='path where trained model should be saved')
parser.add_argument('-a','--augment', type=bool, default=False, metavar='BOOLEAN', help='use augmentation while training data')
parser.add_argument('-d','--datasets', type=str, default="", metavar='STRING', help='paths to folder containing datasets')
parser.add_argument('-r','--ratios', type=str, default="", metavar='STRING', help='comma separated list of ratios to be used to sample data while training')
args = parser.parse_args()


def train_model(dataset, batch_size, nb_epochs, augment, save_path):
	model, scheduler = models.ResNet50(input_shape=dataset.sample_shape, classes=dataset.classes)

	session = keras.backend.get_session()
	init = tf.global_variables_initializer()
	session.run(init)

	wrap = KerasModelWrapper(model)
	sess = keras.backend.get_session()
	attack = attacks.MadryEtAl(dataset, wrap, sess)

	(X_train, Y_train), (X_val, Y_val) = dataset.get_data()
	if augment:
		print(">> Using data augmentation")
		augmentor = dataset.get_augmentations()

	for i in range(nb_epochs):
		batch_no = 1
		train_loss, train_acc = 0, 0
		adv_loss, adv_acc = 0, 0
		if scheduler:
			keras.backend.set_value(model.optimizer.lr, scheduler(i))
		# Shuffle data
		X_train, Y_train = dataset.shuffle(X_train, Y_train)
		for j in range(0, len(X_train), batch_size):
			x_clean, y_clean = X_train[j:j+batch_size], Y_train[j:j+batch_size]
			if augment:
				x_aug = augmentor.augment_images(dataset.un_normalize(x_clean))
				x_aug = dataset.normalize(np.array(x_aug))
				x_clean_use = np.concatenate([x_clean, x_aug], axis=0)
				y_clean_use = np.concatenate([y_clean, y_clean], axis=0)
			else:
				x_clean_use, y_clean_use = x_clean, y_clean
			x_adv   = attack.attack_data(x_clean_use)
			x_batch = np.concatenate([x_clean_use, x_adv], axis=0)
			y_batch = np.concatenate([y_clean_use, y_clean_use], axis=0)

			# helpers.save_image(x_clean_use[0], "./clean.png")
			# helpers.save_image(x_adv[0],         "./adv.png")
			# exit()

			# Train on batch
			model.train_on_batch(x_batch, y_batch)
			# Evauate metrics on perturbed data
			adv_metrics   = model.evaluate(x_adv, y_clean_use, verbose=0, batch_size=batch_size)
			adv_loss     += adv_metrics[0]
			adv_acc      += adv_metrics[1]
			# Evauate metrics on clean data
			train_metrics   = model.evaluate(x_clean_use, y_clean_use, verbose=0, batch_size=batch_size)
			train_loss     += train_metrics[0]
			train_acc      += train_metrics[1]
			sys.stdout.write("Epoch %d: %d / %d : Clean loss: %f, Clean acc: %f, Adv loss: %f, Adv acc: %f  \r" % (i+1, batch_no, 1 + len(X_train)//batch_size, train_loss/(batch_no), train_acc/(batch_no), adv_loss/(batch_no), adv_acc/(batch_no)))
			sys.stdout.flush()
			batch_no += 1
		val_metrics = model.evaluate(X_val, Y_val, batch_size=batch_size, verbose=0)
		print()
		print(">> Val loss: %f, Val acc: %f\n"% (val_metrics[0], val_metrics[1]))
		if i % 10 == 9:
			model.save(save_path + "_%d.h5" % (i + 1))

	model.save(save_path +  "_final.h5")


if __name__ == "__main__":
	import os
	common.conserve_gpu_memory()
	ds = [datasets.CIFAR10()]
	for p in os.listdir(args.datasets):
		full_path = os.path.join(args.datasets, p)
		print(">> Loaded data from %s" % full_path)
		ds.append(datasets.RobustCIFAR10(path=full_path))

	if len(args.ratios) == 0:
		sample_ratios = None
		print("Using all samples from all datasets")
	else:
		sample_ratios = [float(x) for x in args.ratios.split(',')]
		print("Using ratios : ", sample_ratios)

	effective_dataset = datasets.CombinedDatasets(ds, sample_ratios=sample_ratios)
	train_model(effective_dataset, args.batch_size, args.nb_epochs, args.augment, args.save_here)
