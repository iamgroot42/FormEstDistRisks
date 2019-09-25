from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

import keras

import tensorflow as tf
import numpy as np
import argparse
import sys

import models, common, datasets


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-e','--nb_epochs', type=int, default=200, metavar='NUMBER', help='epochs(default: 200)')
parser.add_argument('-g','--save_here', type=str, default="./models/adversarialy_trained", metavar='STRING', help='path where trained model should be saved')
parser.add_argument('-a','--augment', type=bool, default=False, metavar='BOOLEAN', help='use augmentation while training data')
args = parser.parse_args()


def get_attack(wrap):
	attack = MadryEtAl
	attack_params = {'clip_min': 0, 'clip_max': 1}
	attack_object = attack(wrap, sess=session)
	attack_params['nb_iter'] = 7
	attack_params['eps'] = 5e-1
	attack_params['eps_iter'] = 5e-1 / 5
	return attack_object, attack_params


def train_model(dataset, attack_object, batch_size, nb_epochs, augment, save_path):
	session = keras.backend.get_session()
	init = tf.global_variables_initializer()
	session.run(init)

	model, cbks = models.ResNet50(input_shape=dataset.sample_shape, classes=dataset.classes)
	wrap = KerasModelWrapper(model)
	attack_object, attack_params = get_attack(wrap, session)

	(X_train, Y_train), (X_val, Y_val) = dataset.get_data()
	batch_size //= 2 # Adversarial data accounts for half of batch
	if augment:
		batch_size //= 2
		augmentor = dataset.get_augmentations() 

	for i in range(nb_epochs):
		batch_no = 1
		train_loss, train_acc = 0, 0
		for j in range(0, len(X_train), batch_size):
			x_clean, y_clean = X_train[j:j+batch_size], Y_train[j:j+batch_size]
			if augment:
				x_aug = augmentor.augment_images(x_clean)
				x_clean_use = np.concatenate([x_clean, x_aug], axis=0)
				y_clean_use = np.concatenate([y_clean, y_clean], axis=0)
			else:
				x_clean_use, y_clean_use = x_clean, y_clean
			x_adv = attack_object.generate_np(x_clean_use, **attack_params)
			x_batch = np.concatenate([x_clean_use, x_adv], axis=0)
			y_batch = np.concatenate([y_clean_use, y_clean_use], axis=0)
			train_metrics = model.train_on_batch(x_batch, y_batch)
			train_loss += train_metrics[0]
			train_acc += train_metrics[1]
			sys.stdout.write("Epoch %d: %d / %d : Tr loss: %f, Tr acc: %f  \r" % (i+1, batch_no, 1 + len(X_train)//batch_size, train_loss/(batch_no), train_acc/(batch_no)))
			sys.stdout.flush()
			batch_no += 1
		val_metrics = model.evaluate(X_val, Y_val, batch_size=1024, verbose=0)
		print()
		print(">> Val loss: %f, Val acc: %f"% (val_metrics[0], val_metrics[1]))
		if i % 10 == 9:
			model.save(save_path + "_%d.h5" % (i + 1))

	model.save(save_path +  "_final.h5")


if __name__ == "__main__":
	common.conserve_gpu_memory()
	(X_train, Y_train), (X_val, Y_val) = datasets.CIFAR10()
	train_model(dataset, args.batch_size, args.nb_epochs, args.augment, args.save_here)
