from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

import keras

import tensorflow as tf
import numpy as np

import models, common, datasets


def get_attack(wrap, session):
	attack = MadryEtAl
	attack_params = {'clip_min': 0, 'clip_max': 1}
	attack_object = attack(wrap, sess=session)
	attack_params['nb_iter'] = 1e1
	attack_params['eps'] = 5e-1
	# attack_params['eps_iter'] = 5e-1 / 5
	return attack_object, attack_params


def batch_attack(attack_X, attack, attack_params, batch_size):
	perturbed_X = np.array([])
	for i in range(0, attack_X.shape[0], batch_size):
		mini_batch = attack_X[i: i + batch_size,:]
		if mini_batch.shape[0] == 0:
			break
		adv_x_mini = attack.generate_np(mini_batch, **attack_params)
		if perturbed_X.shape[0] != 0:
			perturbed_X = np.append(perturbed_X, adv_x_mini, axis=0)
		else:
			perturbed_X = adv_x_mini
	return perturbed_X


def benchmarking(model, og_dataset, robust_dataset):
	_, (X_og_val, Y_og_val) = og_dataset.get_data()
	_, (X_ro_val, Y_ro_val) = robust_dataset.get_data()
	og_acc = model.evaluate(X_og_val, Y_og_val, batch_size=1024, verbose=0)[1]
	ro_acc = model.evaluate(X_ro_val, Y_ro_val, batch_size=1024, verbose=0)[1]
	print(">> Test accuracy on original data : %f" % og_acc)
	print(">> Test accuracy on robust data : %f" % ro_acc)

	session = keras.backend.get_session()
	wrap = KerasModelWrapper(model)
	attack_object, attack_params = get_attack(wrap, session)

	X_og_adv_val = batch_attack(X_og_val, attack_object, attack_params, 512)
	X_ro_adv_val = batch_attack(X_ro_val, attack_object, attack_params, 512)
	og_adv_acc = model.evaluate(X_og_adv_val, Y_og_val, batch_size=1024, verbose=0)[1]
	ro_adv_acc = model.evaluate(X_ro_adv_val, Y_ro_val, batch_size=1024, verbose=0)[1]
	print(">> Test accuracy on adversarial original data : %f" % og_adv_acc)
	print(">> Test accuracy on adversarial robust data : %f" % ro_adv_acc)


if __name__ == "__main__":
	common.conserve_gpu_memory()
	# model = keras.models.load_model("./models/normally_trained_noaug_final.h5")
	# model = keras.models.load_model("./models/normally_trained_final.h5")
	model = keras.models.load_model("./models/adversarialy_trained_final.h5")
	cifar = datasets.CIFAR10()
	robust_cifar = datasets.RobustCIFAR10()
	benchmarking(model, cifar, robust_cifar)
