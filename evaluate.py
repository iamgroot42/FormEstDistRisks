from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

import keras
import tensorflow as tf
import numpy as np
import argparse

import models, common, datasets, attacks, helpers


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-m','--model', type=str, default="./models/normally_trained_final.h5", metavar='STRING', help='path to model')
args = parser.parse_args()


if __name__ == "__main__":
	common.conserve_gpu_memory()
	model = keras.models.load_model(args.model)
	dataset = datasets.CIFAR10()

	sess = keras.backend.get_session()
	wrap = KerasModelWrapper(model)
	attack = attacks.MadryEtAl(dataset, wrap, sess)

	weak_params   = {"eps": 0.25, "eps_iter": 0.05, "nb_iter": 2500}
	strong_params = {"eps": 0.5,  "eps_iter": 0.1,  "nb_iter": 2500}
	acc, adv_acc = helpers.benchmarking(model, dataset, attack, bs=args.batch_size, cp=weak_params)
	print(">> Test accuracy on original data : %f" % acc)
	print(">> Test accuracy on adversarial data : %f" % adv_acc)
