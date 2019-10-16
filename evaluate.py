from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

import keras
import tensorflow as tf
import numpy as np
import argparse

import models, common, datasets, attacks, helpers


parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size(default: 128)')
parser.add_argument('-a','--attack_type', type=str, default="l2", metavar='STRING', help='type of attack (l1/l2/linf/spatial)')
parser.add_argument('-m','--model', type=str, default="./models/normally_trained_final.h5", metavar='STRING', help='path to model')
args = parser.parse_args()


def map_norm_to_attack(norm):
	mapping = {
		"l1": attacks.SparseL1Descent,
		"l2": attacks.MadryEtAl,
		"linf": attacks.MadryEtAl_Inf,
		"spatial": attacks.SpatialTransformation
	}
	return mapping[norm]


if __name__ == "__main__":
	common.conserve_gpu_memory()
	model = keras.models.load_model(args.model)
	dataset = datasets.CIFAR10()

	sess = keras.backend.get_session()
	wrap = KerasModelWrapper(model)

	desired_attack = map_norm_to_attack(args.attack_type)
	attack = desired_attack(dataset, wrap, sess)

	custom_params   = None
	acc, adv_acc = helpers.benchmarking(model, dataset, attack, bs=args.batch_size, cp=custom_params)
	print(">> Test accuracy on original data : %f" % acc)
	print(">> Test accuracy on adversarial data : %f" % adv_acc)
