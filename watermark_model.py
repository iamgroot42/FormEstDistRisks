from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

from sklearn.utils import shuffle

import keras
from keras.datasets import cifar10

import tensorflow as tf
import numpy as np
import sys

import models, common, datasets


if __name__ == "__main__":
	common.conserve_gpu_memory()

