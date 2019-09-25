from sklearn.utils import shuffle

import keras
from keras.datasets import cifar10

import tensorflow as tf
import numpy as np
import sys

import imgaug.augmenters as iaa

import models
from construct_robust_dataset import load_robust_cifar

import matplotlib

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
keras.backend.set_session(session)

model, cbks = models.ResNet50(input_shape=(32, 32, 3), classes=10)

def raw_to_train(x):
	return x / 255

def train_to_raw(x):
	return 255 * x


def get_image_augmentations():
	seq = iaa.Sequential([
		iaa.Pad(4), # Pad all sides by 4
    	iaa.CropToFixedSize(32, 32), # Crop to fixed size
    	iaa.Fliplr(0.1), # horizontally flip 10% of the images
    	iaa.Affine(rotate=(-2, 2)) # Rotate by 2 degrees
	])
	return seq


def load_normal_data():
	(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
	X_train = X_train.astype('float32')
	X_val   = X_val.astype('float32')
	X_train = raw_to_train(X_train)
	X_val   = raw_to_train(X_val)
	Y_train = keras.utils.to_categorical(Y_train, 10)
	Y_val   = keras.utils.to_categorical(Y_val, 10)
	# Shuffle data
	X_train, Y_train = shuffle(X_train, Y_train)
	return (X_train, Y_train), (X_val, Y_val)


def load_robust_data():
	(X_train, Y_train), (X_val, Y_val) = load_robust_cifar("./datasets/robust_cifar_data.npz")
	Y_train = keras.utils.to_categorical(Y_train, 10)
	Y_val   = keras.utils.to_categorical(Y_val, 10)
	# Shuffle data
	X_train, Y_train = shuffle(X_train, Y_train)
	return (X_train, Y_train), (X_val, Y_val)


(X_train, Y_train), (X_val, Y_val) = load_robust_data()

nb_epochs = 200
batch_size = 128 // 2

init = tf.global_variables_initializer()
session.run(init)

augmentor = get_image_augmentations()

for i in range(nb_epochs):
	batch_no = 1
	train_loss, train_acc = 0, 0
	for j in range(0, len(X_train), batch_size):
		x_clean, y_clean = X_train[j:j+batch_size], Y_train[j:j+batch_size]
		x_aug = augmentor.augment_images(x_clean)
		x_batch = np.concatenate([x_clean, x_aug], axis=0)
		y_batch = np.concatenate([y_clean, y_clean], axis=0)
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
		model.save("./models/normally_trained_%d.h5" % (i + 1))

model.save("./models/normally_trained_final.h5")
