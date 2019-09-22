from cleverhans.attacks import MadryEtAl
from cleverhans.utils_keras import KerasModelWrapper

import keras
from keras.datasets import cifar10

import tensorflow as tf
import numpy as np
import sys

import models

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
keras.backend.set_session(session)

model, cbks = models.ResNet50(input_shape=(32, 32, 3), classes=10)
wrap = KerasModelWrapper(model)

attack = MadryEtAl
attack_params = {'clip_min': 0, 'clip_max': 1}
attack_object = attack(wrap, sess=session)
attack_params['nb_iter'] = 7
attack_params['eps'] =5e-1

def raw_to_train(x):
	# mean = [0.4914, 0.4822, 0.4465]
	# var =  [0.2023, 0.1994, 0.2010]
	# return (x - mean) / var
	return x / 255

def train_to_raw(x):
	# mean = [0.4914, 0.4822, 0.4465]
	# var =  [0.2023, 0.1994, 0.2010]
	# return x * var + mean
	return x * 255

(X_train, Y_train), (X_val, Y_val) = cifar10.load_data()
X_train = X_train.astype('float32')
X_val   = X_val.astype('float32')
X_train = raw_to_train(X_train)
X_val   = raw_to_train(X_val)
Y_train = keras.utils.to_categorical(Y_train, 10)
Y_val   = keras.utils.to_categorical(Y_val, 10)

nb_epochs = 150
batch_size = 128

init = tf.global_variables_initializer()
session.run(init)

for i in range(nb_epochs):
	batch_no = 1
	train_loss, train_acc = 0, 0
	for j in range(0, len(X_train), batch_size):
		x_clean, y_clean = X_train[j:j+batch_size], Y_train[j:j+batch_size]
		x_adv = attack_object.generate_np(x_clean, **attack_params)
		x_batch = np.concatenate([x_clean, x_adv], axis=0)
		y_batch = np.concatenate([y_clean, y_clean], axis=0)
		train_metrics = model.train_on_batch(x_batch, y_batch)
		train_loss += train_metrics[0]
		train_acc += train_metrics[1]
		sys.stdout.write("Epoch %d: %d / %d : Tr loss: %f, Tr acc: %f  \r" % (i+1, batch_no, len(X_train)//batch_size, train_loss/(batch_no), train_acc/(batch_no)))
		sys.stdout.flush()
		batch_no += 1
	val_metrics = model.evaluate(X_val, Y_val, batch_size=1024, verbose=0)
	print()
	print(">> Val loss: %f, Val acc: %f"% (val_metrics[0], val_metrics[1]))
	if i % 10 == 9:
		model.save("adversarialy_trained_%d.h5" % (i + 1))

model.save("adversarialy_trained_final.h5" % (i + 1))