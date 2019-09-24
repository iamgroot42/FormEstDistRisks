import tensorflow as tf
import numpy as np

import keras
import keras.backend as K
from keras.datasets import cifar10
from keras.models import load_model, Model

from tqdm import tqdm

from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta

import matplotlib.pyplot as plt


class CustomPGD:
	def __init__(self, model, eps, eps_iter, nb_iter):
		self.model = model
		self.eps = eps
		self.eps_iter = eps_iter
		self.nb_iter = nb_iter
		self.sess = K.get_session()

	def modified_loss(self, target_rep, x):
		# Implement fake-relu (usecase???)
		rep = self.model(x)
		loss = tf.norm(rep - target_rep, ord='euclidean', axis=-1) / tf.norm(target_rep, ord='euclidean', axis=-1)
		return loss

	def tf_optimize_loop(self, x, x_preds, x_r_seed, init_loss):
		adv_x = tf.identity(x_r_seed)

		def cond(i, *_):
			return tf.less(i, self.nb_iter)

		def body(i, adv_x, best_advx, best_loss):
			loss = self.modified_loss(x_preds, adv_x)
			grads,  = tf.gradients(loss, adv_x)

			# Normalize gradients
			g_norm = tf.norm(tf.reshape(grads, [tf.shape(grads)[0], -1]), ord='euclidean', axis=-1)
			g_norm = tf.reshape(g_norm, [-1, 1, 1, 1])
			scaled_grads = grads / (g_norm + 1e-10)
			# Take step
			adv_x = adv_x - scaled_grads * self.eps_iter

			# Clip perturbation eta to L-2 norm ball
			eta = adv_x - x
			# eta = clip_eta(eta, 2, self.eps)
			adv_x = tf.clip_by_value(x + eta, 0, 1)
			
			# Keep track of best loss and perturbation
			best_advx =  tf.where(tf.less(loss, best_loss), adv_x, best_advx)
			best_loss =  tf.where(tf.less(loss, best_loss), loss,  best_loss)

			return i + 1, adv_x, best_advx, best_loss

		_, _, best_adv_x, best_loss = tf.while_loop(cond, body, (tf.zeros([]), adv_x, adv_x, init_loss),
										back_prop=True,
										maximum_iterations=self.nb_iter)
		return tf.identity(best_adv_x), tf.identity(best_loss)

	def find_optimal_datapoint(self, target, target_preds, seed):
		x         = tf.placeholder(tf.float32, shape=(None,) + tuple(target.shape)[1:])
		x_preds   = tf.placeholder(tf.float32, shape=(None,) + tuple(target_preds.shape)[1:])
		x_r_seed  = tf.placeholder(tf.float32, shape=(None,) + tuple(seed.shape)[1:])
		init_loss = tf.placeholder(tf.float32, shape=(None,))
		adv_x, loss = self.tf_optimize_loop(x, x_preds, x_r_seed, init_loss)
		adv_x, loss = self.sess.run([adv_x, loss], feed_dict={x: target, x_preds: target_preds, x_r_seed: seed, init_loss: np.inf * np.ones((target.shape[0]))})
		return adv_x


def sample_from_data(data, n_points):
	choices = list(range(len(data)))
	picked = np.random.choice(choices, n_points)
	return data[picked]


def construct_robust_dataset(model, data, sample_strategy, batch_size=512):
	pgd = CustomPGD(model, eps=1, eps_iter=0.1, nb_iter=1000)
	robust_data = []
	for i in tqdm(range(0, len(data), batch_size)):
		x = data[i: i + batch_size]
		x_preds = model.predict(x, batch_size=batch_size)
		x_seed = sample_strategy(data, batch_size)
		x_r = pgd.find_optimal_datapoint(x, x_preds, x_seed)
		robust_data.append(x_r)
	return robust_data


def cifar_to_robust(model):
	feature_extractor = Model(model.inputs, model.layers[-3].output)
	print(feature_extractor.layers)
	(X_train, _), (X_val, _) = cifar10.load_data()
	X_train   = (X_train.astype('float32')) / 255
	X_val     = (X_val.astype('float32')) / 255
	robust_train = construct_robust_dataset(feature_extractor, X_train, sample_from_data)
	robust_val   = construct_robust_dataset(feature_extractor, X_val, sample_from_data)
	return (robust_train, robust_val)


def create_and_save_robust_cifar(model, path):
	(X_train, X_val) = cifar_to_robust(model)
	np.savez(path, X_train=X_train, X_val=X_val)


def load_robust_cifar(path):
	data = np.load(path)
	X_train, X_val = data['X_train'], data['X_val']
	(_, y_train), (_, y_val) = cifar10.load_data()
	return (X_train, y_train), (X_val, y_val)


if __name__ == "__main__":
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth=True
	session = tf.compat.v1.Session(config=config)
	keras.backend.set_session(session)

	model = load_model("./adversarialy_trained_final.h5")
	create_and_save_robust_cifar(model, "./robust_cifar_data.npz")
