import tensorflow as tf
import numpy as np

import keras
import keras.backend as K
from keras.models import load_model, Model

from tqdm import tqdm
import argparse

from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta

from vis.utils.utils import apply_modifications

import common, datasets


parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', type=str, metavar='STRING', help='batch size(default: 128)')
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER', help='batch size (default: 128)')
parser.add_argument('-g','--save_here', type=str, default="./datasets/robust_cifar_data.npz", metavar='STRING', help='path where generated data should be saved')
parser.add_argument('-f','--fake_relu', type=bool, default=True, metavar='BOOLEAN', help='use fake-relu for last relu activation layer?')
args = parser.parse_args()


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

			# Clip perturbation eta to L-2 norm ball (not needed)
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


def construct_robust_dataset(model, data, sample_strategy, batch_size):
	pgd = CustomPGD(model, eps=np.inf, eps_iter=0.1, nb_iter=1000)
	robust_data = []
	for i in tqdm(range(0, len(data), batch_size)):
		x = data[i: i + batch_size]
		x_preds = model.predict(x, batch_size=batch_size)
		x_seed = sample_strategy(data, len(x))
		x_r = pgd.find_optimal_datapoint(x, x_preds, x_seed)
		robust_data.append(x_r)
	return robust_data


@tf.custom_gradient
def fake_relu_activation(x):
	result = tf.maximum(x, 0)
	def custom_grad(dy):
		return dy
	return result, custom_grad


def cifar_to_robust(model, fake_relu, batch_size):
	feature_extractor = Model(model.inputs, model.layers[-3].output) # Almost always true, since last 2 layers are softmax & dense(classes)
	if fake_relu:
		update_layer_activation(feature_extractor, fake_relu_activation, -2)
	dataset = datasets.CIFAR10()
	(X_train, y_train), (X_val, y_val) = dataset.get_data()
	robust_train = construct_robust_dataset(feature_extractor, X_train, sample_from_data, batch_size)
	robust_val   = construct_robust_dataset(feature_extractor, X_val, sample_from_data, batch_size)
	return (robust_train, y_train, robust_val, y_val)


def create_and_save_robust_cifar(model, path, fake_relu, batch_size):
	(X_train, Y_train, X_val, Y_val) = cifar_to_robust(model, fake_relu, batch_size)
	np.savez(path, X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)


def update_layer_activation(model, activation, index=-1):
    model.layers[index].activation = activation
    return apply_modifications(model, custom_objects={"fake_relu_activation": fake_relu_activation})


if __name__ == "__main__":
	common.conserve_gpu_memory()
	model = load_model(args.model)
	create_and_save_robust_cifar(model, args.save_here, args.fake_relu, args.batch_size)
