import keras
import tensorflow as tf
import numpy as np

import models, common, datasets, attacks, helpers


def prepare_target_labels(dataset, actual_labels):
	labels = []
	for l in np.argmax(actual_labels, axis=1):
		possible_labels = [i for i in range(dataset.classes) if i != l]:
		picked_label = np.random.choice(possible_labels)
		labels.append(picked_label)
	labels = keras.utils.to_categorical(labels, dataset.classes)
	return labels


def watermark_with_attack(dataset, attack, model, percentage, bs=128):
	(X_train, Y_train), _  = dataset.get_data()
	choices = list(range(len(X_train)))
	assert (percentage >= 0 and percentage <= 1), "Percentage of data to use for watermarking must be valid"
	picked_indices = np.random.choice(choices, int(len(X_train) * percentage))

	Y_target = prepare_target_labels(Y_train[picked_indices])
	X_attacked = attack.batch_attack(X_train[picked_indices], batch_size=bs, custom_params={'y_target': Y_target})

	return (X_attacked, Y_target)


def specific_noise_watermark(dataset, attack, model, percentage, bs=128):
	# Generate specific-random noise vector
	noise_vector = np.random.standard_normal(dataset.sample_shape)
	(X_train, Y_train), _  = dataset.get_data()
	choices = list(range(len(X_train)))
	assert (percentage >= 0 and percentage <= 1), "Percentage of data to use for watermarking must be valid"
	picked_indices = np.random.choice(choices, int(len(X_train) * percentage))

	Y_target = prepare_target_labels(Y_train[picked_indices])
	X_attacked = np.clip(X_train[picked_indices] + noise_vector, 0, 1)

	return (X_attacked, Y_target), noise_vector
