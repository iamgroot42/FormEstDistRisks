import keras
import tensorflow as tf


def conserve_gpu_memory():
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth=True
	session = tf.compat.v1.Session(config=config)
	keras.backend.set_session(session)
