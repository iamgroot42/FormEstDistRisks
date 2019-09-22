from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers, regularizers
from keras import backend as K


def residual_network(input_shape, n_classes, stack_n=5, get_logits=False):
	weight_decay       = 1e-4
	img_input = Input(input_shape)

	def scheduler(epoch):
		if epoch < 81:
			return 0.1
		if epoch < 122:
			return 0.01
		return 0.001

	def residual_block(x, o_filters, increase=False):
		stride = (1,1)
		if increase:
			stride = (2,2)
		o1 = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
		conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(o1)
		o2  = Activation('relu')(BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
		conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(o2)
		if increase:
			projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(o1)
			block = add([conv_2, projection])
		else:
			block = add([conv_2, x])
		return block

	x = Conv2D(filters=16, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(img_input)
	for _ in range(stack_n):
		x = residual_block(x, 16, False)
	x = residual_block(x, 32, True)
	for _ in range(1, stack_n):
		x = residual_block(x, 32, False)
	x = residual_block(x, 64, True)
	for _ in range(1, stack_n):
		x = residual_block(x, 64, False)
	x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
	x = Activation('relu')(x)
	x = GlobalAveragePooling2D()(x)
	logits = Dense(n_classes,kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(weight_decay))(x)
	output = Activation('softmax')(logits)
	cbks = [LearningRateScheduler(scheduler), ModelCheckpoint('./checkpoint-resnet-{epoch}.h5', save_best_only=False, mode='auto', period=10)]
	resnet = None
	if get_logits:
		resnet = Model(img_input, logits)
		return resnet, cbks
	else:
		resnet = Model(img_input, output)
	sgd = optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True, decay=5e-4)
	resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return resnet, cbks


def ResNet50(input_shape, classes):
	# 6 * stack_n + 2 = 50
	return residual_network(input_shape, classes, stack_n=8, get_logits=False)
