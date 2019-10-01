import keras


def ResNet50(input_shape, classes):
	initial_lr = 1e-3

	def scheduler(epoch):
		lr = initial_lr
		if epoch > 180:
			lr *= 0.5e-3
			print("[Model] Learning rate : %f" % lr)
		elif epoch > 160:
			lr *= 1e-3
			print("[Model] Learning rate : %f" % lr)
		elif epoch > 120:
			lr *= 1e-2
			print("[Model] Learning rate : %f" % lr)
		elif epoch > 80:
			lr *= 1e-1
			print("[Model] Learning rate : %f" % lr)
		return lr

	model = keras.applications.resnet50.ResNet50(include_top=False, 
		input_shape=input_shape,
		pooling='avg',
		classes=classes, 
		weights=None)
	x = keras.layers.Dense(classes, name='fc1000')(model.output)
	x = keras.layers.Activation('softmax')(x)
	resnet = keras.models.Model(model.inputs, x)
	opt = keras.optimizers.Adam(lr=initial_lr)
	resnet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return resnet, scheduler


def VGG16(input_shape, classes):
	initial_lr = 1e-1
	lr_drop = 20

	def scheduler(epoch):
		lr = initial_lr * (0.5 ** (epoch // lr_drop))
		print("[Model] Learning rate : %f" % lr)
		return lr

	model = keras.applications.vgg16.VGG16(include_top=False, 
		input_shape=input_shape,
		pooling='avg',
		classes=classes, 
		weights=None)
	x = keras.layers.Dense(classes, name='fc1000')(model.output)
	x = keras.layers.Activation('softmax')(x)
	resnet = keras.models.Model(model.inputs, x)
	opt = keras.optimizers.SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True)
	resnet.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	return resnet, scheduler
