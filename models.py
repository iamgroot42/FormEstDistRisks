import keras


def ResNet50(input_shape, classes):
	initial_lr = 1e-1

	def scheduler(epoch):
		if epoch % 50 == 0 and epoch > 0:
			print("[Model] Dropping learning rate by a magnitude!")
		return initial_lr / (10 ** (epoch // 50))

	model = keras.applications.resnet50.ResNet50(include_top=False, 
		input_shape=input_shape,
		pooling='avg',
		classes=classes, 
		weights=None)
	x = keras.layers.Dense(classes, name='fc1000')(model.output)
	x = keras.layers.Activation('softmax')(x)
	resnet = keras.models.Model(model.inputs, x)
	sgd = keras.optimizers.SGD(lr=initial_lr, momentum=0.9, nesterov=True, decay=5e-4)
	resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	return resnet, scheduler
