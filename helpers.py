import matplotlib

def benchmarking(model, dataset, attack, bs, cp=None):
	_, (X_val, Y_val) = dataset.get_data()
	acc = model.evaluate(X_val, Y_val, batch_size=bs, verbose=0)[1]

	X_adv_val = attack.batch_attack(X_val, batch_size=bs, custom_params=cp)
	adv_acc = model.evaluate(X_adv_val, Y_val, batch_size=bs, verbose=0)[1]
	return acc, adv_acc


def save_image(image, path):
	matplotlib.image.imsave(path, image)
