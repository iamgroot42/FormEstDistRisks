

def benchmarking(model, dataset, attack, bs):
	_, (X_val, Y_val) = dataset.get_data()
	acc = model.evaluate(X_val, Y_val, batch_size=bs, verbose=0)[1]

	X_adv_val = attack.batch_attack(X_val, batch_size=bs)
	adv_acc = model.evaluate(X_adv_val, Y_val, batch_size=bs, verbose=0)[1]
	return acc, adv_acc
