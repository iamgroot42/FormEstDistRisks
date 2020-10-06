import torch as ch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

# Custom module imports
import utils


def get_sampled_data(sample_ratio):
	# ds = utils.CIFAR10().get_dataset()
	ds = utils.RobustCIFAR10("/p/adversarialml/as9rw/datasets/cifar10_split2/", None).get_dataset()
	n_classes = 10

	# ds = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/new_exp/small/100p_dog/").get_dataset()
	# n_classes = 2

	# data_loader, _ = ds.make_loaders(batch_size=500, workers=8, shuffle_val=False)
	_, data_loader = ds.make_loaders(batch_size=500, workers=8, shuffle_val=False, only_val=True)

	images, labels = utils.load_all_loader_data(data_loader)
	images_, labels_ = [], []
	for i in range(n_classes):
		# eligible_indices = np.nonzero(labels == i)[:,0]
		eligible_indices = np.nonzero(labels == 5)[:,0]
		np.random.shuffle(eligible_indices) 
		# Pick according to ratio
		picked_indices = eligible_indices[:int(len(eligible_indices) * sample_ratio)]
		images_.append(images[picked_indices])
		labels_.append(labels[picked_indices])
	images = ch.cat(images_)
	labels = ch.cat(labels_)

	use_ds = utils.BasicDataset(images, labels)
	return use_ds



def get_reps_for_data(model, dl):
	reps = []
	ys = []
	for x, y in dl:
		# rep, _ = model(x.cuda(), fake_relu=False)
		rep, _ = model(x.cuda(), with_latent=True, fake_relu=False, just_latent=True)
		reps.append(rep.detach().cpu())
		ys.append(y)
	return ch.cat(reps).numpy(), ch.cat(ys).numpy()


if __name__ == "__main__":
	labels  = [
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		1, 1,
		1, 1,
		1, 1,
		1, 1,
		1, 1,
		1, 1
	]
	paths = [
		"0p_linf", "0p_linf_2",
		"10p_linf", "10p_linf_2",
		"20p_linf", "20p_linf_2",
		"30p_linf", "30p_linf_2",
		"40p_linf", "40p_linf_2",
		"50p_linf", "50p_linf_2",
		"60p_linf", "60p_linf_2",
		"70p_linf", "70p_linf_2",
		"80p_linf", "80p_linf_2",
		"90p_linf", "90p_linf_2",
		"100p_linf", "100p_linf_2"
	]

	sample_ratio = 1.0
	# sample_ratio = 0.5
	# sample_ratio = 0.05
	use_ds = get_sampled_data(sample_ratio)

	# Generate PI (permutation invariant) model representations
	prefix = "/p/adversarialml/as9rw/new_exp_models/small/"
	suffix = "checkpoint.pt.best"
	constants = utils.BinaryCIFAR(None)
	reps, ys = [], []
	for i, path in tqdm(enumerate(paths)):
		data_loader = DataLoader(use_ds, batch_size=1024, shuffle=False, num_workers=8)
		model = constants.get_model(os.path.join(prefix, path, suffix) , "vgg19", parallel=True)
		
		rep, _   = get_reps_for_data(model, data_loader)
		label = np.ones((rep.shape[0],)) * labels[i]

		rep_m, rep_std = np.mean(rep, 0), np.std(rep, 0)
		# Sort as a way to capture permutation invariance
		sorted_ids = np.argsort(rep_m)
		rep_m, rep_std = rep_m[sorted_ids], rep_std[sorted_ids]
		rep = np.concatenate([rep_m, rep_std])
		label = labels[i]
		
		reps.append(rep)
		ys.append(label)

	# reps = np.concatenate(reps, 0)
	# ys = np.concatenate(ys, 0)
	reps = np.array(reps)
	ys = np.array(ys)
	
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neural_network import MLPClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.model_selection import cross_val_score

	# X_train, X_test, y_train, y_test = train_test_split(reps, ys, test_size=0.2, stratify=ys)
	# print(y_train.shape[0], y_test.shape[0])

	X_train, y_train = reps, ys


	from sklearn.model_selection import GridSearchCV
	from sklearn.ensemble import VotingClassifier

	# parameters = {'solver': ['lbfgs'], 'max_iter': [500, 1000, 1500], 'alpha': 10.0 ** -np.arange(1, 4), 'hidden_layer_sizes': [(128, 32, 8), (128, 32)]}
	# clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)
	# clf = MLPClassifier(solver='lbfgs', max_iter=500, hidden_layer_sizes=(128, 32, 8), alpha=0.01)
	clf = DecisionTreeClassifier(random_state=0, max_depth=2)

	# scores = cross_val_score(clf, reps, y, cv=5)
	clf.fit(X_train, y_train)
	# print(clf.best_params_)

	from sklearn import tree
	text_representation = tree.export_text(clf)
	# print(text_representation)

	print("Training accuracy: %.2f" % (100 * clf.score(X_train, y_train)))
	# print("Testing accuracy: %.2f" % (100 * clf.score(X_test, y_test)))

	# Perform same test for unseen models
	paths_test = ["10p_linf", "50p_linf"]

	for path in tqdm(paths_test):
		data_loader = DataLoader(use_ds, batch_size=1024, shuffle=False, num_workers=8)
		model = constants.get_model(os.path.join(prefix, path, suffix) , "vgg19", parallel=True)

		rep, _   = get_reps_for_data(model, data_loader)

		rep_m, rep_std = np.mean(rep, 0), np.std(rep, 0)
		sorted_ids = np.argsort(rep_m)
		rep_m, rep_std = rep_m[sorted_ids], rep_std[sorted_ids]
		rep = np.concatenate([rep_m, rep_std])
		rep = np.expand_dims(rep, 0)

		preds = clf.predict_proba(rep)[:,1]

		# print(np.min(preds), np.max(preds))
		print("For %s model, mean score on data is %.2f" % (path, np.mean(preds)))
