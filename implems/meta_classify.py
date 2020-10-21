import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
from tensorflow import keras
import os

import utils


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='census', help='which dataset to work on (census/mnist/celeba)')
	args = parser.parse_args()
	utils.flash_utils(args)


	if args.dataset == 'census':
		# Census Income dataset

		base_path = "census_models_mlp_many"
		paths = ['original', 'income', 'sex', 'race']
		ci = utils.CensusIncome("./census_data/")

		sex_filter    = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, 0.65)
		race_filter   = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  1.0)
		income_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, 0.5)

		_, (x_te, y_te), cols = ci.load_data()
		cols = list(cols)
		# desired_property = cols.index("sex:Female")

		# Get intermediate layer representations
		from sklearn.neural_network._base import ACTIVATIONS

		import matplotlib.pyplot as plt
		import matplotlib as mpl
		mpl.rcParams['figure.dpi'] = 200

		def layer_output(data, MLP, layer=0):
			L = data.copy()
			for i in range(layer):
				L = ACTIVATIONS['relu'](np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
			return L

		import math

		w, b = [], []
		labels = []
		for i, path_seg in enumerate(paths):
			for path in os.listdir(os.path.join(base_path, path_seg)):
				clf = load(os.path.join(base_path, path_seg, path))

				# Look at initial layer weights, biases
				# processed = np.exp(clf.coefs_[0])
				# processed = processed / (processed + 1)
				# processed = clf.coefs_[0] ** 2
				processed = clf.coefs_[0]
				processed = np.mean(processed, 1)
				w.append(processed)
				b.append(clf.intercepts_[0])
				labels.append(i)

		w = np.array(w)
		b = np.array(w)
		labels = np.array(labels)

		clf = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=200)
		from sklearn.model_selection import train_test_split
		X_train, X_test, y_train, y_test = train_test_split(w, labels, test_size=0.2, random_state=42)
		clf.fit(X_train, y_train)
		print(clf.score(X_train, y_train))
		print(clf.score(X_test, y_test))
