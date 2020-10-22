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

		base_path = "census_models_mlp"
		paths = ['original', 'income', 'sex', 'race']
		ci = utils.CensusIncome("./census_data/")

		sex_filter    = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, 0.65)
		race_filter   = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  1.0)
		income_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, 0.5)

		_, (x_te, y_te), cols = ci.load_data()
		cols = list(cols)
		# desired_property = cols.index("sex:Female")
		desired_property = cols.index("race:White")
		# Focus on performance of desired property
		# desired_ids = (y_te == 1)[:,0]
		# desired_ids = x_te[:, desired_property] >= 0
		# x_te, y_te  = x_te[desired_ids], y_te[desired_ids]

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

		print(cols)
		focus_feature = cols.index("sex:Male")
		print(cols.index("sex:Male"), cols.index("sex:Female"))
		w, b = [], []
		for path_seg in paths:
			w_, b_ = [], []
			for path in os.listdir(os.path.join(base_path, path_seg)):
				clf = load(os.path.join(base_path, path_seg, path))

				# Look at initial layer weights, biases
				v, c = np.unique(np.argsort(-np.abs(clf.coefs_[0]), 0)[:5, :], return_counts=True)
				print((v[np.argsort(-c)[:10]]))

				# w_.append(clf.coefs_[0][focus_feature])
				# w_.append(np.sum(clf.coefs_[0][focus_feature]))
				# w_.append(np.argmax(clf.coefs_[0], 0))
				# w_.append(np.sum(clf.coefs_[0][focus_feature] <= 0))
				w_.append(np.abs(clf.coefs_[0][focus_feature]) / np.sum(np.abs(clf.coefs_[0]), 0))
				# print(sorted(w_[-1], reverse=True)[:3])
				# w_.append(np.sum(np.abs(clf.coefs_[0][focus_feature])))
				b_.append(clf.intercepts_[0][focus_feature])

			# print(sorted(w_))
			print()
			w.append(w_)
			b.append(b_)

		print(cols[focus_feature])
		colors  = ['indianred', 'limegreen', 'blue', 'orange']
		for i in range(len(w)):
			# plt.scatter(w[i], np.zeros_like(w[i]), label=paths[i])
			# plt.scatter(b[i], np.zeros_like(b[i]), label=paths[i])
			# print(np.mean(b[i]), np.std(b[i]))
			print(np.mean(w[i]), np.std(w[i]))
			# if i != 3: continue
			plt.hist(w[i], label=paths[i], color=[colors[i]] * len(w[i]))
			# plt.hist(b[i], label=paths[i])

		plt.legend()
		plt.savefig("../visualize/model_weight_analysis.png")
