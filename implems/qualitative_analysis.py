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
	base_path = "census_models_mlp"
	paths = ['original', 'income', 'sex', 'race']
	ci = utils.CensusIncome("./census_data/")

	sex_filter    = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, 0.65)
	race_filter   = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  1.0)
	income_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, 0.5)

	_, (x_te, y_te), cols = ci.load_data()
	cols = list(cols)
	# print(cols)
	# desired_property = cols.index("sex:Female")
	desired_property = cols.index("race:White")

	# (x_tr, y_tr), _, cols = ci.load_data()
	# cols = list(cols)
	# print(np.median(x_tr[:, cols.index("age")]))
	# exit(0)
	# desired_property = 1
	# desired_property = cols.index("race:White")
	# Focus on performance of desired property
	# desired_ids = (y_te == 1)[:,0]
	desired_ids = x_te[:, desired_property] >= 0

	# need_em = []
	# for path_seg in paths:
	# 	per_model = []
	# 	for path in os.listdir(os.path.join(base_path, path_seg)):
	# 		clf = load(os.path.join(base_path, path_seg, path))

	# 		preds = clf.predict(x_te)
	# 		# per_model.append(np.nonzero(preds != y_te[:,0])[0])
	# 		per_model.append(np.nonzero(preds != y_te[:,0])[0])
	# 	# Look at common incorrect-examples across these models
	# 	need_em.append(list(set(per_model[0]).intersection(*per_model)))

	# for i, ne in enumerate(need_em):
	# 	x_ = x_te[ne]
	# 	print(paths[i], np.mean(x_[:, desired_property]), np.std(x_[:, desired_property]), np.median(x_[:, desired_property]))


	need_em = []
	for path_seg in paths:
		per_model = []
		for path in os.listdir(os.path.join(base_path, path_seg)):
			clf = load(os.path.join(base_path, path_seg, path))

			preds = clf.predict(x_te)
			per_model.append(preds)
		per_model = np.array(per_model)
		avg_pred = np.mean(per_model, 0)
		need_em.append(np.around(avg_pred))


	print(np.unique(x_te[:, desired_property]))

	need_em = np.array(need_em)

	# Cases where original and sex don't match
	ids = np.nonzero(need_em[0] != need_em[3])[0]
	print(np.unique(x_te[ids, desired_property], return_counts=True))
