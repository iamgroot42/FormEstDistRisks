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

		paths = ['census_models/original/', 'census_models/income/', 'census_models/sex/', 'census_models/race/']
		ci = utils.CensusIncome("./census_data/")

		sex_filter    = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, 0.65)
		race_filter   = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  1.0)
		income_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, 0.5)

		_, (x_te, y_te, cols) = ci.load_data()
		cols = list(cols)
		desired_property = cols.index("sex:Female")
		# Focus on performance of desired property
		desired_ids = x_te[:, desired_property] == 1
		x_te, y_te  = x_te[desired_ids], y_te[desired_ids]

		for path_seg in paths:
			perfs = []
			for path in os.listdir(path_seg):
				clf = load(os.path.join(path_seg, path))
				perfs.append(clf.score(x_te, y_te.ravel()))
			print("%s : %.4f +- %.4f" % (path_seg.split("/")[-2], np.mean(perfs), np.std(perfs)))
