import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.neural_network import MLPClassifier
from joblib import dump, load
import os

import utils



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--savepath', type=str, default='', help='folder where trained model(s) should be saved')
	parser.add_argument('--filter', type=str, default='', help='while filter to use (sex/race/income/none)')
	parser.add_argument('--ratio', type=float, default=0.5, help='what ratio of the new sampled dataset should be true')
	parser.add_argument('--num', type=int, default=20, help='how many classifiers to train?')
	args = parser.parse_args()
	utils.flash_utils(args)

	# Census Income dataset
	ci = utils.CensusIncome("./census_data/")
	ratio = args.ratio

	if args.filter == "sex":
		data_filter = lambda df: utils.filter(df, lambda x: x['sex:Female'] == 1, ratio) #0.65
	elif args.filter == "race":
		data_filter = lambda df: utils.filter(df, lambda x: x['race:White'] == 0,  ratio) #1.0
	elif args.filter == "income":
		data_filter = lambda df: utils.filter(df, lambda x: x['income'] == 1, ratio) #0.5
	elif args.filter == "none":
		data_filter = None
	else:
		raise ValueError("Invalid filter requested")

	for i in range(1, args.num + 1):
		(x_tr, y_tr), (x_te, y_te), _ = ci.load_data(data_filter)
		print("Training classifier %d" % i)
		clf = MLPClassifier(hidden_layer_sizes=(60, 30, 30), max_iter=200)
		clf.fit(x_tr, y_tr.ravel())
		train_acc = 100 * clf.score(x_tr, y_tr.ravel())
		test_acc  = 100 * clf.score(x_te, y_te.ravel())
		print("Classifier %d : Train acc %.2f , Test acc %.2f\n" %
			(i, train_acc, test_acc))

		dump(clf, os.path.join(args.savepath,  str(i) + "_%.2f" % test_acc))
