import numpy as np
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr   


def get_sensitivities(path):
	features = []
	with open(path, 'r') as f:
		for line in tqdm(f):
			values = np.array([float(x) for x in line.rstrip('\n').split(',')])
			features.append(values)
	return np.array(features)


def plot_specific(values, plot_save_path, dump=False):
	x = list(range(values.shape[1]))

	if dump:
		sorted_first = np.argsort(values[42])
	
		with open("./sorted_indices.txt", 'w') as f:
			sorted_text = ",".join([str(x) for x in sorted_first])
			f.write(sorted_text)
	else:
		with open("./sorted_indices.txt", 'r') as f:
			for line in f:
				sorted_text = [int(x) for x in line.rstrip('\n').split(',')]
				sorted_first = np.array(sorted_text)
				break

	# Observe specific neurons for better visualization
	# valid_indices = np.all(values <= 500, axis=1)
	# values = values[valid_indices]

	# Check how many neurons follow same image sorted ordering
	sorted_values = 0
	prval, twopv = 0, 0
	for i, v in enumerate(values):
		sorted_value = np.sort(v)
		sorted_values += np.all(sorted_value == v[sorted_first])
		coeffs = pearsonr(sorted_value, v[sorted_first])
		prval += coeffs[0]
		twopv += coeffs[1]
	print("Exact-ordering compatible neurons : %d / %d" % (sorted_values, values.shape[0]))
	print("Average Pearson Correlation with given ordering : %f , 2-tailed p-value :  %f" % (prval / len(values) , twopv / len(values)))

	# cutoff = 5
	# sorted_first = sorted_first[-cutoff:]
	# x = x[-cutoff:]
	for i, v in enumerate(values):
		plt.plot(x, v[sorted_first], label="Feature #%d" % (i+1))
	plt.grid()
	plt.savefig("%s.png" % plot_save_path)


def get_stats(base_path):
	mean = np.load(os.path.join(base_path, "feature_mean.npy"))
	std  = np.load(os.path.join(base_path, "feature_std.npy"))
	return (np.expand_dims(mean, 1), np.expand_dims(std, 1))


def scale_senses(senses, mean, std):
	return (senses - np.repeat(mean, senses.shape[1], axis=1)) / (std + np.finfo(float).eps)


if __name__ == "__main__":
	import sys
	# Increase DPI
	import matplotlib as mpl
	mpl.rcParams['figure.dpi'] = 200

	senses = get_sensitivities(sys.argv[1])
	(mean, std) = get_stats(sys.argv[2])
	scaled_senses = scale_senses(senses, mean, std)
	plot_specific(scaled_senses, sys.argv[3])
