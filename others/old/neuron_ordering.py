import numpy as np
from tqdm import tqdm
import os


def get_sensitivities(path):
	features = []
	with open(path, 'r') as f:
		for line in tqdm(f):
			values = np.array([float(x) for x in line.rstrip('\n').split(',')])
			features.append(values)
	return np.array(features)


def get_stats(base_path):
	mean = np.load(os.path.join(base_path, "feature_mean.npy"))
	std  = np.load(os.path.join(base_path, "feature_std.npy"))
	return (np.expand_dims(mean, 1), np.expand_dims(std, 1))


def scale_senses(senses, mean, std):
	# return senses
	return (senses - np.repeat(mean, senses.shape[1], axis=1)) / (std + np.finfo(float).eps)


def play_with_matrix(mat):
	sum_m = []
	for i in range(mat.shape[0]):
		inf_counts = np.sum(mat[i] == np.inf)
		mat_interest = mat[i][mat[i] != np.inf]
		sum_m.append(np.average(np.abs(mat_interest)))
	print(np.min(sum_m), np.max(sum_m))
	return sum_m


def dump_ordering(values, fpath):
	# Most sensitive to least sensetive
	neuron_indices = np.argsort(values)
	with open(fpath, 'w') as f:
		f.write(",".join([str(x) for x in neuron_indices]) + "\n")


if __name__ == "__main__":
	import sys
	deltas_path = sys.argv[1]
	(mean, std) = get_stats(sys.argv[2])
	deltas = get_sensitivities(deltas_path)
	scaled_deltas = scale_senses(deltas, mean, std)
	scores = play_with_matrix(scaled_deltas)
	dump_ordering(scores, sys.argv[3])
