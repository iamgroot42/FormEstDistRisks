import numpy as np
from tqdm import tqdm

from matplotlib import pyplot as plt


def get_sensitivities(path):
	features = []
	with open(path, 'r') as f:
		for line in tqdm(f):
			values = np.array([float(x) for x in line.rstrip('\n').split(',')])
			features.append(values)
	return np.array(features)


def plot_specific(values, plot_save_path):
	x = list(range(values.shape[1]))
	sorted_first = np.argsort(values[42])
	# cutoff = 5
	# sorted_first = sorted_first[-cutoff:]
	# x = x[-cutoff:]
	for i, v in enumerate(values):
		plt.plot(x, v[sorted_first], label="Feature #%d" % (i+1))
	plt.grid()
	plt.savefig("%s.png" % plot_save_path)


if __name__ == "__main__":
	import sys
	# Increase DPI
	import matplotlib as mpl
	mpl.rcParams['figure.dpi'] = 200

	senses = get_sensitivities(sys.argv[1])
	plot_specific(senses, sys.argv[2])
