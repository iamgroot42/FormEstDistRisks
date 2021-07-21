import numpy as np
import os

from matplotlib import pyplot as plt

# Increase DPI
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


def read_values(path):
	values = []
	with open(path, 'r') as f:
		for line in f:
			line = [float(x) for x in line.rstrip('\n').split(' ')]
			values.append(line)
	return np.abs(np.array(values))


def threshold_crossed(x, y):
	return 1 * (x >= y)


def gamma_decay(files_path, granularity=1000, x_cap=0.025):
	all_attacks = []
	for file in os.listdir(files_path):
		if "clean" in file:
			continue
		v = read_values(os.path.join(files_path, file))
		all_attacks.append(v[0])

		test_gamma_values = list(range(granularity))[: int(granularity * x_cap)]
		gamma_values = [x / granularity for x in test_gamma_values]

		y = [np.sum(threshold_crossed(v[0], x)) for x in gamma_values]
		plt.plot(gamma_values, y, label=file.split(".")[0])

	
	return np.array(all_attacks)


def joint_gamma_decay(all_attacks, plot_save_path, granularity=1000, x_cap=0.025):
	test_gamma_values = list(range(granularity))[: int(granularity * x_cap)]
	gamma_values = [x / granularity for x in test_gamma_values]
	satisfied = lambda val: np.all(np.array([threshold_crossed(x, val) for x in all_attacks]), axis=0)
	y = [np.sum(satisfied(x)) for x in gamma_values]
	plt.plot(gamma_values, y, label="joint")

	plt.legend()
	plt.grid()
	# plt.yscale("log")
	plt.savefig("%s.png" % plot_save_path)


if __name__ == "__main__":
	import sys
	files_path = sys.argv[1]
	plot_save_path = sys.argv[2]
	x_cap = 0.0125
	# Plot decay in number of gamma-robust features with the value of gamma (per 1 v/s all class case)
	all_attacks = gamma_decay(files_path, granularity=1000, x_cap=x_cap)
	# Plot jointly gamma-robust features with varying values of gamma (per 1 v/s all class case)
	joint_gamma_decay(all_attacks, plot_save_path, x_cap=x_cap)
	# Plot p-robust features for varying values of p (per 1 v/s all class case)