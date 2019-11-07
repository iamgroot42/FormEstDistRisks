import numpy as np
import os

from matplotlib import pyplot as plt

# Increase DPI
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300


def read_values(path):
	i = 0
	values, indices = [], []
	with open(path, 'r') as f:
		for line in f:
			line = [float(x) for x in line.rstrip('\n').split(' ')]
			if i == 0:
				values.append(line)
			else:
				line = [int(x) for x in line]
				indices.append(line)
			i = (i + 1) % 2
	return np.abs(np.array(values)), np.array(indices)


def threshold_crossed(x, y):
	return 1 * (x >= y)


def gamma_decay(files_path, plot_name, focus_class, granularity=1000, x_cap=0.35):
	all_attacks = []
	for file in os.listdir(files_path):
		v, _ = read_values(os.path.join(files_path, file))
		all_attacks.append(v[focus_class])

		test_gamma_values = list(range(granularity))[: int(granularity * x_cap)]
		gamma_values = [x / granularity for x in test_gamma_values]

		y = [np.sum(threshold_crossed(v[focus_class], x)) for x in gamma_values]
		plt.plot(gamma_values, y, label=file.split(".")[0])

	plt.legend()
	plt.grid()
	# plt.savefig('%s.png' % plot_name)
	return np.array(all_attacks)


def joint_gamma_decay(all_attacks, granularity=1000, x_cap=0.35):
	plt.clf()
	test_gamma_values = list(range(granularity))[: int(granularity * x_cap)]
	gamma_values = [x / granularity for x in test_gamma_values]
	satisfied = lambda val: np.all(np.array([threshold_crossed(x, val) for x in all_attacks]), axis=0)
	y = [np.sum(satisfied(x)) for x in gamma_values]
	
	plt.plot(gamma_values, y)
	plt.grid()
	plt.savefig("combined_gamma_robust.png")


if __name__ == "__main__":
	import sys
	files_path = sys.argv[1]
	plot_save_path = sys.argv[2]
	# Plot decay in number of gamma-robust features with the value of gamma (per 1 v/s all class case)
	all_attacks = gamma_decay(files_path, plot_save_path, focus_class=0, granularity=1000)
	# Plot jointly gamma-robust features with varying values of gamma (per 1 v/s all class case)
	joint_gamma_decay(all_attacks)
	# Plot p-robust features for varying values of p (per 1 v/s all class case)