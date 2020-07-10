import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


# For eachh neuron, count neurons with which its correlation coefficient is in [-1, -x] U [x, 1]
def get_threshold_satisfiers(corr_matrix, threshold=0.9):
	all_neurons = list(range(corr_matrix.shape[0]))
	cluster_identifiers = []
	for i in range(corr_matrix.shape[0]):
		if i in all_neurons:
			cluster_identifiers.append(i)
			for j in range(corr_matrix.shape[0]):
				if corr_matrix[i][j] >= threshold or corr_matrix[i][j] <= -threshold:
					if j in all_neurons: all_neurons.remove(j)
	return cluster_identifiers


def get_trends(corr_matrix):
	number = 1000
	x, y = [], []
	for i in tqdm(range(number + 1)):
		clusters = get_threshold_satisfiers(corr_matrix, i / number)
		x.append(i / number)
		y.append(len(clusters))
	return (x, y)


def get_mapping(path, scaling=10000):
	mapping = {}
	with open(path, 'r') as f:
		for line in f:
			b, l = line.rstrip('\n').split(':')
			l = l.strip('[]').split(',')
			l = [float(i) / scaling for i in l]
			mapping[int(b)] = np.array(l)
	return mapping


if __name__ == "__main__":
	corr_paths = ["nat_correlation.npy", "linf_correlation.npy", "l2_correlation.npy", "sense_correlation.npy"]
	corr_matrix = np.load(corr_paths[0])
	corr_matrix[corr_matrix != corr_matrix] = 0
	z = get_threshold_satisfiers(corr_matrix, 0.7)
	print(len(z))
	print(z)
	# for path in corr_paths:
	# 	corr_matrix = np.load(path)
	# 	corr_matrix[corr_matrix != corr_matrix] = 0
	# 	x, y = get_trends(corr_matrix)
	# 	plt.plot(x, y, label=path.split('.')[0].split("_")[0])
	# plt.legend()
	# plt.xlabel("Correlation coefficient threshold")
	# plt.ylabel("Number of neuron clusters")
	# plt.savefig('correlation_threshold_trends.png')
	# counts = np.logical_or(corr_matrix >= threshold, corr_matrix <= -threshold)
	# corr_matrix[np.logical_not(counts)] = 0
	# print(corr_matrix)
	# counts = np.sum(counts, 0)
	# print(counts)

	filepaths = ["nat_stats.txt", "linf_stats.txt"]
	labels    = ["standard", "$L_\infty$ robust"]
	for k, fp in enumerate(filepaths):
		mapping = get_mapping(fp)
		corr_matrix = np.load(corr_paths[k])
		corr_matrix[corr_matrix != corr_matrix] = 0
		# Create correlation matrix and populate
		# (i, j) -> when neuron i is targeted, neuron j ends up satisfying delta requirement
		corr = np.zeros((len(mapping), len(mapping)))
		for kk, v in mapping.items():
			corr[kk] = v
		# Look at mean +- std Jaccard similarity across neurons for varying thresholds (to classify as correlated or not)
		thresholds = 100
		means, stds = [], []
		for i in range(thresholds + 1):
			stats = []
			for j in range(corr.shape[0]):
				try:
					x = set(np.argwhere(corr[j] >= (i / thresholds))[0])
				except:
					x = set()
				try:
					y = set(np.argwhere(corr_matrix[j] >= (i / thresholds))[0])
				except:
					y = set()
				js = len(x.intersection(y)) / len(x.union(y))
				stats.append(js)
			means.append(np.mean(stats))
			stds.append(np.std(stats))			
		plt.errorbar(np.arange(len(means)), means, yerr=stds, label=labels[k])
	plt.grid(True)
	plt.legend()
	plt.xlabel('Similarity Threshold')
	plt.ylabel('Average Jaccard Similarity Index')
	plt.savefig("jaccard_corr.png")
