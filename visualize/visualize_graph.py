import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import networkx as nx
import matplotlib.pyplot as plt
import utils
import numpy as np
from tqdm import tqdm
import torch as ch


if __name__ =="__main__":
	import sys

	folder_path = sys.argv[1]
	model_type  = sys.argv[2]
	graph_name  = sys.argv[3]
	batch_size  = 1500 * 4
	n_classes   = 10
	count_matrix = np.zeros((n_classes, n_classes))

	constants = utils.RobustCIFAR10(folder_path, None)
	# constants = utils.CIFAR10()

	model = utils.CIFAR10().get_model(model_type , "vgg19", parallel=True)
	ds = constants.get_dataset()
	_, data_loader = ds.make_loaders(batch_size=batch_size, workers=16, shuffle_val=False, only_val=True)

	for (images, labels) in tqdm(data_loader, total=len(data_loader)):
		logits, _ = model(images.cuda())
		labels_ = ch.argmax(logits, 1)
		for i, j in zip(labels, labels_):
			count_matrix[i][j] += 1

	# Filter out small values
	count_matrix[count_matrix < 15] = 0

	# Normalize row-wise
	for i in range(count_matrix.shape[0]):
		count_matrix[i] /= np.sum(count_matrix[i])

	# Normalize column-wise
	# for i in range(count_matrix.shape[1]):
		 # count_matrix[:, i] /= np.sum(count_matrix[:, i])
	count_matrix *= 100

	# print(count_matrix)

	# Normalize column-wise
	# count_matrix = count_matrix.T
	# count_matrix[:, ] /= np.sum(count_matrix, 0)
	# count_matrix = count_matrix.T
	# count_matrix *= 100

	# Normalize weights, bring to [0,100] range
	# count_matrix /= np.sum(count_matrix)
	# count_matrix *= 100

	G=nx.DiGraph()
	mappinf     = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	node_colors = ['r', 'r', 'b', 'b', 'b', 'b', 'b', 'b', 'r', 'r']
	for x in mappinf:
		G.add_node(x)

	# for i in range(n_classes):
	# 	wt = np.around(count_matrix[i][i], decimals=2)
	# 	G.add_edge(mappinf[i],mappinf[i], weight=wt)

	for i in range(n_classes):
		for j in range(n_classes):
			if count_matrix[i][j] != np.nan:
				wt = np.around(count_matrix[i][j], decimals=2)
				# print(wt, mappinf[i], mappinf[j])
				# if wt > 0.7:
				if i==5 or j==5:
					if wt > 10: G.add_edge(mappinf[i],mappinf[j], weight=wt)

	# print(count_matrix[5, :])
	# print(count_matrix[:, 5])
	for i in range(10):
		print(np.sum(count_matrix[:, i]))

	pos=nx.circular_layout(G)
	nx.draw_networkx(G, pos, node_color=node_colors, node_size=1300, font_color='w')
	labels = nx.get_edge_attributes(G,'weight')
	nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=6)
	plt.savefig("./%s.png" % graph_name)
