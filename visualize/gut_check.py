import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.rcParams['figure.dpi'] = 200


path = "/p/adversarialml/as9rw/cifar10_stats/resnet50/nat/deltas.txt"
senses = utils.get_sensitivities(path)

x = list(range(senses.shape[0]))

counts = []
for i in range(senses.shape[0]):
	counts.append(np.sum(senses[i] == np.inf))
counts = np.array(counts)

indices = np.argsort(counts)
counts = counts[indices]

def chuck_inf_means(senses):
	chucked = []
	for i in range(senses.shape[0]):
		x = senses[i]
		chucked.append(np.mean(x[x < 1e3]))
	return np.array(chucked)

deltas = chuck_inf_means(senses)[indices]

plt.plot(x, counts)
plt.plot(x, deltas)
plt.savefig("gut.png")