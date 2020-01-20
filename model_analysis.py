import seaborn as sns
import numpy as np
import utils
import pandas as pd

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def process_neurons(sense_path, scale_path, threshold=1e5):
	senses   = utils.get_sensitivities(sense_path)
	(mean, std) = utils.get_stats(scale_path)
	senses = utils.scaled_values(senses, mean, std)
	wanted = np.zeros((senses.shape[0],))
	senses = np.abs(senses)
	for i in range(wanted.shape[0]):
		# Consider only values below threshold
		picked = senses[i, senses[i] != np.inf]
		picked = picked[picked <= threshold]
		if len(picked) == 0:
			wanted[i] = threshold * 1.5
		else:
			wanted[i] = np.mean(picked)
	return wanted

# Get processed delta values
nat_senses  = process_neurons("/p/adversarialml/as9rw/binary_stats/nat/deltas_nat.txt", "/p/adversarialml/as9rw/binary_stats/nat/stats")
l1_senses   = process_neurons("/p/adversarialml/as9rw/binary_stats/l1/delta_l1.txt", "/p/adversarialml/as9rw/binary_stats/l1/stats")
l2_senses   = process_neurons("/p/adversarialml/as9rw/binary_stats/l2/delta_l2.txt", "/p/adversarialml/as9rw/binary_stats/l2/stats")
linf_senses = process_neurons("/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt", "/p/adversarialml/as9rw/binary_stats/linf/stats")

# Make PD
data = {'Natural': nat_senses, 'L-1': l1_senses, 'L-2': l2_senses, 'L-inf': linf_senses}
df = pd.DataFrame(data, columns = ['Natural', 'L-1', 'L-2', 'L-inf'])

sns.set(style="whitegrid")
ax = sns.violinplot(data=df, palette="muted", split=True, cut=0)# , inner="quartile")

fig = ax.get_figure()
fig.savefig("analmodel.png")
