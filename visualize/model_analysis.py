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


# Set threshold
threshold = 1e5

# Get processed delta values
# nat_senses  = process_neurons("/p/adversarialml/as9rw/binary_stats/nat/deltas_nat.txt", "/p/adversarialml/as9rw/binary_stats/nat/stats", threshold=threshold)
# l1_senses   = process_neurons("/p/adversarialml/as9rw/binary_stats/l1/delta_l1.txt", "/p/adversarialml/as9rw/binary_stats/l1/stats", threshold=threshold)
# l2_senses   = process_neurons("/p/adversarialml/as9rw/binary_stats/l2/delta_l2.txt", "/p/adversarialml/as9rw/binary_stats/l2/stats", threshold=threshold)
# linf_senses = process_neurons("/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt", "/p/adversarialml/as9rw/binary_stats/linf/stats", threshold=threshold)
nat_senses  = process_neurons("/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt", "/p/adversarialml/as9rw/cifar10_stats/nat/stats", threshold=threshold)
l2_senses   = process_neurons("/p/adversarialml/as9rw/cifar10_stats/l2/deltas.txt", "/p/adversarialml/as9rw/cifar10_stats/l2/stats", threshold=threshold)
linf_senses = process_neurons("/p/adversarialml/as9rw/cifar10_stats/linf/deltas.txt", "/p/adversarialml/as9rw/cifar10_stats/linf/stats", threshold=threshold)


# Get processed delta values, but scaled according to adv examples
# nat_adv_senses  = process_neurons("/p/adversarialml/as9rw/binary_stats/nat/deltas_nat.txt", "/p/adversarialml/as9rw/binary_stats/nat/l2_stats", threshold=threshold)
# l1_adv_senses   = process_neurons("/p/adversarialml/as9rw/binary_stats/l1/delta_l1.txt", "/p/adversarialml/as9rw/binary_stats/l1/l2_stats", threshold=threshold)
# l2_adv_senses   = process_neurons("/p/adversarialml/as9rw/binary_stats/l2/delta_l2.txt", "/p/adversarialml/as9rw/binary_stats/l2/l2_stats", threshold=threshold)
# linf_adv_senses = process_neurons("/p/adversarialml/as9rw/binary_stats/linf/delta_linf.txt", "/p/adversarialml/as9rw/binary_stats/linf/l2_stats", threshold=threshold)
nat_adv_senses  = process_neurons("/p/adversarialml/as9rw/cifar10_stats/nat/deltas.txt", "/p/adversarialml/as9rw/cifar10_stats/nat/l2_stats", threshold=threshold)
l2_adv_senses   = process_neurons("/p/adversarialml/as9rw/cifar10_stats/l2/deltas.txt", "/p/adversarialml/as9rw/cifar10_stats/l2/l2_stats", threshold=threshold)
linf_adv_senses = process_neurons("/p/adversarialml/as9rw/cifar10_stats/linf/deltas.txt", "/p/adversarialml/as9rw/cifar10_stats/linf/l2_stats", threshold=threshold)

# Make PD
# data = {'Natural': nat_senses, 'L-1': l1_senses, 'L-2': l2_senses, 'L-inf': linf_senses,
# 		'a_Natural': nat_adv_senses, 'a_L-1': l1_adv_senses, 'a_L-2': l2_adv_senses, 'a_L-inf': linf_adv_senses}
# df = pd.DataFrame(data, columns = ['Natural', 'a_Natural', 'L-1', 'a_L-1', 'L-2', 'a_L-2', 'L-inf', 'a_L-inf'])
data = {'Natural': nat_senses, 'L-2': l2_senses, 'L-inf': linf_senses,
		'a_Natural': nat_adv_senses, 'a_L-2': l2_adv_senses, 'a_L-inf': linf_adv_senses}
df = pd.DataFrame(data, columns = ['Natural', 'a_Natural', 'L-2', 'a_L-2', 'L-inf', 'a_L-inf'])

# Filter out near-impossible attacks to neurons
filtered_df = df[(df.Natural <= threshold) & (df['L-2'] <= threshold) & (df['a_L-2'] <= threshold) \
				& (df['L-inf'] <= threshold) & (df['a_L-inf'] <= threshold)]

sns.set(style="whitegrid")
ax = sns.violinplot(data=filtered_df, palette="muted", split=True, cut=0)# , inner="quartile")

fig = ax.get_figure()
fig.savefig("analmodel.png")

# Side-by-Side analysis mode

# model, delta, adv = [], [], []
# # Add data one segment at a time
# def add_to_data(name, senses, type):
# 	model.append([name] * len(senses))
# 	delta.append(senses)
# 	adv.append([type] * len(senses))

# # Add normal data
# add_to_data('Natural', nat_senses, 'no')
# add_to_data('L-1', l1_senses, 'no')
# add_to_data('L-2', l2_senses, 'no')
# add_to_data('L-inf', linf_senses, 'no')

# # Add adversarial data
# add_to_data('Natural', nat_adv_senses, 'yes')
# add_to_data('L-1', l1_adv_senses, 'yes')
# add_to_data('L-2', l2_adv_senses, 'yes')
# add_to_data('L-inf', linf_adv_senses, 'yes')

# # Collapse categories into long lists
# model = np.concatenate(model, axis=0)
# delta = np.concatenate(delta, axis=0)
# adv   = np.concatenate(adv, axis=0)

# # Make PD
# data = {'model': model, 'abs_delta': delta, 'adversarial': adv}
# df = pd.DataFrame(data, columns = ['model', 'abs_delta', 'adversarial'])

# sns.set(style="whitegrid")
# ax = sns.violinplot(x="model", y="abs_delta", hue="adversarial", data=df, palette="muted", split=True, cut=0)# , inner="quartile")

# fig = ax.get_figure()
# fig.savefig("analmodel_sidebyside.png")
