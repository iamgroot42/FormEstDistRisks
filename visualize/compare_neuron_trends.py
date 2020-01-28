import numpy as np
import utils
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['figure.dpi'] = 200
plt.style.use('seaborn-whitegrid')

# prefix        = "/p/adversarialml/as9rw/binary_stats/nat/"
prefix               = "/p/adversarialml/as9rw/cifar10_stats/linf/"
normal_stats         = prefix + "stats"
l2_stats             = prefix + "l2_stats"
linf_stats           = prefix + "linf_stats"
custom_l2_stats      = prefix + "custom_l2_stats"
custom_linf_stats    = prefix + "custom_linf_stats"

deltas_path   = prefix + "deltas.txt"

(mean, std) = utils.get_stats(normal_stats)

# Plot only first which_n (for better visualization)
start, end  = 1400, 1440
picked = list(range(start, end))
# picked = np.nonzero(mean >= 0)[0]
x = list(range(len(picked)))

# (mean_adv, std_adv) = utils.get_stats(l1_stats)
# mean_adv = mean_adv[picked]
# std_adv  = std_adv[picked]
# plt.errorbar(x, mean_adv, yerr=std_adv, fmt='*', color='red', ecolor='coral', elinewidth=2, capsize=0, label='L1')
mean = mean[picked]
std  = std[picked]
plt.errorbar(x, mean, yerr=std, fmt='o', color='black', ecolor='lightgray', elinewidth=2, capsize=0, label='natural')

(mean_adv, std_adv) = utils.get_stats(custom_l2_stats)
mean_adv = mean_adv[picked]
std_adv  = std_adv[picked]
plt.errorbar(x, mean_adv, yerr=std_adv, fmt='X', color='red', ecolor='lightcoral', elinewidth=2, capsize=0, label='custom L2')

(mean_adv, std_adv) = utils.get_stats(l2_stats)
mean_adv = mean_adv[picked]
std_adv  = std_adv[picked]
plt.errorbar(x, mean_adv, yerr=std_adv, fmt='x', color='blue', ecolor='lightblue', elinewidth=2, capsize=0, label='L2')

# (mean_adv, std_adv) = utils.get_stats(custom_linf_stats)
# mean_adv = mean_adv[picked]
# std_adv  = std_adv[picked]
# plt.errorbar(x, mean_adv, yerr=std_adv, fmt='*', color='sienna', ecolor='peachpuff', elinewidth=2, capsize=0, label='custom Linf')

# (mean_adv, std_adv) = utils.get_stats(linf_stats)
# mean_adv = mean_adv[picked]
# std_adv  = std_adv[picked]
# plt.errorbar(x, mean_adv, yerr=std_adv, fmt='+', color='gold', ecolor='beige', elinewidth=2, capsize=0, label='Linf')


plt.legend(loc="upper left")
plt.xlabel("Neurons (ordered)")
plt.ylabel("Activation values")
plt.savefig('neuron_spikes.png')
plt.clf()

# Look at neurons where there is a distribution shift wrt activations with and without adversarial examples
sortd = np.argsort(-np.abs(mean_adv - mean))

senses = utils.get_sensitivities(deltas_path)
# Replace inf with v.large valevalues
senses = np.where(senses >= 1e20, 1e20, senses)
senses = senses[sortd]

mean_sense, std_sense = np.mean(senses, axis=1), np.std(senses, axis=1)

mean     = mean[sortd]
mean_adv = mean_adv[sortd]

plt.errorbar(x, mean_adv, yerr=std_adv, fmt='+', color='gold', ecolor='beige', elinewidth=2, capsize=0, label='Linf')
plt.errorbar(x, mean, yerr=std, fmt='o', color='black', ecolor='lightgray', elinewidth=2, capsize=0, label='natural')
plt.errorbar(x, mean_sense, yerr=std_sense, fmt='d', color='green', ecolor='lightgreen', elinewidth=2, capsize=0, label='delta')

# Plot delta values along with these differences : study possible correlation between delta values and standard activation neuron values

plt.legend(loc="upper left")
plt.xlabel("Neurons (ordered)")
plt.ylabel("Activation and Delta values")
plt.savefig('neuron_deltas.png')
