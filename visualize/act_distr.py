import numpy as np
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


# stats_path     = "/p/adversarialml/as9rw/cifar10_stats/nat/stats"
# statsl2_path   = "/p/adversarialml/as9rw/cifar10_stats/l2/stats"
# statslinf_path = "/p/adversarialml/as9rw/cifar10_stats/linf/stats"

# stats_path     = "/p/adversarialml/as9rw/binary_stats/nat/stats"
# stats_path     = "/u/as9rw/work/fnb/1e1_1e4_1e-2_16_1/"
stats_path     = "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/"
# statsl2_path   = "/p/adversarialml/as9rw/binary_stats/l2/stats"
# statslinf_path = "/p/adversarialml/as9rw/binary_stats/linf/stats"


(mean, std) = utils.get_stats(stats_path)
x = list(range(mean.shape[0]))

# Normal model
indices = np.argsort(mean)
mean = mean[indices]
std  = std[indices]
plt.errorbar(x, mean, yerr=std, ecolor='turquoise', color='blue', label='normal')

# L-2 model
# (mean, std) = utils.get_stats(statsl2_path)
# indices = np.argsort(mean)
# mean = mean[indices]
# std  = std[indices]
# plt.errorbar(x, mean, yerr=std, ecolor='lightcoral', color='red', label='L-2')

# L-inf model
# (mean, std) = utils.get_stats(statslinf_path)
# indices = np.argsort(mean)
# mean = mean[indices]
# std  = std[indices]
# plt.errorbar(x, mean, yerr=std, ecolor='lightgreen', color='green', label='L-inf')


plt.title('CIFAR-10 model (custom reg)')
plt.legend()
plt.savefig("just_acts.png")

