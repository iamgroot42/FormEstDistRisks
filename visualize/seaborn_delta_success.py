import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

def get_success_rates(path):
	rates = []
	with open(path, 'r') as f:
		for line in f:
			number = float(line.rstrip('\n').split(' : ')[1].split(' %')[0])
			rates.append(number)
	return np.array(rates)


# nat_rates  = get_success_rates("/u/as9rw/work/fnb/nat_vgg_l2")
nat_rates  = get_success_rates("/u/as9rw/work/fnb/achieving_nat")
# nat_rates  = get_success_rates("/u/as9rw/work/fnb/nat_vgg_linf")
# l2_rates   = get_success_rates("/u/as9rw/work/fnb/l2_vgg_l2")
# linf_rates = get_success_rates("/u/as9rw/work/fnb/linf_vgg_l2")
# l2_rates   = get_success_rates("/u/as9rw/work/fnb/l2_vgg_linf")
linf_rates = get_success_rates("/u/as9rw/work/fnb/achieving_linf")
custom_rates = get_success_rates("/u/as9rw/work/fnb/achieving_custom")

# data = {'$L_{2}$ robust': l2_rates, '$L_{\infty}$ robust': linf_rates, 'Standard': nat_rates, 'Sensitivity': custom_rates}
# df = pd.DataFrame(data, columns = ['Standard', '$L_{2}$ robust', '$L_{\infty}$ robust', 'Sensitivity'])
# ax = sns.lineplot(data=df, linewidth=2.5)

data = {'Sensitivity Training': custom_rates, '$L_{\infty}$ robust': linf_rates, 'Standard': nat_rates}
df = pd.DataFrame(data, columns = ['Standard', 'Sensitivity Training', '$L_{\infty}$ robust'])
ax = sns.lineplot(data=df, linewidth=2.5, palette=["#1f77bf", "#d62728", "#ff7f0e"])

ax.set_xlabel("Neurons Ordered by Decreasing Sensitivity", fontsize=15)
ax.set_ylabel("$\Delta$ Success Rate (%)", fontsize=15)
ax.legend(fontsize=13)
ax.set(xlim=(0,512))
fig = ax.get_figure()
fig.savefig("delta_success_linf.png")