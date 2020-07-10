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

from matplotlib import rc
rc('text', usetex=True)

nat_rates  = get_success_rates("nat_again")
l2_rates   = get_success_rates("l2_vgg_linf")
linf_rates = get_success_rates("linf_vgg_linf")
custom_rates = get_success_rates("custom_vgg_linf")

# data = {'$L_{2}$ robust': l2_rates, '$L_{\infty}$ robust': linf_rates, 'Standard': nat_rates, 'Sensitivity': custom_rates}
# df = pd.DataFrame(data, columns = ['Standard', '$L_{2}$ robust', '$L_{\infty}$ robust', 'Sensitivity'])
# ax = sns.lineplot(data=df, linewidth=2.5)

data = {'$L_{2}$ robust': l2_rates, '$L_{\infty}$ robust': linf_rates, 'Standard': nat_rates}
df = pd.DataFrame(data, columns = ['Standard', '$L_{2}$ robust', '$L_{\infty}$ robust'])
ax = sns.lineplot(data=df, linewidth=2.5, palette=["#1f77bf", "#d62728", "#ff7f0e"])

ax.axhline(100,   ls='--', c='#1f77bf')
ax.axhline(53.59, ls='--', c='#ff7f0e')
ax.axhline(71.29, ls='--', c='#d62728')

ax.set_xlabel("Neurons Ordered by Increasing $\Delta(i, x)$", fontsize=15)
ax.set_ylabel("Attack Success Rates (\%)", fontsize=15)
ax.legend(fontsize=13, loc='upper left', ncol=3, bbox_to_anchor=(0, 0.95), labelspacing=0.2, borderpad=0.2)
ax.set(xlim=(0,512))
fig = ax.get_figure()
fig.savefig("linf_attack.png")