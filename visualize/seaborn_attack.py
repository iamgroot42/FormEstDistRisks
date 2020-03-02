import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style="whitegrid")


import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

rs = np.random.RandomState(365)
values = rs.randn(365, 4).cumsum(axis=0)
dates = pd.date_range("1 1 2016", periods=365, freq="D")
data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
data = data.rolling(7).mean()

def get_success_rates(path):
	rates = []
	with open(path, 'r') as f:
		for line in f:
			number = float(line.rstrip('\n').split(' : ')[1].split(' %')[0])
			rates.append(number)
	return np.array(rates)


nat_rates  = get_success_rates("/u/as9rw/work/fnb/nat_vgg_l2")
# l2_rates   = get_success_rates("/u/as9rw/work/fnb/l2_vgg_l2")
# linf_rates = get_success_rates("/u/as9rw/work/fnb/linf_vgg_l2")
l2_rates   = get_success_rates("/u/as9rw/work/fnb/l2_vgg_linf")
linf_rates = get_success_rates("/u/as9rw/work/fnb/linf_vgg_linf")

# data = {'$L_{2}$ robust': nat_rates, '$L_{\infty}$ robust': linf_rates, 'Normal': l2_rates}
# df = pd.DataFrame(data, columns = ['Normal', '$L_{2}$ robust', '$L_{\infty}$ robust'])
data = {'$L_{2}$ robust': l2_rates, '$L_{\infty}$ robust': linf_rates}
df = pd.DataFrame(data, columns = ['$L_{2}$ robust', '$L_{\infty}$ robust'])
ax = sns.lineplot(data=df, palette="tab10", linewidth=2.5)
ax.set(xlabel='Neurons Ordered by Decreasing Sensitivity', ylabel='Attack Succes Rates (%)')

fig = ax.get_figure()
fig.savefig("attack_anal.png")