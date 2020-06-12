# import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import matplotlib.pyplot as plt

nat_latent  = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], [94.62,94.10,93.35,91.83,89.79,90.05,89.88,89.24,90.01,98.92,99.94,99.68,98.5,99.15,99.79]]
linf_latent = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14], [47.76,47.52,47.19,46.71,44.94,44.31,44.00,43.36,43.59,47.74,48.90,47.51,48.05,46.66,44.81]]

delta_nat  = [[3,4,5,6,7,8,9,10,11,12,13,14], [3.7, 16.19, 19.84, 41.67, 32.53, 51.75, 97.82, 98.72, 98.29, 99.85, 98.92, 87.1]]
delta_linf = [[3,4,5,6,7,8,9,10,11,12,13,14], [0.97, 2.51, 2.88, 9.8, 23.17, 15.74, 31.30, 39.62, 39.13, 38.95, 38.63, 36.47]]

plt.plot(nat_latent[0], nat_latent[1], label="Latent-Space [Standard]", c='C0')
plt.plot(delta_nat[0], delta_nat[1], label="16-NS Delta Attack [Standard]", c='C0', linestyle=':')
plt.hlines(100, 0, 14, label='PGD [Standard]', colors=['C0'], linestyles='dashed')


plt.plot(linf_latent[0], linf_latent[1], label="Latent-Space [Linf]", c='C1')
plt.plot(delta_linf[0], delta_linf[1], label="16-NS Delta Attack [Linf]", c='C1', linestyle=':')
plt.hlines(50.20, 0, 14, label='PGD [Linf]', colors=['C1'], linestyles='dashed')

plt.title("Attack Success Rates for Different Approaches")

plt.xlabel('Layers')
plt.ylabel('Attack Success Rate (%)')
plt.legend()

plt.savefig("generic_graph.png")
