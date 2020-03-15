import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

from labellines import labelLines

neurons = np.array([0, 500, 1000, 1200, 1400, 1600, 1650, 1660, 1665], dtype=np.float32)
neurons *= 100/1669
accs    = np.array([84.57, 84.57, 84.58, 84.56, 83.88, 79.72, 63.01, 29.99, 10])
srs     = np.array([46.15, 46.15, 46.1, 46.13, 45.9, 47.92, 58.35, 80.39, 90])

f  = interp1d(neurons, accs, kind='linear')
f2 = interp1d(neurons, srs, kind='linear')

granular = list(range(0,1670, 5))
plt.plot(neurons, accs, 'o', color='green')
plt.plot(neurons, accs, label='Accuracy (%)', color='green')
plt.plot(neurons, srs, 'o', color='red')
plt.plot(neurons, srs, label='Attack Error Rate (%)', color='red')

# plt.errorbar(neurons, srs, label='Attack Succes Rate')
plt.grid(True)
labelLines(plt.gca().get_lines(), align=False, fontsize=10, backgroundcolor=(1.0, 1.0, 1.0, 0.75), xvals=[16, 35])
# plt.legend()
plt.xticks(np.arange(0, 100, step=10))
plt.xlabel('Neurons Dropped (%)', fontsize=15)
plt.savefig("powerpoint_copy.png")