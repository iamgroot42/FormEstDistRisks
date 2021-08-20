import numpy as np


def get_generated_stats():
    layers = {
        0: [

        ],
        1: [
  
        ],
    }
    return layers


def get_nat_stats():
    layers = {
        0: [
            [55, 60, 60],
            [55, 55, 55],
            [60, 55, 55],
            [60, 60, 60],
            [65, 65, 65],
            [50, 55, 60]
        ],
        1: [
            [55, 55, 55],
            [65, 65, 65],
            [55, 60, 55],
            [60, 60, 60],
            [50, 50, 50],
            [55, 55, 55]
        ],
    }
    return layers


info = get_nat_stats()
for k, v in info.items():
    v = np.mean(v, 1)
    print("Mean: %.2f | Max: %.2f" % (np.mean(v), np.max(v)))
