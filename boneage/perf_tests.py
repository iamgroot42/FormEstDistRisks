from model_utils import load_model
from data_utils import BoneWrapper, get_df, get_features
import torch.nn as nn
import numpy as np
import utils
from tqdm import tqdm
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_accs(val_loader, folder_path):
    accs = []

    criterion = nn.BCEWithLogitsLoss().cuda()
    for mpath in tqdm(os.listdir(folder_path)):
        model = load_model(os.path.join(folder_path, mpath))
        model = model.cuda()

        _, vacc = utils.validate_epoch(
            val_loader, model, criterion, verbose=False)
        accs.append(vacc)

    return np.array(accs)


def calculate_same_prob(mu, sigma):
    # We want probability that means are indeed similar
    # Return prob of them being within +- limit of each other
    limit = 0.5
    upper = norm.cdf(limit, loc=mu, scale=sigma)
    lower = norm.cdf(-limit, loc=mu, scale=sigma)
    return upper - lower


if __name__ == "__main__":
    import sys

    loadpath_1 = sys.argv[1]
    loadpath_2 = sys.argv[2]
    ratio = float(sys.argv[3])

    batch_size = 256 * 32

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    df = utils.heuristic(
        df_val, filter, ratio,
        10000, class_imbalance=1.0, n_tries=300)

    ds = BoneWrapper(
        df, df, features=features)

    _, val_loader = ds.get_loaders(batch_size, shuffle=False)

    accs_1 = get_accs(val_loader, loadpath_1)
    accs_2 = get_accs(val_loader, loadpath_2)

    # Look at [0, 100]
    accs_1 *= 100
    accs_2 *= 100

    # Calculate N value
    m1, v1 = np.mean(accs_1), np.var(accs_1)
    m2, v2 = np.mean(accs_2), np.var(accs_2)

    n_samples = len(accs_1)
    mean_new = np.abs(m1 - m2)
    var_new = (v1 + v2) / n_samples
    Z = mean_new / np.sqrt(var_new)

    prob = calculate_same_prob(mean_new, var_new)
    print(prob)

    print("Mean-1: %.3f, Mean-2: %.3f" % (m1, m2))
    print("Var-1: %.3f, Var-2: %.3f" % (v1, v2))
    print("Number samples: %d" % n_samples)

    print("Z value: %.3f" % Z)

    plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    plt.savefig("./quick_see.png")


# Z values
# 0.2 (30.509, 14.151)
# 0.3 (10.576, 6.479)
# 0.4 (4.017, 1.33)
# 0.6 (?, 6.435)
# 0.7 (3.951, 20.942)
# 0.8 (1.008, 30.177)
