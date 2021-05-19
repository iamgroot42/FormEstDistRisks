from model_utils import load_model, BASE_MODELS_DIR
from data_utils import BoneWrapper, get_df, get_features
import numpy as np
import utils
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_activations(val_loader, folder_path, latent=0):
    acts_all_0, acts_all_1 = [], []

    for mpath in tqdm(os.listdir(folder_path)):
        model = load_model(os.path.join(folder_path, mpath))
        model = model.cuda()

        acts_0, acts_1 = [], []
        for (x, _, gender) in val_loader:
            la = model(x.cuda(), latent=latent).detach().cpu().numpy()
            activations = np.sum(la > 0, 1)

            acts_0.append(activations[gender == 0])
            acts_1.append(activations[gender == 1])

        acts_0 = np.concatenate(acts_0)
        acts_1 = np.concatenate(acts_1)
        acts_all_0.append(acts_0)
        acts_all_1.append(acts_1)

    return np.array(acts_all_0), np.array(acts_all_1)


if __name__ == "__main__":
    import sys

    loadpath_1 = os.path.join(BASE_MODELS_DIR, sys.argv[1], sys.argv[2])
    loadpath_2 = os.path.join(BASE_MODELS_DIR, sys.argv[1], sys.argv[3])

    batch_size = 256 * 32

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    df = utils.heuristic(
        df_val, filter, 0.5,
        10000, class_imbalance=1.0, n_tries=300)

    # Prepare dataloader
    ds = BoneWrapper(
        df, df, features=features)
    _, val_loader = ds.get_loaders(batch_size, shuffle=False)

    latent = 1

    # Get activations
    acts_1_0, acts_1_1 = get_activations(val_loader, loadpath_1, latent)
    acts_2_0, acts_2_1 = get_activations(val_loader, loadpath_2, latent)

    # Calculate N values for first model set
    print("First model set:")
    mean_1, std_1 = np.mean(acts_1_0, 1), np.std(acts_1_0, 1)
    mean_2, std_2 = np.mean(acts_1_1, 1), np.std(acts_1_1, 1)
    N_vals = np.abs(mean_1 - mean_2) / (std_1 + std_2)
    print("Range: [%.3f, %.3f] | Median: %.3f" %
          (np.min(N_vals), np.max(N_vals), np.median(N_vals)))

    print()

    # Calculate N values for second model set
    print("Second model set:")
    mean_1, std_1 = np.mean(acts_2_0, 1), np.std(acts_2_0, 1)
    mean_2, std_2 = np.mean(acts_2_1, 1), np.std(acts_2_1, 1)
    N_vals = np.abs(mean_1 - mean_2) / (std_1 + std_2)
    print("Range: [%.3f, %.3f] | Median: %.3f" %
          (np.min(N_vals), np.max(N_vals), np.median(N_vals)))

    # Look at activation histogram for two random models from both folders
    # trend_1 = acts_1[2]
    # trend_2 = acts_2[2]

    # plt.hist(trend_1, bins=50, label="G_0", color='C0')
    # plt.hist(trend_2, bins=50, label="G_0", color='C1')

    # plt.savefig("./act_trends.png")


# 0.2 v/s 0.5
# LATENT 0
# Range: [0.000, 0.152] | Median: 0.071
# Range: [0.001, 0.149] | Median: 0.057

# LATENT 1
# Range: [0.001, 0.098] | Median: 0.065
# Range: [0.000, 0.096] | Median: 0.044

# Conclusion: look at latent 0, activation=0


# 0.4 v/s 0.5
# LATENT 0
# Range: [0.001, 0.125] | Median: 0.064
# Range: [0.000, 0.137] | Median: 0.061

# LATENT 1
