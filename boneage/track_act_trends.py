from model_utils import load_model, BASE_MODELS_DIR
from data_utils import BoneWrapper, get_df, get_features
import numpy as np
import utils
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_activations(val_loader, folder_path):
    acts_all_0, acts_all_1 = [], []
    latents = [0, 1]

    for mpath in tqdm(os.listdir(folder_path)[:10]):
        model = load_model(os.path.join(folder_path, mpath))
        model = model.cuda()

        acts_0, acts_1 = [], []
        for (x, _, gender) in val_loader:
            temp_0, temp_1 = [], []
            for latent in latents:
                la = model(x.cuda(), latent=latent).detach().cpu().numpy()
                activations = np.sum(la > 0, 1)
                temp_0.append(activations[gender == 0])
                temp_1.append(activations[gender == 1])

            acts_0.append(temp_0)
            acts_1.append(temp_1)

        acts_0 = np.array(acts_0)
        acts_1 = np.array(acts_1)
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

    # Get data with ratios
    df_1 = utils.heuristic(
        df_val, filter, float(sys.argv[2]),
        200, class_imbalance=1.0, n_tries=300)
    df_2 = utils.heuristic(
        df_val, filter, float(sys.argv[3]),
        200, class_imbalance=1.0, n_tries=300)

    # Prepare dataloaders
    ds_1 = BoneWrapper(
        df_1, df_1, features=features)
    _, val_loader_1 = ds_1.get_loaders(batch_size, shuffle=False)
    ds_2 = BoneWrapper(
        df_2, df_2, features=features)
    _, val_loader_2 = ds_2.get_loaders(batch_size, shuffle=False)

    # Get activations
    acts_1_0, acts_1_1 = get_activations(val_loader_1, loadpath_1)
    acts_2_0, acts_2_1 = get_activations(val_loader_2, loadpath_1)

    # Look at activation histogram for two random models from both folders
    trend_1 = np.squeeze(acts_1_0[6], 0).T
    trend_2 = np.squeeze(acts_2_0[6], 0).T

    for te in trend_1:
        plt.plot([1, 2], te + 0.1, color='C0', alpha=0.4)
    for te in trend_2:
        plt.plot([1, 2], te - 0.1, color='C1', alpha=0.4)

    # plt.hist(trend_1, bins=20, label="G_0", color='C0', alpha=0.5)
    # plt.hist(trend_2, bins=20, label="G_0", color='C1', alpha=0.5)
    # plt.legend()
    plt.savefig("./act_trends_1.png")
    exit(0)

    # Get activations
    plt.clf()

    # Look at activation histogram for two random models from both folders
    trend_1 = acts_1_1[6]
    trend_2 = acts_2_1[6]

    plt.hist(trend_2, bins=20, label="G_0", color='C1', alpha=0.5)
    plt.hist(trend_1, bins=20, label="G_0", color='C0', alpha=0.5)
    plt.legend()
    plt.savefig("./act_trends_2.png")
