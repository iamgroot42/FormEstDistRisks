from model_utils import load_model
from data_utils import BoneWrapper, get_df, get_features
import torch.nn as nn
import numpy as np
import utils
from tqdm import tqdm
import os

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

    return accs


if __name__ == "__main__":
    import sys

    ratio = float(sys.argv[1])
    loadpath_1 = sys.argv[2]
    loadpath_2 = sys.argv[3]

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

    plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    plt.savefig("./quick_see.png")
