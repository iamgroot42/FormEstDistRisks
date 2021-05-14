import data_utils
import pandas as pd
import torch.nn as nn
import torch as ch
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
        model = data_utils.BoneModel(1024)
        model.load_state_dict(ch.load(os.path.join(folder_path, mpath)))
        model.eval()
        model = model.cuda()

        _, vacc = utils.validate_epoch(
            val_loader, model, criterion, verbose=False)
        accs.append(vacc)

    return accs


if __name__ == "__main__":
    import sys

    ratio = float(.2)
    loadpath_1 = "/p/adversarialml/as9rw/models_boneage/split_2/0.2"
    loadpath_2 = "/p/adversarialml/as9rw/models_boneage/split_2/0.5"

    batch_size = 256 * 32

    def filter(x): return x["gender"] == 1

    # Get DF
    df_train = pd.read_csv("./data/temp/split_2/train.csv")
    df_val = pd.read_csv("./data/temp/split_2/val.csv")

    # Load features
    features = {}
    features["train"] = ch.load("./data/temp/split_2/features_train.pt")
    features["val"] = ch.load("./data/temp/split_2/features_val.pt")

    # Get data with ratio
    df = utils.heuristic(
        df_val, filter, ratio,
        10000, class_imbalance=1.0, n_tries=300)

    ds = data_utils.BoneWrapper(
        df, df,
        features=features)

    _, val_loader = ds.get_loaders(batch_size, shuffle=False)

    accs_1 = get_accs(val_loader, loadpath_1)
    accs_2 = get_accs(val_loader, loadpath_2)

    plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    plt.savefig("./quick_see.png")