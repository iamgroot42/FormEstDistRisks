from model_utils import load_model, get_model_folder_path
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


def get_models(folder_path, n_models=1000):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_accs(val_loader, models):
    accs = []

    criterion = nn.BCEWithLogitsLoss().cuda()
    for model in tqdm(models):
        model = model.cuda()

        vloss, vacc = utils.validate_epoch(
            val_loader, model, criterion, verbose=False)
        accs.append(vacc)
        # accs.append(vloss)

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

    ratio_1 = sys.argv[1]
    ratio_2 = sys.argv[2]
    batch_size = 256 * 32

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    df_1 = utils.heuristic(
        df_val, filter, float(ratio_1),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    df_2 = utils.heuristic(
        df_val, filter, float(ratio_2),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    # Prepare data loaders
    ds_1 = BoneWrapper(
        df_1, df_1, features=features)
    ds_2 = BoneWrapper(
        df_2, df_2, features=features)
    loaders = [
        ds_1.get_loaders(batch_size, shuffle=False)[1],
        ds_2.get_loaders(batch_size, shuffle=False)[1]
    ]

    # Load victim models
    models_victim_1 = get_models(get_model_folder_path("victim", ratio_1))
    models_victim_2 = get_models(get_model_folder_path("victim", ratio_2))

    # Load adv models
    # total_models = 100
    # models_1 = get_models(get_model_folder_path(
    #     "adv", ratio_1), total_models // 2)
    # models_2 = get_models(get_model_folder_path(
    #     "adv", ratio_2), total_models // 2)

    z_vals = []
    allaccs_1, allaccs_2 = [], []
    for loader in loaders:
        # accs_1 = get_accs(loader, models_1)
        # accs_2 = get_accs(loader, models_2)

        # # Look at [0, 100]
        # accs_1 *= 100
        # accs_2 *= 100

        # # Calculate Z value
        # m1, v1 = np.mean(accs_1), np.var(accs_1)
        # m2, v2 = np.mean(accs_2), np.var(accs_2)
        # mean_new = np.abs(m1 - m2)
        # var_new = (v1 + v2) / total_models
        # Z = mean_new / np.sqrt(var_new)

        # prob = calculate_same_prob(mean_new, var_new)
        # print(prob)

        # print("Mean-1: %.3f, Mean-2: %.3f" % (m1, m2))
        # print("Var-1: %.3f, Var-2: %.3f" % (v1, v2))
        # print("Number of samples: %d" % total_models)
        # z_vals.append(Z)

        # tracc, threshold = utils.find_threshold_acc(accs_1, accs_2)
        # print("[Adversary] Threshold based accuracy: %.2f at threshold %.2f" %
        #       (100 * tracc, threshold))

        # Compute accuracies on this data for victim
        accs_victim_1 = get_accs(loader, models_victim_1)
        accs_victim_2 = get_accs(loader, models_victim_2)

        # Look at [0, 100]
        accs_victim_1 *= 100
        accs_victim_2 *= 100

        # Threshold based on adv models
        # combined = np.concatenate((accs_victim_1, accs_victim_2))
        # classes = np.concatenate(
        #     (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
        # specific_acc = utils.get_threshold_acc(combined, classes, threshold)
        # print("[Victim] Accuracy at specified threshold: %.2f" %
        #       (100 * specific_acc))

        # Collect all accuracies for basic baseline
        allaccs_1.append(accs_victim_1)
        allaccs_2.append(accs_victim_2)

    print("Z values:", z_vals)

    # Basic baseline: look at model performance on test sets from both G_b
    # Predict b for whichever b it is higher
    allaccs_1 = np.array(allaccs_1).T
    allaccs_2 = np.array(allaccs_2).T

    preds_1 = (allaccs_1[:, 0] > allaccs_1[:, 1])
    preds_2 = (allaccs_2[:, 0] < allaccs_2[:, 1])
    basic_baseline_acc = (np.mean(preds_1) + np.mean(preds_2)) / 2
    print("Basic baseline accuracy: %.3f" % (100 * basic_baseline_acc))

    # plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    # plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    # plt.savefig("./quick_see.png")


# Z values
# 0.2 (30.509, 14.151)
# 0.3 (10.576, 6.479)
# 0.4 (4.017, 1.33)
# 0.6 (?, 6.435)
# 0.7 (3.951, 20.942)
# 0.8 (1.008, 30.177)


# Z values with 100 models, 10 runs:
# 0.2: (5.78, 4.94), (4.89, 10.83), (4.18, 7.99), (2.29, 9.47), (7.38, 7.31), (3.31, 9.25), (6.18, 9.25), (4.13, 10.13), (1.85, 6.88), (6.08, 6.93)
# 0.2: 5.78, 10.83, 7.99, 9.47, 7.31, 9.25, 9.25, 10.13, 6.88, 6.93

# 0.3: (5.49, 0.97), (4.10, 0.87), (2.59, 0.47), (5.92, 2.14), (4.4, 1.24), (6.26, 2.78), (5.03, 0.73), (7.32, 1.26), (5.82, 4.29), (4.55, 2.68)
# 0.3: 5.49, 4.10, 2.59, 5.92, 4.4, 6.26, 5.03, 7.32, 5.82, 4.55

# 0.4: (1.56, 1.18), (1.04, 2.17), (0.75, 3.23), (3.75, 0.29), (0.65, 0.16), (0.37, 1.01), (3.27, 2.19), (0.71, 0.27), (2.12, 1.08), (1.22, 0.33)
# 0.4: 1.56, 2.17, 3.23, 3.75, 0.65, 1.01, 3.27, 0.71, 2.12, 1.22

# 0.6: (1.67, 2.36), (1.18, 3.78), (2.31, 5.38), (0.54, 1.45), (3.17, 3.55), (2.01, 4.50), (0.49, 0.05), (3.57, 4.01), (2.09, 3.22), (0.84, 0.55)
# 0.6: 2.36, 3.78, 5.38, 1.45, 3.55, 4.5, 0.49, 4.01, 3.22, 0.84

# 0.7: (3.30, 4.76), (1.10, 6.72), (2.19, 4.39), (0.50, 4.32), (3.49, 2.51), (0.86, 5.87), (1.30, 2.73), (0.13, 5.62), (2.28, 6.52), (1.08, 5.93)
# 0.7: 4.70, 6.72, 4.39, 4.32, 3.49, 5.87, 2.73, 5.62, 6.52, 5.93

# 0.8: (0.65, 13.19), (4.25, 8.7), (4.67, 15.67), (2.79, 8.81), (0.30, 7.28), (2.96, 10.81), (3.19, 10.21), (2.62, 10.0), (1.42, 7.67), (2.6, 8.75)
# 0.8: 13.19, 8.7, 15.67, 8.81, 7.28, 10.81, 10.21, 10.0, 7.67, 8.75

# Accuracies with 100 models, 10 runs:
# 0.2: 64.35, 61.35, 60.85, 60.6, 57.10, 58.10, 57.5, 60.1, 62.1, 63.6
# 0.3: 58.8, 52.3, 50.9, 55.0, 53.25, 58.05, 51.3, 54.3, 54.8, 55.5
# 0.4: 51.35, 52.5, 53.1, 53.6, 50.2, 51.95, 53.15, 52.85, 50.35, 51.25
# 0.6: 52.5, 55.1, 55.15, 57.25, 51.1, 52.1, 50.45, 53.75, 53.0, 50.35
# 0.7: 56.75, 60.6, 61.35, 58.3, 51.55, 56.2, 51.5, 58.7, 55.6, 54.1
# 0.8: 69.45, 65.75, 69.0, 67.1, 63.8, 64.45, 56.4, 64.65, 57.25, 69.65
