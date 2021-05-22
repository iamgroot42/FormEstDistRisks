from model_utils import get_models_path, load_model
from data_utils import CensusWrapper
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import norm
from utils import get_z_value, get_threshold_acc, find_threshold_acc
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


def get_accs(data, models):
    x_te, y_te = data
    accs = []
    for model in tqdm(models):
        vacc = model.score(x_te, y_te)
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

    ratio_1 = sys.argv[1]
    ratio_2 = sys.argv[2]
    focus_prop = "sex"

    # Get victim models
    models_victim_1 = get_models(
        get_models_path(focus_prop, "victim", ratio_1))
    models_victim_2 = get_models(
        get_models_path(focus_prop, "victim", ratio_2))

    # Load adv models
    total_models = 100
    models_1 = get_models(get_models_path(
        focus_prop, "victim", ratio_1), total_models // 2)
    models_2 = get_models(get_models_path(
        focus_prop, "victim", ratio_2), total_models // 2)

    # Prepare data wrappers
    ds_1 = CensusWrapper(
        filter_prop=focus_prop,
        ratio=float(ratio_1), split="adv")
    ds_2 = CensusWrapper(
        filter_prop=focus_prop,
        ratio=float(ratio_2), split="adv")

    # Fetch test data from both ratios
    _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
    _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
    y_te_1 = y_te_1.ravel()
    y_te_2 = y_te_2.ravel()

    # Iterate through data from both distrs
    z_vals = []
    loaders = [(x_te_1, y_te_1), (x_te_2, y_te_2)]
    for loader in loaders:
        # Load models and get accuracies
        accs_1 = get_accs(loader, models_1)
        accs_2 = get_accs(loader, models_2)

        # Look at [0, 100]
        accs_1 *= 100
        accs_2 *= 100

        # Calculate N value
        m1, v1 = np.mean(accs_1), np.var(accs_1)
        m2, v2 = np.mean(accs_2), np.var(accs_2)
        mean_new = np.abs(m1 - m2)
        var_new = (v1 + v2) / total_models
        Z = get_z_value(accs_1, accs_2)

        prob = calculate_same_prob(mean_new, var_new)
        print(prob)

        print("Mean-1: %.3f, Mean-2: %.3f" % (m1, m2))
        print("Var-1: %.3f, Var-2: %.3f" % (v1, v2))
        print("Z value: %.3f" % Z)
        z_vals.append(Z)

        tracc, threshold = find_threshold_acc(accs_1, accs_2)
        print("[Adversary] Threshold based accuracy: %.2f at threshold %.2f" %
              (100 * tracc, threshold))

        # Compute accuracies on this data for victim
        accs_victim_1 = get_accs(loader, models_victim_1)
        accs_victim_2 = get_accs(loader, models_victim_2)

        # Look at [0, 100]
        accs_victim_1 *= 100
        accs_victim_2 *= 100

        # Threshold based on adv models
        combined = np.concatenate((accs_victim_1, accs_victim_2))
        classes = np.concatenate(
            (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
        specific_acc = get_threshold_acc(combined, classes, threshold)
        print("[Victim] Accuracy at specified threshold: %.2f" %
              (100 * specific_acc))

    print("Z values:", z_vals)

    plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    plt.savefig("./quick_see.png")


# Z values
# race: (20.943, 6.618) on 1000 models, (8.198, 2.947) on 100 models
# sex: (11.765, 1.284) on 1000 models, (4.116, 0.088) on 100 models

# RACE
# Using adv data, threshold computed on adv models
# Tested on victim models
# 75.85% accuracy JUST BASED ON accuracy
# 80.9% accuracy when just 100 models per class used!
# So not exactly an interesting property

# SEX
# 51.65% accuracy based on accuracy
# So indeed, interesting property


# Race
# Z values
# (10.92, 3.32), (13.6, 5.3), (9.53, 1.49), (13.2, 5.3), (11.75, 3.3), (11.57, 6.71), (12.52, 5.21), (10.7, 5.86), (4.76, 2.89), (11.78, 7.58)
# 10.92, 13.6, 9.53, 13.2, 11.75, 11.57, 12.52, 10.7, 4.76, 11.78
# Accuracies
# 83.26, 80.85, 81.15, 82.25, 81.9, 81.9, 80.9, 82.3, 78.7, 82.3


# Sex
# Z values
# (5.9, 5.4), (3.55, 3.023), (2.17, 2.14), (4.216, 3.058), (6.77, 5.69), (3.06, 2.55), (3.93, 3.24), (4.30, 3.85), (3.35, 2.83), (6.98, 5.67)
# 5.9, 3.55, 2.17, 4.22, 6.77, 3.06, 3.93, 4.30, 3.35, 6.98
# Accuracies
# 63.85, 62.2, 63.95, 64.5, 61.0, 64.85, 63.05, 61.9, 63.45, 58.2, 