from model_utils import get_models_path, load_model
from data_utils import CensusWrapper
import numpy as np
from tqdm import tqdm
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_accs(x_te, y_te, folder_path):
    accs = []
    for mpath in tqdm(os.listdir(folder_path)):
        model = load_model(os.path.join(folder_path, mpath))

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


def get_threshold_acc(X, Y, threshold):
    # Rule-1: everything above threshold is 1 class
    acc_1 = np.mean((X >= threshold) == Y)
    # Rule-2: everything below threshold is 1 class
    acc_2 = np.mean((X <= threshold) == Y)
    return max(acc_1, acc_2)


def find_threshold_acc(accs_1, accs_2, granularity=0.1):
    lower, upper = np.min(accs_1), np.max(accs_2)
    combined = np.concatenate((accs_1, accs_2))
    classes = np.concatenate((np.zeros_like(accs_1), np.ones_like(accs_2)))
    best_acc = 0.0
    best_threshold = 0
    while lower < upper:
        best_of_two = get_threshold_acc(combined, classes, lower)
        if best_of_two > best_acc:
            best_threshold = lower
            best_acc = best_of_two

        lower += granularity

    return best_acc, best_threshold


if __name__ == "__main__":
    import sys

    ratio_1 = sys.argv[1]
    ratio_2 = sys.argv[2]
    ratio = float(sys.argv[3])

    focus_prop = "sex"

    path_1 = get_models_path(focus_prop, "victim", ratio_1)
    path_2 = get_models_path(focus_prop, "victim", ratio_2)

    # Prepare data wrapper
    ds = CensusWrapper(
        filter_prop=focus_prop, ratio=ratio, split="adv")

    # Fetch test data
    _, (x_te, y_te), cols = ds.load_data()
    y_te = y_te.ravel()

    # Load models and get accuracies
    accs_1 = get_accs(x_te, y_te, path_1)
    accs_2 = get_accs(x_te, y_te, path_2)

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

    tracc, threshold = find_threshold_acc(accs_1, accs_2)
    print("Threshold based accuracy: %.2f at threshold %.2f" %
          (100 * tracc, threshold))

    # Threshold based on adv models
    my_threshold = 85.76
    combined = np.concatenate((accs_1, accs_2))
    classes = np.concatenate((np.zeros_like(accs_1), np.ones_like(accs_2)))
    specific_acc = get_threshold_acc(combined, classes, my_threshold)
    print("Accuracy at specified threshold: %.2f" % (100 * specific_acc))

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

