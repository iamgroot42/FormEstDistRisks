from model_utils import get_models_path, load_model
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_z_value, get_threshold_acc, find_threshold_acc, flash_utils
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2")
    args = parser.parse_args()
    flash_utils(args)

    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))

    # Load adv models
    total_models = 100
    models_1 = get_models(get_models_path(
        args.filter, "victim", args.ratio_1), total_models // 2)
    models_2 = get_models(get_models_path(
        args.filter, "victim", args.ratio_2), total_models // 2)

    # Prepare data wrappers
    ds_1 = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.ratio_1), split="adv")
    ds_2 = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.ratio_2), split="adv")

    # Fetch test data from both ratios
    _, (x_te_1, y_te_1), _ = ds_1.load_data(custom_limit=10000)
    _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
    y_te_1 = y_te_1.ravel()
    y_te_2 = y_te_2.ravel()

    # Iterate through data from both distrs
    z_vals, f_accs = [], []
    loaders = [(x_te_1, y_te_1), (x_te_2, y_te_2)]
    allaccs_1, allaccs_2 = [], []
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

        print("Mean-1: %.3f, Mean-2: %.3f" % (m1, m2))
        print("Var-1: %.3f, Var-2: %.3f" % (v1, v2))
        print("Z value: %.3f" % Z)
        z_vals.append(Z)

        tracc, threshold = find_threshold_acc(accs_1, accs_2, granularity=0.01)
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
        f_accs.append(100 * specific_acc)

        # Collect all accuracies for basic baseline
        allaccs_1.append(accs_victim_1)
        allaccs_2.append(accs_victim_2)

    print("Z values:", z_vals)

    # Basic baseline: look at model performance on test sets from both G_b
    # Predict b for whichever b it is higher
    allaccs_1 = np.array(allaccs_1)
    allaccs_2 = np.array(allaccs_2)

    preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
    preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])

    basic_baseline_acc = (np.mean(preds_1) + np.mean(preds_2)) / 2
    print("[Victim] Accuracy on chosen Z value: %.2f" %
          f_accs[np.argmin(z_vals)])
    print("Baseline for M_0: %.3f, Baseline for M_1: %.3f" %
          (100 * np.mean(preds_1), 100 * np.mean(preds_2)))
    print("Basic baseline accuracy: %.3f" % (100 * basic_baseline_acc))

    plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    plt.savefig("./quick_see.png")
