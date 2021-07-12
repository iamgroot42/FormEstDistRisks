from sklearn.ensemble import RandomForestClassifier
from sympy import E
from model_utils import get_model, BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES, PRESERVE_PROPERTIES
import numpy as np
from tqdm import tqdm
import os
import torch as ch
import argparse
from utils import flash_utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 200


def get_models(folder_path, n_models=100):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = get_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_stats(mainmodel, dataloader, mask=None):
    activationCounts = []

    # only purpose is to extract x_te
    for (x_te, y, _) in (dataloader):

        for i in range(7):
        #for i in [0, 1, 5, 6]:
            if mask is not None:
                x_eff = x_te[mask]
            else:
                x_eff = x_te
            # Get activation values for data at x_te[index]
            acts = mainmodel(x_eff.cuda(), latent=i).detach()

            # Count positive activations
            activationCounts.append(ch.sum(acts > 0, 1).cpu().numpy())

        return np.concatenate(activationCounts)


# applies masking in get_stats
def get_acts(loader, models, model_data, ratio, mask=None):
    for model in tqdm(models):
        acts = get_stats(model, loader, mask)
        model_data[0].append(acts)
        model_data[1].append(ratio)

    return np.array(acts)


def extract_and_prepare(loaders, models_1, models_2, mask=None):
    model_data = [[], []]

    for loader in loaders:
        get_acts(loader, models_1, model_data, ratio=args.ratio_1, mask=mask)
        get_acts(loader, models_2, model_data, ratio=args.ratio_2, mask=mask)

    model_data[0] = np.array(model_data[0])
    model_data[1] = np.array(model_data[1])

    return model_data


def main(args):
    # Get victim models
    models_victim_1 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.filter, args.ratio_1))
    models_victim_2 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.filter, args.ratio_2))

    # Prepare data wrappers
    ds_1 = CelebaWrapper(args.filter, float(
        args.ratio_1), "adv", cwise_samples=1e6)
    ds_2 = CelebaWrapper(args.filter, float(
        args.ratio_2), "adv", cwise_samples=1e6)

    loaders = [
        ds_1.get_loaders(batch_size=400, shuffle=False)[1],
        ds_2.get_loaders(batch_size=400, shuffle=False)[1]
    ]

    relevant_acc = []
    for _ in range(args.trials):

        # Load adv models
        total_models = args.total_models
        models_1 = get_models(os.path.join(
            BASE_MODELS_DIR, "adv", args.filter, args.ratio_1), total_models // 2)
        models_2 = get_models(os.path.join(
            BASE_MODELS_DIR, "adv", args.filter, args.ratio_2), total_models // 2)

        # First train classifier using ALL test data
        clf = RandomForestClassifier(n_estimators = 10, max_depth=3)

        # Data that will store x and y values
        model_data = extract_and_prepare(loaders, models_1, models_2)
        x_tr = model_data[0]
        y_tr = (model_data[1] != args.ratio_1)
        clf.fit(x_tr, y_tr)
        train_acc = 100 * clf.score(x_tr, y_tr)
        print("[Using All Data] Train Accuracy", train_acc)

        # Get accuracy with this method on unseen victim models
        model_data = extract_and_prepare(
            loaders, models_victim_1, models_victim_2)
        x_te = model_data[0]
        y_te = (model_data[1] != args.ratio_1)
        print("[Using All Data] Accuracy on unseen models",
              100 * clf.score(x_te, y_te))

        # Then, look at feature importances used by the model
        # And pick the top 100 points
        num_layers = 7 #4
        n_points = len(clf.feature_importances_) // num_layers
        point_wise_importances = []
        for i in range(num_layers):
            point_wise_importances.append(clf.feature_importances_[
                                          n_points * i:n_points * (i+1)])

        # Account all activations for a point in importance calculation
        print(point_wise_importances)
        fis = np.sum(point_wise_importances, 0)
        fis = np.argsort(-fis)[:args.top_k]

        # Knowing which data points are best to be used, repeat process all over again
        # Data that will store x and y values
        clf = RandomForestClassifier(n_estimators = 10, max_depth=3)

        # Data that will store x and y values
        model_data = extract_and_prepare(
            loaders, models_1, models_2, mask=fis)
        x_tr = model_data[0]
        y_tr = (model_data[1] != args.ratio_1)
        clf.fit(x_tr, y_tr)
        train_acc = 100 * clf.score(x_tr, y_tr)
        print("[Using Selected Data] Train Accuracy", train_acc)

        # Get accuracy with this method on unseen victim models
        model_data = extract_and_prepare(
            loaders, models_victim_1, models_victim_2, mask=fis)
        x_te = model_data[0]
        y_te = (model_data[1] != args.ratio_1)

        relevant_acc.append(100 * clf.score(x_te, y_te))

        print("[Using Selected Data] Accuracy on unseen models",
              relevant_acc[-1])

    return relevant_acc


if __name__ == "__main__":
    # Example command: python celeb_acts_metaclassifier.py --filter Male --trials 1 --total_models 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--top_k', default=20, type=int,
                        help="top K datapoints")
    parser.add_argument('--trials', default=100, type=int,
                        help="number of trials")
    parser.add_argument('--total_models', default=40, type=int,
                        help="number of trials")
    args = parser.parse_args()
    flash_utils(args)

    args.ratio_1 = "0.5"
    ratios_try = ["0.0", "0.1", "0.2", "0.3",
                  "0.4", "0.6", "0.7", "0.8", "0.9", "1.0"]
    data = []
    for rt in ratios_try:
        args.ratio_2 = rt
        relevant_accs = main(args)
        for acc in relevant_accs:
            data.append([float(rt), acc])

    columns = [
        r'%s proportion of training data ($\alpha$)' % args.filter,
        "Accuracy (%)"
    ]
    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0], y=columns[1], data=df, color='C0', showfliers=False,)

    # Accuracy range, with space to show good performance
    sns_plot.set(ylim=(40, 101))

    # Add dividing line in centre
    lower, upper = plt.gca().get_xlim()
    midpoint = (lower + upper) / 2
    plt.axvline(x=midpoint, color='black',
                linewidth=1.0, linestyle='--')

    # Make sure axis label not cut off
    plt.tight_layout()

    # Save plot
    sns_plot.figure.savefig(
        "/u/jyc9fyf/celebaModels/activations_meta_estimators_%d.png" % args.total_models)
