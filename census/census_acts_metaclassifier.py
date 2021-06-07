from model_utils import get_models_path, load_model
from sklearn.ensemble import RandomForestClassifier
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES, PROPERTY_FOCUS
import model_utils
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import flash_utils
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import model_utils

mpl.rcParams['figure.dpi'] = 200


def get_models(folder_path, n_models=100):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_acts(data, models, model_data, ratio, mask=None):
    x_te, _ = data
    if mask is not None:
        x_te = x_te[mask]
    for model in tqdm(models):
        # Will represent the activation count corresponding to each layer
        acts = model_utils.layer_output(x_te, model, layer=3, get_all=True)
        act_counts = [np.sum(x > 0, 1) for x in acts]
        model_data[0].append(np.concatenate(act_counts))
        model_data[1].append(ratio)


def extract_and_prepare(loaders, models_1, models_2, mask=None):
    model_data = [[], []]

    for loader in loaders:
        # Load models and get/plot activations
        get_acts(loader, models_1, model_data, ratio=args.ratio_1, mask=mask)
        get_acts(loader, models_2, model_data, ratio=args.ratio_2, mask=mask)

    model_data[0] = np.array(model_data[0])
    model_data[1] = np.array(model_data[1])

    return model_data


def main(args):
    # Get victim models
    models_victim_1 = get_models(
        get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
        get_models_path(args.filter, "victim", args.ratio_2))

    # Prepare data wrappers
    ds = CensusWrapper(
        filter_prop=args.filter,
        ratio=float(args.ratio_1), split="adv")

    # Fetch test data from both ratios
    _, (x_te, y_te), _ = ds.load_data(custom_limit=10000)
    y_te = y_te.ravel()

    # Iterate through data from both distrs
    loaders = [(x_te, y_te)]

    relevant_acc = []
    for _ in range(args.trials):

        # Load adv models
        total_models = args.num_models
        models_1 = get_models(get_models_path(
            args.filter, "victim", args.ratio_1), total_models // 2)
        models_2 = get_models(get_models_path(
            args.filter, "victim", args.ratio_2), total_models // 2)

        # First train classifier using ALL test data
        clf = RandomForestClassifier(max_depth=3)

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
        n_points = len(clf.feature_importances_) // 3
        point_wise_importances = [
            clf.feature_importances_[:n_points],
            clf.feature_importances_[n_points:2*n_points],
            clf.feature_importances_[n_points*2:]
        ]
        # Account all activations for a point in importance calculation
        fis = np.sum(point_wise_importances, 0)
        fis = np.argsort(-fis)[:args.top_k]

        # Knowing which data points are best to be used, repeat process all over again
        # Data that will store x and y values
        clf = RandomForestClassifier(max_depth=3)

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
    # Example command: python census_acts.py --filter sex --ratio_1 0.5 --ratio_2 1.0
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--top_k', default=20, type=int,
                        help="top K datapoints")
    parser.add_argument('--trials', default=100, type=int,
                        help="number of trials")
    parser.add_argument('--num_models', default=40, type=int,
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
        r'%s proportion of training data ($\alpha$)' % PROPERTY_FOCUS[args.filter],
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
    sns_plot.figure.savefig("./activations_meta_%d.png" % args.num_models)
