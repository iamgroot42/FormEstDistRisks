from model_utils import get_models_path, load_model
from sklearn.ensemble import RandomForestClassifier
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import model_utils
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import flash_utils
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
        # Will represent the activation count corresponding to each layer in 1 model
        all_acts = []
        acts = model_utils.layer_output(x_te, model, layer=1)
        acts1 = model_utils.layer_output(x_te, model, layer=2)
        acts2 = model_utils.layer_output(x_te, model, layer=3)

        activationCount = np.sum(acts > 0, 1)
        activationCount1 = np.sum(acts1 > 0, 1)
        activationCount2 = np.sum(acts2 > 0, 1)

        all_acts.append(activationCount)
        all_acts.append(activationCount1)
        all_acts.append(activationCount2)

        model_data[0].append(np.concatenate(all_acts))
        model_data[1].append(ratio)


if __name__ == "__main__":
    # Example command: python census_acts.py --filter sex --ratio_1 0.5 --ratio_2 1.0
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        required=True,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2", required=True)
    args = parser.parse_args()
    flash_utils(args)

    # Get victim models
    models_victim_1 = get_models(
       get_models_path(args.filter, "victim", args.ratio_1))
    models_victim_2 = get_models(
       get_models_path(args.filter, "victim", args.ratio_2))

    # Load adv models
    total_models = 200
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
    _, (x_te_1, y_te_1), _ = ds_2.load_data(custom_limit=10000)
    # set both to ds_1 or ds_2 for current experiment
    _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
    y_te_1 = y_te_1.ravel()
    y_te_2 = y_te_2.ravel()

    # Iterate through data from both distrs
    z_vals, f_accs = [], []
    loaders = [(x_te_1, y_te_1), (x_te_2, y_te_2)]

    # First train classifier using ALL test data
    clf = RandomForestClassifier(max_depth=3)
    #Data that will store x and y values
    model_data = [[], []]

    for loader in loaders:
        # Load models and get/plot activations
        get_acts(loader, models_1, model_data, ratio=args.ratio_1)
        get_acts(loader, models_2, model_data, ratio=args.ratio_2)

    model_data[0] = np.array(model_data[0])
    model_data[1] = np.array(model_data[1])

    x_tr = model_data[0]
    y_tr = (model_data[1] != args.ratio_1)
    clf.fit(x_tr, y_tr)
    train_acc = 100 * clf.score(x_tr, y_tr)
    print("Accuracy using all data", train_acc)

    # Then, look at feature importances used by the model
    # And pick the top 100 points
    n_points = len(clf.feature_importances_) // 3
    point_wise_importances = [
        clf.feature_importances_[:n_points],
        clf.feature_importances_[n_points:2*n_points],
        clf.feature_importances_[n_points*2:]
    ]
    # Account all activations for a point in importance calculation
    top_k = 20
    fis = np.sum(point_wise_importances, 0)
    fis = np.argsort(-fis)[:top_k]

    # Knowing which data points are best to be used, repeat process all over again
    # Data that will store x and y values
    clf = RandomForestClassifier(max_depth=2)
    model_data = [[], []]

    for loader in loaders:
        # Load models and get/plot activations
        get_acts(loader, models_1, model_data, ratio=args.ratio_1, mask=fis)
        get_acts(loader, models_2, model_data, ratio=args.ratio_2, mask=fis)

    model_data[0] = np.array(model_data[0])
    model_data[1] = np.array(model_data[1])

    x_tr = model_data[0]
    y_tr = (model_data[1] != args.ratio_1)
    clf.fit(x_tr, y_tr)
    train_acc = 100 * clf.score(x_tr, y_tr)
    print("Accuracy using selected data", train_acc)

    # Get accuracy with this method on unseen victim models
    model_data = [[], []]

    for loader in loaders:
        # Load models and get/plot activations
        get_acts(loader, models_victim_1, model_data,
                 ratio=args.ratio_1, mask=fis)
        get_acts(loader, models_victim_2, model_data,
                 ratio=args.ratio_2, mask=fis)

    model_data[0] = np.array(model_data[0])
    model_data[1] = np.array(model_data[1])

    x_te = model_data[0]
    y_te = (model_data[1] != args.ratio_1)

    print("Accuracy on unseen models", 100 * clf.score(x_te, y_te))
