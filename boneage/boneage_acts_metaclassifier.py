from sklearn.ensemble import RandomForestClassifier
from sympy import E
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
import joblib
from sklearn.model_selection import train_test_split
from glob import glob
import pandas as pd
import data_utils
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import flash_utils, heuristic
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from model_utils import load_model, get_pre_processor, BASE_MODELS_DIR
from data_utils import BoneWrapper, get_df, get_features

mpl.rcParams['figure.dpi'] = 200


def get_models(folder_path, n_models=100):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        model = model.cuda() #Shift to GPU
        models.append(model)
    return models


#def get_stats(mainmodel, dataloader, mask=None):
def get_stats(mainmodel, dataloader, mask = None):

    all_acts = []
    activationCount = 0
    for (x, y, sex) in (dataloader):

        for i in range(7):
            if mask is not None:
                x_eff = x[mask]
            else:
                x_eff = x
            acts = mainmodel(x_eff.cuda(), latent=i).detach()
            activationCount = ch.sum(acts > 0, 1).cpu().numpy()
            all_acts.append(activationCount)

    all_acts = np.concatenate(all_acts)

    return all_acts


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
        BASE_MODELS_DIR, "victim", args.ratio_1))
    models_victim_2 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.ratio_2))

#From boneage
    def filter(x): return x["gender"] == 1

    # Ready data
    _, df_val = get_df("adv")

    # Get data with ratio
    df1 = heuristic(
        df_val, filter, float(args.ratio_2),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    df2 = heuristic(
        df_val, filter, float(args.ratio_2),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    ds_1 = BoneWrapper(df1, df1)
    ds_2 = BoneWrapper(df2, df2)

    # Prepare data wrappers
    #ds_1 = BoneWrapper(args.filter, float(
    #    args.ratio_1), "adv", cwise_samples=1e6)
    #ds_2 = BoneWrapper(args.filter, float(
    #    args.ratio_2), "adv", cwise_samples=1e6)

    loaders = [
        ds_1.get_loaders(batch_size=400, shuffle=False)[1],
        ds_2.get_loaders(batch_size=400, shuffle=False)[1]
    ]

    relevant_acc = []
    for tr in range(args.trials):

        # Load adv models
        total_models = args.total_models
        models_1 = get_models(os.path.join(
            BASE_MODELS_DIR, "adv", args.ratio_1), total_models // 2)
        models_2 = get_models(os.path.join(
            BASE_MODELS_DIR, "adv", args.ratio_2), total_models // 2)

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
        num_layers = 7
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
        
        #Save Model
        joblib_file = "/u/jyc9fyf/celebaModels/celeb_metaclassifier_" + str(args.total_models) + "_" + str(args.ratio_2) +"_Trial" + str(tr) + ".pkl"
        joblib.dump(clf, joblib_file)

    return relevant_acc


if __name__ == "__main__":
    # Example command: python celeb_acts_metaclassifier.py --filter Male --trials 1 --total_models 10
    #CUDA_VISIBLE_DEVICES=2 python boneage_acts_metaclassifier.py --filter gender --trials 1 --total_models 10
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=['gender'],
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
    ratios_try = ["0.2", "0.3",
                  "0.4", "0.6", "0.7", "0.8"]
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


