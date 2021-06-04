from model_utils import get_models_path, load_model
from data_utils import CensusWrapper, SUPPORTED_PROPERTIES
import model_utils
import numpy as np
from tqdm import tqdm
import os
import argparse
from utils import get_z_value, get_threshold_acc, find_threshold_acc, flash_utils
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


def get_acts(data, models, colors, ratio):
    x_te, y_te = data
    all_acts = [] #Will represent the activation count corresponding to each layer in 1 model
    for model in tqdm(models):
        acts = model_utils.layer_output(x_te[0:99], model, layer = 0) #change to layer = 0,1, or 2
        acts1 = model_utils.layer_output(x_te[0:99], model, layer = 1) #change to layer = 0,1, or 2
        acts2 = model_utils.layer_output(x_te[0:99], model, layer = 2) #change to layer = 0,1, or 2
        print(acts)
        activationCount = np.sum(acts > 0, 1)
        activationCount1 = np.sum(acts1 > 0, 1)
        activationCount2 = np.sum(acts2 > 0, 1)

        all_acts.append(activationCount)
        all_acts.append(activationCount1)
        all_acts.append(activationCount2)

        #plt.plot(x, all_acts, color = colors)
        model_data[0].append(np.concatenate(all_acts))
        model_data[1].append(ratio)

        all_acts = []

    return np.array(acts), ratio


if __name__ == "__main__":
    #Example command: python census_acts.py --filter sex --ratio_1 0.5 --ratio_2 1.0
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2")
    args = parser.parse_args()
    flash_utils(args)

    # Get victim models
    #models_victim_1 = get_models(
    #    get_models_path(args.filter, "victim", args.ratio_1))
    #models_victim_2 = get_models(
    #    get_models_path(args.filter, "victim", args.ratio_2))

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
    _, (x_te_1, y_te_1), _ = ds_2.load_data(custom_limit=10000) #set both to ds_1 or ds_2 for current experiment
    _, (x_te_2, y_te_2), _ = ds_2.load_data(custom_limit=10000)
    y_te_1 = y_te_1.ravel()
    y_te_2 = y_te_2.ravel()

    # Iterate through data from both distrs
    z_vals, f_accs = [], []
    loaders = [(x_te_1, y_te_1), (x_te_2, y_te_2)]
    allaccs_1, allaccs_2 = [], []

    clf = model_utils.get_model()
    model_data = [[], []] #Data that will store x and y values

    for loader in loaders:
        
        # Load models and get/plot activations
        allaccs_1.append(get_acts(loader, models_1, colors = "orangered", ratio = args.ratio_1)) 
        allaccs_2.append(get_acts(loader, models_2, colors = "yellowgreen", ratio = args.ratio_2))
    
    x_tr = model_data[0]
    y_tr = model_data[1]

    clf.fit(x_tr, y_tr)
    
    print(model_data)
    train_acc = 100 * clf.score(x_tr, y_tr)

    print(train_acc)



   
