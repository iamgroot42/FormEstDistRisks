from model_utils import get_model, BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES
import torch.nn as nn
import numpy as np
import utils
import torch as ch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

mpl.rcParams['figure.dpi'] = 200



def get_stats(mainmodel, dataloader, point, return_acts=True):

    all_acts = [] if return_acts else None
    activationCount = 0
    #x_te, y_te, _ = dataloader

    for (x_te, y, _) in (dataloader):
        
        for i in range(0,7):

            #y_ = y_te.cuda()

            acts = mainmodel(x_te[point:point + 1].cuda(), latent=i).detach() #Get activation values for data at x_te[index]
            actsMax = acts.shape[1]                            #Get maximum activations

            activationCount = ch.sum(acts > 0, 1).cpu().numpy()   #Count positive activations

            actFrac = activationCount / actsMax       #Divide by max activations to get fraction of max activations

            #print(acts.shape)
            #print(activationCount.shape)
            if return_acts:
                all_acts.append(actFrac)
        if (i == 6):
            print(all_acts)
            all_acts = np.concatenate(all_acts)     
            print(all_acts)
            return all_acts                     #Should return length 7 array of activations per layer




def get_models(folder_path, n_models=100):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = get_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_acts(loader, models, colors, point):
    all_acts = [] #Will represent the activation count corresponding to each layer in 1 model
    for model in tqdm(models):
        acts = get_stats(model, loader, point = point)
        plt.plot(x, acts, color = colors)


    return np.array(acts)


if __name__ == "__main__":
    #Example command: python celeb_acts.py --filter Male --ratio_1 0.5 --ratio_2 1.0 --total_models 100
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2")
    parser.add_argument('--total_models', type=int, default=100)
    args = parser.parse_args()
    utils.flash_utils(args)


    # Prepare data wrappers
    ds_1 = CelebaWrapper(args.filter, float(
        args.ratio_1), "adv", cwise_samples=1e6)
    ds_2 = CelebaWrapper(args.filter, float(
        args.ratio_2), "adv", cwise_samples=1e6)

    # Get loaders
    loaders = [
        ds_1.get_loaders(batch_size=400, shuffle=False)[1],
        ds_2.get_loaders(batch_size=400, shuffle=False)[1]
    ]



    # Load adv models
    total_models = args.total_models
    models_1 = get_models(os.path.join(
        BASE_MODELS_DIR, "adv", args.filter, args.ratio_1), total_models // 2)
    models_2 = get_models(os.path.join(
        BASE_MODELS_DIR, "adv", args.filter, args.ratio_2), total_models // 2)

    allaccs_1, allaccs_2 = [], []

    for point in range(0,15): #repeat for first 15 data points x

        for loader in loaders:

            x = [0,1,2,3,4,5,6]
            allaccs_1.append(get_acts(loader, models_1, colors = "orangered", point = point)) 
            allaccs_2.append(get_acts(loader, models_2, colors = "yellowgreen", point = point))
            




        #plt.plot(allaccs_1, [0,1,2])
        #plt.plot(accs_2, [0,1,2])
        plt.xticks(np.arange(min(x), max(x)+1, 1.0))
        plt.title("Activations on models with ratios 0.5 (red) vs 0.0 (green)")
        plt.savefig("/u/jyc9fyf/celebGraphs/0.5_vs_0_ds_" + str(point) + "_.png")
        plt.clf()
