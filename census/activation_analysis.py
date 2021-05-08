import utils
import data_utils
from model_utils import layer_output, load_model, get_models_path
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def main(args):
    # Get filters
    filter = data_utils.get_default_filter(args.filter, args.ratio)

    # Load datase
    ci = data_utils.CensusIncome()

    # Get test data
    (x_tr, y_tr), (x_te, y_te), _ = ci.load_data(filter, split="adv")

    # Load models
    m1_dir = get_models_path(args.filter, "adv", str(args.focus_1))
    m2_dir = get_models_path(args.filter, "adv", str(args.focus_2))
    models_1 = [load_model(os.path.join(m1_dir, m))
                for m in tqdm(os.listdir(m1_dir))]
    models_2 = [load_model(os.path.join(m2_dir, m))
                for m in tqdm(os.listdir(m2_dir))]

    # Look at activations for particular layer
    acts_1 = np.array([layer_output(x_te, m, args.layer) for m in tqdm(models_1)])
    acts_2 = np.array([layer_output(x_te, m, args.layer) for m in tqdm(models_2)])

    # Count number of activations
    acts_1 = np.sum(acts_1 > 0, 1)
    acts_2 = np.sum(acts_2 > 0, 1)

    return acts_1, acts_2


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratio', type=float,
                        help='ratio to work with')
    parser.add_argument('--focus_1', type=float,
                        help='ratio of property of first set of models')
    parser.add_argument('--focus_2', type=float,
                        help='ratio of property of second set of models')
    parser.add_argument('--filter', choices=data_utils.SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--layer', type=int,
                        help='layer to extract activations from')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Set dark background
    plt.style.use('dark_background')

    acts_1, acts_2 = main(args)

    # Plot histograms of activations for these two models
    plt.hist(acts_1, bins=100, label="%s:%f" % (args.filter, args.focus_1))
    plt.hist(acts_2, bins=100, label="%s:%f" % (args.filter, args.focus_2))
    plt.savefig("./activations.png")
