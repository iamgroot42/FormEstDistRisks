import data_utils
import torch as ch
import argparse
from model_utils import get_model
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


@ch.no_grad()
def get_model_preds(model_dir, ds, args, max_read=None):
    _, test_idx = ds.get_idx_split()
    X = ds.get_features()
    Y = ds.get_labels()

    preds = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        # Load model
        model = get_model(ds, args)
        model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        model.eval()

        out = model(X)[test_idx].detach()
        pred = out.argmax(dim=-1, keepdim=True)

        preds.append(pred)

    preds = ch.stack(preds)
    return preds, Y[test_idx]


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--property', choices=data_utils.SUPPORTED_PROPERTIES)
    parser.add_argument('--deg', type=int)
    args = parser.parse_args()
    print(args)

    # Get datasets ready
    ds_1 = data_utils.ArxivNodeDataset("adv")
    ds_2 = data_utils.ArxivNodeDataset("adv")

    # Modify dataset properties
    if args.property == " mean":
        deg_1, deg_2 = 13, args.deg

        # Modify mean degree
        ds_1.change_mean_degree(deg_1)
        ds_2.change_mean_degree(deg_2)
    else:
        deg_1, deg_2 = "og", args.deg

        # Get rid of nodes above a specified node-degree
        ds_2.keep_below_degree_threshold(deg_2)

    # Directories where saved models are stored
    dir_1 = "models/victim/deg%s" % str(deg_1)
    dir_2 = "models/victim/deg%s" % str(deg_2)

    # Function to plot accuracies for both sets of models
    def plot_accs(ds, deg):

        preds_1, y_gt = get_model_preds(dir_1, ds, args)
        preds_2, _ = get_model_preds(dir_2, ds, args)

        accs_1 = ch.Tensor([ch.mean(1. * (x[:, 0] == y_gt[:, 0]))
                            for x in preds_1]).numpy()
        accs_2 = ch.Tensor([ch.mean(1. * (x[:, 0] == y_gt[:, 0]))
                            for x in preds_2]).numpy()

        # plt.plot(np.arange(len(accs_1)), np.sort(
        #     accs_1), label="Deg-%s on %s" % (str(deg_1), str(deg)))
        # plt.plot(np.arange(len(accs_2)), np.sort(
        #     accs_2), label="Deg-%s on %s" % (str(deg_2), str(deg)))

        plt.hist(accs_1, bins=50, density=True, alpha=0.5,
                 label="Deg-%s on %s" % (str(deg_1), str(deg)))
        plt.hist(accs_2, bins=50, density=True, alpha=0.5,
                 label="Deg-%s on %s" % (str(deg_2), str(deg)))

    # Plot accuracies for both datasets
    plot_accs(ds_1, deg_1)
    plot_accs(ds_2, deg_2)

    plt.legend()
    plt.savefig("./acc_distr_%s_%s.png" % (str(deg_1), str(deg_2)))


if __name__ == "__main__":
    main()
