import data_utils
import torch as ch
import argparse
from model_utils import get_model, BASE_MODELS_DIR
import os
from utils import find_threshold_acc, get_threshold_acc
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def load_models(model_dir, ds, args, max_read=None):
    iterator = os.listdir(model_dir)
    if max_read is not None:
        iterator = np.random.permutation(iterator)[:max_read]

    models = []
    for mpath in tqdm(iterator):
        # Load model
        model = get_model(ds, args)
        model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        model.eval()
        models.append(model)
    return models


@ch.no_grad()
def get_model_preds(models, ds):
    _, test_idx = ds.get_idx_split()
    X = ds.get_features()
    Y = ds.get_labels()

    preds = []
    for model in tqdm(models):
        # Load model
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
    parser.add_argument(
        '--property', choices=data_utils.SUPPORTED_PROPERTIES, default="mean")
    parser.add_argument('--deg', type=int)
    parser.add_argument('--gpu', action="store_true")
    args = parser.parse_args()
    print(args)

    # Get datasets ready
    ds_1 = data_utils.ArxivNodeDataset("adv")
    ds_2 = data_utils.ArxivNodeDataset("adv")

    # Modify dataset properties
    if args.property == "mean":
        deg_1, deg_2 = 13, args.deg

        # Modify mean degree. prune random nodes
        ds_1.change_mean_degree(deg_1, 0.01)
        ds_2.change_mean_degree(deg_2, 0.01)
    else:
        deg_1, deg_2 = "og", args.deg

        # Get rid of nodes above a specified node-degree
        ds_2.keep_below_degree_threshold(deg_2)

    # Directories where saved models are stored
    dir_victim_1 = os.path.join(BASE_MODELS_DIR, "victim", "deg" + str(deg_1))
    dir_victim_2 = os.path.join(BASE_MODELS_DIR, "victim", "deg" + str(deg_2))
    dir_1 = os.path.join(BASE_MODELS_DIR, "adv", "deg" + str(deg_1))
    dir_2 = os.path.join(BASE_MODELS_DIR, "adv", "deg" + str(deg_2))

    z_vals, f_accs = [], []
    degrees = [deg_1, deg_2]
    loders = [ds_1, ds_2]
    allaccs_1, allaccs_2 = [], []
    for j, loader in enumerate(loders):

        # Load victim models
        models_victim_1 = load_models(dir_victim_1, loader, args)
        models_victim_2 = load_models(dir_victim_2, loader, args)

        # Load adv models
        total_models = 100
        models_1 = load_models(dir_1, loader, args, total_models // 2)
        models_2 = load_models(dir_2, loader, args, total_models // 2)

        # Get model predictions
        preds_1, y_gt = get_model_preds(models_1, loader)
        preds_2, _ = get_model_preds(models_2, loader)

        # Get accuracies
        accs_1 = ch.Tensor([ch.mean(1. * (x[:, 0] == y_gt[:, 0]))
                            for x in preds_1]).numpy()
        accs_2 = ch.Tensor([ch.mean(1. * (x[:, 0] == y_gt[:, 0]))
                            for x in preds_2]).numpy()

        # Plot stuff
        plt.hist(accs_1, bins=50, density=True, alpha=0.5,
                 label="Deg-%s on %s" % (str(deg_1), str(degrees[j])))
        plt.hist(accs_2, bins=50, density=True, alpha=0.5,
                 label="Deg-%s on %s" % (str(deg_2), str(degrees[j])))

        # Look at [0, 100]
        accs_1 *= 100
        accs_2 *= 100

        # Calculate Z value
        m1, v1 = np.mean(accs_1), np.var(accs_1)
        m2, v2 = np.mean(accs_2), np.var(accs_2)
        mean_new = np.abs(m1 - m2)
        var_new = (v1 + v2) / total_models
        Z = mean_new / np.sqrt(var_new)

        print("Mean-1: %.3f, Mean-2: %.3f" % (m1, m2))
        print("Var-1: %.3f, Var-2: %.3f" % (v1, v2))
        print("Number of samples: %d" % total_models)
        z_vals.append(Z)

        tracc, threshold = find_threshold_acc(accs_1, accs_2, granularity=0.01)
        print("[Adversary] Threshold based accuracy: %.2f at threshold %.2f" %
              (100 * tracc, threshold))

        # Compute accuracies on this data for victim
        preds_victim_1, y_gt = get_model_preds(models_victim_1, loader)
        preds_victim_2, _ = get_model_preds(models_victim_2, loader)
        accs_victim_1 = ch.Tensor(
            [ch.mean(1. * (x[:, 0] == y_gt[:, 0])) for x in preds_victim_1]).numpy()
        accs_victim_2 = ch.Tensor(
            [ch.mean(1. * (x[:, 0] == y_gt[:, 0])) for x in preds_victim_2]).numpy()

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

    print("Z values:", ["%.2f" % x for x in z_vals])
    print("Maximum Z value:", max(z_vals))

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

    plt.legend()
    plt.savefig("./acc_distr_%s_%s.png" % (str(deg_1), str(deg_2)))


if __name__ == "__main__":
    main()
