from data_utils import ArxivNodeDataset, SUPPORTED_PROPERTIES
from tqdm import tqdm
import torch as ch
import argparse
from utils import flash_utils
import os
import numpy as np
from model_utils import get_model, BASE_MODELS_DIR


@ch.no_grad()
def get_differences(models, x_use, test_idx, latent_focus, reduce=True):
    # View resulting activation distribution for current models
    reprs = np.array([m(x_use, latent=latent_focus)[test_idx].detach().cpu().numpy() for m in models])
    # Count number of neuron activations
    reprs = (1. * np.sum(reprs > 0, 2))
    if reduce:
        reprs = np.mean(reprs, 1)
    return reprs


def get_acc(latents_0, latents_1, thresh):
    first = np.sum(latents_0 >= thresh)
    second = np.sum(latents_1 < thresh)
    acc = (first + second) / (latents_0.shape[0] + latents_1.shape[0])
    return acc


def get_threshold(latents_1, latents_2):
    min_elems = min(np.min(latents_1), np.min(latents_2))
    max_elems = max(np.max(latents_1), np.max(latents_2))
    thresholds = np.arange(min_elems, max_elems)

    accuracies = []
    for thresh in thresholds:
        acc = get_acc(latents_1, latents_2, thresh)
        accuracies.append(acc)
    return thresholds[np.argmax(accuracies)]


def ordered_samples(models_0, models_1, data, test_idx, args):
    diffs_0, diffs_1 = [], []

    diffs_0 = get_differences(models_0, data, test_idx,
                              args.latent_focus, reduce=False)
    diffs_1 = get_differences(models_1, data, test_idx,
                              args.latent_focus, reduce=False)

    diffs_0 = diffs_0.T
    diffs_1 = diffs_1.T
    # diffs = (np.mean(diffs_1, 1) - np.mean(diffs_0, 1))
    diffs = (np.min(diffs_1, 1) - np.max(diffs_0, 1))
    # diffs = (np.min(diffs_0, 1) - np.max(diffs_1, 1))
    # Pick examples with maximum difference
    diff_ids = np.argsort(-np.abs(diffs))[:args.n_samples]
    print("Best samples had differences", diffs[diff_ids])
    return test_idx[diff_ids]


def load_models(dir, args, ds_1, ds_2, n_models=None, shuffle=False):
    iterator = os.listdir(dir)
    if shuffle:
        np.random.permutation(iterator)
    iterator = iterator[:n_models]

    models_1, models_2 = [], []
    for mpath in tqdm(iterator):
        # Load model for DS-1
        model = get_model(ds_1, args)
        model.load_state_dict(ch.load(os.path.join(dir, mpath)))
        model.eval()
        models_1.append(model)

        # Load model for DS-2
        model = get_model(ds_2, args)
        model.load_state_dict(ch.load(os.path.join(dir, mpath)))
        model.eval()
        models_2.append(model)

    return models_1, models_2


def get_patterns(X_1, X_2, use_idx, data, args):
    reprs_0 = get_differences(X_1, data, use_idx, args.latent_focus)
    reprs_1 = get_differences(X_2, data, use_idx, args.latent_focus)

    if args.align:
        reprs_0_baseline = get_differences(
            X_1, data, use_idx, args.latent_focus)
        reprs_1_baseline = get_differences(
            X_2, data, use_idx, args.latent_focus)

        reprs_0_use = reprs_0 - reprs_0_baseline
        reprs_1_use = reprs_1 - reprs_1_baseline
    else:
        reprs_0_use = reprs_0
        reprs_1_use = reprs_1
    return (reprs_0_use, reprs_1_use)


def specific_case(X_train_1, X_train_2, ds, args):
    # Get some normal data for estimates of activation values

    _, test_idx = ds.get_idx_split()
    X = ds.get_features()

    x_use_idx = ordered_samples(X_train_1, X_train_2, X, test_idx, args)

    # Get threshold on train data
    # Plot performance for train models
    reprs_0_use, reprs_1_use = get_patterns(
        X_train_1, X_train_2, x_use_idx, X, args)

    threshold = get_threshold(reprs_0_use, reprs_1_use)
    train_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    return x_use_idx, X, threshold, train_acc


def main(args):
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/deg%s/" % (args.first))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "victim/deg%s/" % (args.second))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/deg%s/" % (args.first))
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/deg%s/" %
                              (args.second))

    ds_1, ds_2 = ArxivNodeDataset("victim"), ArxivNodeDataset("victim")

    # Modify dataset properties
    if args.property == "mean":
        # Modify mean degree. prune random nodes
        ds_1.change_mean_degree(float(args.first), 0.01)
        ds_2.change_mean_degree(float(args.second), 0.01)
    else:
        # Get rid of nodes above a specified node-degree
        ds_1.keep_below_degree_threshold(float(args.first))
        ds_1.keep_below_degree_threshold(float(args.second))

    n_train_models = args.n_models // 2
    X_train_2 = load_models(train_dir_2, args, ds_1,
                            ds_2, n_train_models, shuffle=True)
    X_train_1 = load_models(train_dir_1, args, ds_1,
                            ds_2, n_train_models, shuffle=True)

    # Load test models
    n_test_models = 100 // 2
    X_test_1 = load_models(test_dir_1, args, ds_1, ds_2, n_test_models)
    X_test_2 = load_models(test_dir_2, args, ds_1, ds_2, n_test_models)

    x_use_idx_1, normal_data_1, threshold, train_acc_1 = specific_case(
        X_train_1[0], X_train_2[0], ds_1, args)
    x_use_idx_2, normal_data_2, threshold, train_acc_2 = specific_case(
        X_train_1[1], X_train_2[1], ds_2, args)
    print("Train accuracies:", train_acc_1, train_acc_2)

    if train_acc_1 > train_acc_2:
        x_use_idx = x_use_idx_1
        normal_data = normal_data_1
        use = 0
    else:
        x_use_idx = x_use_idx_2
        normal_data = normal_data_2
        use = 1

    # Plot performance for test models
    reprs_0_use, reprs_1_use = get_patterns(
        X_test_1[use], X_test_2[use], x_use_idx, normal_data, args)

    # Get threshold on test data
    test_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    print("Test accuracy: %.3f" % test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--latent_focus', type=int, default=0)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument(
        '--property', choices=SUPPORTED_PROPERTIES, default="mean")
    parser.add_argument('--first', help="Ratio for D_0", default="13")
    parser.add_argument('--second', help="Ratio for D_1",
                        type=str, required=True)
    parser.add_argument('--align', action="store_true",
                        help="Look at relative change in activation trends")
    parser.add_argument('--use_best', action="store_true",
                        help="Use lowest-loss example instead of all of them")
    args = parser.parse_args()

    flash_utils(args)

    main(args)
