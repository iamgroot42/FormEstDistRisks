from data_utils import BotNetWrapper

from tqdm import tqdm
import torch as ch
import argparse
from utils import flash_utils
import os
import numpy as np
from model_utils import get_model, BASE_MODELS_DIR


@ch.no_grad()
def get_differences(models, data_loader, latent_focus, reduce=False):
    # View resulting activation distribution for current models

    reprs_all = []
    for x in data_loader:
        x = x.to('cuda')
        reprs = np.array(
            [m(x, latent=latent_focus).detach().cpu().numpy() for m in models])
        # Count number of neuron activations
        reprs = (1. * np.sum(reprs > 0, 2))
        if reduce:
            reprs = np.mean(reprs, 1)
        reprs_all.append(reprs)
        break
    return reprs_all


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


def ordered_samples(models_0, models_1, loader, args):
    reprs_0 = get_differences(models_0, loader,
                              args.latent_focus, reduce=False)
    reprs_1 = get_differences(models_1, loader,
                              args.latent_focus, reduce=False)

    diff_ids_picked = []
    for r0, r1 in zip(reprs_0, reprs_1):
        # Identify nodes per graph
        diffs = (np.min(r1, 0) - np.max(r0, 0))
        # Pick examples with maximum difference
        diff_ids = np.argsort(-np.abs(diffs))[:args.n_samples]
        diff_ids_picked.append(diff_ids)

    # Combine this collection
    return diff_ids_picked


def load_models(dir, args, n_models=None, shuffle=False):
    iterator = os.listdir(dir)
    if shuffle:
        np.random.permutation(iterator)
    iterator = iterator[:n_models]

    models = []
    for mpath in tqdm(iterator):
        # Load models
        model = get_model(args)
        model.load_state_dict(ch.load(os.path.join(dir, mpath)))
        model.eval()
        models.append(model)

    return models


def get_patterns(X_1, X_2, data_loader, picked_nodes, args):
    reprs_0 = get_differences(X_1, data_loader,
                              args.latent_focus)
    reprs_1 = get_differences(X_2, data_loader,
                              args.latent_focus)

    # Use picked nodes per graph
    reprs_0_use, reprs_1_use = [], []
    for pn, r0, r1 in zip(picked_nodes, reprs_0, reprs_1):
        reprs_0_use.append(r0[:, pn])
        reprs_1_use.append(r1[:, pn])

    reprs_0_use = np.concatenate(reprs_0_use, 1)
    reprs_1_use = np.concatenate(reprs_1_use, 1)

    return (reprs_0_use, reprs_1_use)


def specific_case(X_train_1, X_train_2, ds, args):
    # Get some normal data for estimates of activation values

    _, test_loader = ds.get_loaders(args.batch_size, shuffle=False)

    x_use_picked_nodes = ordered_samples(
        X_train_1, X_train_2, test_loader, args)

    # Get threshold on train data
    # Plot performance for train models
    reprs_0_use, reprs_1_use = get_patterns(
        X_train_1, X_train_2, test_loader, x_use_picked_nodes,
        args)

    print(reprs_0_use.shape)
    print(reprs_1_use.shape)

    threshold = get_threshold(reprs_0_use, reprs_1_use)
    train_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    return x_use_picked_nodes, threshold, train_acc


def main(args):
    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim", "0")
    train_dir_2 = os.path.join(BASE_MODELS_DIR, "victim", "1")
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv", "0")
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv", "1")
    # Get datasets ready
    ds_1 = BotNetWrapper(split="adv", prop_val=0)
    ds_2 = BotNetWrapper(split="adv", prop_val=1)

    n_train_models = args.n_models // 2
    X_train_1 = load_models(train_dir_1, args, n_train_models, shuffle=True)
    X_train_2 = load_models(train_dir_2, args, n_train_models, shuffle=True)

    # Load test models
    n_test_models = 100 // 2
    X_test_1 = load_models(test_dir_1, args, n_test_models)
    X_test_2 = load_models(test_dir_2, args, n_test_models)

    x_use_picked_nodes_1, threshold, train_acc_1 = specific_case(
        X_train_1, X_train_2, ds_1, args)
    x_use_picked_nodes_2, threshold, train_acc_2 = specific_case(
        X_train_1, X_train_2, ds_2, args)
    print("Train accuracies:", train_acc_1, train_acc_2)

    if train_acc_1 > train_acc_2:
        x_use_picked_nodes = x_use_picked_nodes_1
        _, test_loader = ds_1.get_loaders(args.batch_size, shuffle=False)
    else:
        x_use_picked_nodes = x_use_picked_nodes_2
        _, test_loader = ds_2.get_loaders(args.batch_size, shuffle=False)

    # Plot performance for test models
    reprs_0_use, reprs_1_use = get_patterns(
        X_test_1, X_test_2, test_loader,
        x_use_picked_nodes, args)

    # Get threshold on test data
    test_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    print("Test accuracy: %.3f" % test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--n_feat', type=int, default=1)
    parser.add_argument('--latent_focus', type=int, default=0)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--use_best', action="store_true",
                        help="Use lowest-loss example instead of all of them")
    args = parser.parse_args()
    args.gpu = True
    args.batch_size = 1
    flash_utils(args)

    main(args)
