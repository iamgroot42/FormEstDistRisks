from tqdm import tqdm
import argparse
from utils import flash_utils
import os
import numpy as np
from data_utils import SUPPORTED_PROPERTIES
from model_utils import load_model, BASE_MODELS_DIR, layer_output
from data_utils import CensusWrapper


def get_differences(models, x_use, latent_focus, reduce=True):
    # View resulting activation distribution for current models
    reprs = np.array([layer_output(x_use, m, layer=latent_focus+1)
                      for m in models])

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


def ordered_samples(models_0, models_1, data, args):
    diffs_0, diffs_1 = [], []

    diffs_0 = get_differences(models_0, data, args.latent_focus, reduce=False)
    diffs_1 = get_differences(models_1, data, args.latent_focus, reduce=False)

    # print(diffs_0.shape)
    # print(diffs_1.shape)

    diffs_0 = diffs_0.T
    diffs_1 = diffs_1.T
    # diffs = (np.mean(diffs_1, 1) - np.mean(diffs_0, 1))
    diffs = (np.min(diffs_1, 1) - np.max(diffs_0, 1))
    # diffs = (np.min(diffs_0, 1) - np.max(diffs_1, 1))
    # Pick examples with maximum difference
    diff_ids = np.argsort(-np.abs(diffs))[:args.n_samples]
    print("Best samples had differences", diffs[diff_ids])
    return data[diff_ids]


def get_all_models(dir, n_models, shuffle=False):
    models = []
    files = os.listdir(dir)
    if shuffle:
        np.random.permutation(files)
    files = files[:n_models]
    for pth in tqdm(files):
        m = load_model(os.path.join(dir, pth))
        models.append(m)
    return models


def get_patterns(X_1, X_2, data, normal_data, args):
    reprs_0 = get_differences(X_1, data, args.latent_focus)
    reprs_1 = get_differences(X_2, data, args.latent_focus)

    if args.align:
        reprs_0_baseline = get_differences(
            X_1, normal_data, args.latent_focus)
        reprs_1_baseline = get_differences(
            X_2, normal_data, args.latent_focus)

        reprs_0_use = reprs_0 - reprs_0_baseline
        reprs_1_use = reprs_1 - reprs_1_baseline
    else:
        reprs_0_use = reprs_0
        reprs_1_use = reprs_1
    return (reprs_0_use, reprs_1_use)


def specific_case(X_train_1, X_train_2, ratio, args):
    # Get some normal data for estimates of activation values
    ds = CensusWrapper(filter_prop=args.filter, ratio=ratio, split="adv")

    _, (x_te, y_te), _ = ds.load_data()

    x_use = ordered_samples(X_train_1, X_train_2, x_te, args)

    # Get threshold on train data
    # Plot performance for train models
    reprs_0_use, reprs_1_use = get_patterns(
        X_train_1, X_train_2, x_use, x_te, args)

    threshold = get_threshold(reprs_0_use, reprs_1_use)
    train_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    return x_use, x_te, threshold, train_acc


def main(args):
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.first))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.second))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.first))
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/%s/" %
                              (args.filter, args.second))

    n_train_models = args.n_models // 2
    X_train_1 = get_all_models(train_dir_1, n_train_models, shuffle=True)
    X_train_2 = get_all_models(train_dir_2, n_train_models, shuffle=True)

    # Load test models
    n_test_models = 100
    X_test_1 = get_all_models(test_dir_1, n_test_models // 2)
    X_test_2 = get_all_models(test_dir_2, n_test_models // 2)

    x_use_1, normal_data, threshold, train_acc_1 = specific_case(
        X_train_1, X_train_2, float(args.first), args)
    x_use_2, normal_data, threshold, train_acc_2 = specific_case(
        X_train_1, X_train_2, float(args.second), args)
    print("Train accuracies:", train_acc_1, train_acc_2)

    if train_acc_1 > train_acc_2:
        x_use = x_use_1
    else:
        x_use = x_use_2

    # Plot performance for test models
    reprs_0_use, reprs_1_use = get_patterns(
        X_test_1, X_test_2, x_use, normal_data, args)

    # Get threshold on test data
    test_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    print("Test accuracy: %.3f" % test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Census')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--latent_focus', type=int, default=0)
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        default="sex", choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--first', help="Ratio for D_0", default="0.5")
    parser.add_argument('--second', help="Ratio for D_1",
                        type=str, required=True)
    parser.add_argument('--align', action="store_true",
                        help="Look at relative change in activation trends")
    parser.add_argument('--use_best', action="store_true",
                        help="Use lowest-loss example instead of all of them")
    args = parser.parse_args()

    flash_utils(args)

    main(args)
