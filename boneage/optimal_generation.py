import torch as ch
from tqdm import tqdm
import argparse
from utils import flash_utils, heuristic
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from model_utils import load_model, get_pre_processor, BASE_MODELS_DIR
from data_utils import BoneWrapper, get_df, get_features

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


# Get features across multiple layers
def combined_features(fe_model, models, x, layers):
    # Get activations for specified layers for both models
    reprs = []
    for layer in layers:
        repr = get_differences(fe_model, models, x, layer, reduce=False)
        reprs.append(repr.flatten())

    # Create feature vectors using these layers
    f_vecs = np.array(reprs).T
    return f_vecs


# Train decision-tree on combined features
def train_dt(fe_model, models_0, models_1, x_opt, layers, depth=2):
    fvecs_0 = combined_features(fe_model, models_0, x_opt.cuda(), layers)
    fvecs_1 = combined_features(fe_model, models_1, x_opt.cuda(), layers)

    x = list(fvecs_0) + list(fvecs_1)
    y = [0] * len(fvecs_0) + [1] * len(fvecs_1)

    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(x, y)
    return clf, clf.score(x, y)


# Get activation differences
def get_differences(fe_model, models, x_use, latent_focus, reduce=True):
    # View resulting activation distribution for current models
    reprs = ch.stack([m(fe_model(x_use), latent=latent_focus).detach()
                      for m in models], 0)
    # Count number of neuron activations
    reprs = (1. * ch.sum(reprs > 0, 2))
    if reduce:
        reprs = ch.mean(reprs, 1)
    reprs = reprs.cpu().numpy()
    return reprs


# Get accuracy using simple classification based on activation threshold
def get_acc(latents_0, latents_1, thresh):
    first = np.sum(latents_0 >= thresh)
    second = np.sum(latents_1 < thresh)
    acc = (first + second) / (latents_0.shape[0] + latents_1.shape[0])
    return acc


# Calculate optimal thresholds on given data
def get_threshold(latents_1, latents_2):
    min_elems = min(np.min(latents_1), np.min(latents_2))
    max_elems = max(np.max(latents_1), np.max(latents_2))
    thresholds = np.arange(min_elems, max_elems)
    accuracies = []
    for thresh in thresholds:
        acc = get_acc(latents_1, latents_2, thresh)
        accuracies.append(acc)
    return thresholds[np.argmax(accuracies)]


# Order data according to maximal difference in activations
def ordered_samples(fe_model, models_0, models_1, loader, args):
    diffs_0, diffs_1, inputs = [], [], []
    for tup in loader:
        x = tup[0]
        inputs.append(x)
        x = x.cuda()
        reprs_0 = get_differences(
            fe_model, models_0, x, args.latent_focus, reduce=False)
        reprs_1 = get_differences(
            fe_model, models_1, x, args.latent_focus, reduce=False)
        diffs_0.append(reprs_0)
        diffs_1.append(reprs_1)

    diffs_0 = np.concatenate(diffs_0, 1).T
    diffs_1 = np.concatenate(diffs_1, 1).T
    # diffs = (np.mean(diffs_1, 1) - np.mean(diffs_0, 1))
    diffs = (np.min(diffs_1, 1) - np.max(diffs_0, 1))
    # diffs = (np.min(diffs_0, 1) - np.max(diffs_1, 1))
    inputs = ch.cat(inputs)
    # Pick examples with maximum difference
    diff_ids = np.argsort(-np.abs(diffs))[:args.n_samples]
    print("Best samples had differences", diffs[diff_ids])
    return inputs[diff_ids].cuda()


# Generate optimal data to help differentiate between the two models
def gen_optimal(feature_extractor, models, labels, sample_shape,
                n_samples, n_steps, step_size, latent_focus,
                use_normal=None):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    if use_normal is None:
        x_rand_data = ch.rand(*((n_samples,) + sample_shape)).cuda()
    else:
        x_rand_data = use_normal.clone().cuda()
    print(models[0](feature_extractor(x_rand_data),
                    latent=latent_focus).shape[1:])

    iterator = tqdm(range(n_steps))
    # Focus on latent=4 for now
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)

        # Get representations from all models
        reprs = ch.stack(
            [m(feature_extractor(x_rand), latent=latent_focus) for m in models], 0)
        reprs_z = ch.mean(reprs[labels == 0], 2)
        reprs_o = ch.mean(reprs[labels == 1], 2)
        const = 2.
        const_neg = 0.5
        loss = ch.mean((const - reprs_z) ** 2) + \
            ch.mean((const_neg + reprs_o) ** 2)
        # loss = ch.mean((const_neg + reprs_z) ** 2) + ch.mean((const - reprs_o) ** 2)
        grad = ch.autograd.grad(loss, [x_rand])

        with ch.no_grad():
            zero_acts = ch.sum(1. * (reprs[labels == 0] > 0), 2)
            one_acts = ch.sum(1. * (reprs[labels == 1] > 0), 2)
            l1 = ch.mean((const - reprs_z) ** 2)
            l2 = ch.mean((const_neg + reprs_o) ** 2)
            # l1 = ch.mean((const_neg + reprs_z) ** 2)
            # l2 = ch.mean((const - reprs_o) ** 2)
            iterator.set_description("Loss: %.3f | ZA: %.1f | OA: %.1f | Loss(1): %.3f | Loss(2): %.3f" % (
                loss.item(), zero_acts.mean(), one_acts.mean(), l1, l2))

        with ch.no_grad():
            x_rand_data = x_rand_data - step_size * grad[0]
            x_rand_data = ch.clamp(x_rand_data, 0, 1)

    return x_rand.clone().detach(), (l1 + l2).item()


# Load all models
def get_all_models(dir, n_models, latent_focus, fake_relu):
    models = []
    files = os.listdir(dir)[:n_models]
    for pth in tqdm(files):
        m = load_model(os.path.join(dir, pth),
                       fake_relu=fake_relu,
                       latent_focus=latent_focus)
        m = m.cuda()
        models.append(m)
    return models


# Get activation differences
def get_patterns(fe_model, X_1, X_2, data, normal_data, args):
    reprs_0 = get_differences(fe_model, X_1, data, args.latent_focus)
    reprs_1 = get_differences(fe_model, X_2, data, args.latent_focus)

    if args.align:
        reprs_0_baseline = get_differences(
            fe_model, X_1, normal_data, args.latent_focus)
        reprs_1_baseline = get_differences(
            fe_model, X_2, normal_data, args.latent_focus)

        reprs_0_use = reprs_0 - reprs_0_baseline
        reprs_1_use = reprs_1 - reprs_1_baseline
    else:
        reprs_0_use = reprs_0
        reprs_1_use = reprs_1
    return (reprs_0_use, reprs_1_use)


# Load and prepare test data
def get_ds(ratio):
    def filter(x): return x["gender"] == 1

    # Ready data
    _, df_val = get_df("adv")

    # Get data with ratio
    df = heuristic(
        df_val, filter, ratio,
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    ds = BoneWrapper(df, df)
    return ds


def specific_case(X_train_1, X_train_2, Y_train, ratio, args):
    # Get feature extractor
    fe_model = get_pre_processor().cuda()

    # Get some normal data for estimates of activation values
    ds = get_ds(ratio)

    if args.use_normal:
        _, test_loader = ds.get_loaders(args.n_samples, shuffle=False)
        normal_data = next(iter(test_loader))[0]
    else:
        _, test_loader = ds.get_loaders(100, shuffle=True)
        normal_data = next(iter(test_loader))[0].cuda()

    if args.use_natural:
        x_use = ordered_samples(
            fe_model, X_train_1, X_train_2, test_loader, args)
    else:
        if args.start_natural:
            normal_data = ordered_samples(
                fe_model, X_train_1, X_train_2, test_loader, args)
            print("Starting with natural data")

        x_opt, losses = [], []
        use_normal = None
        for i in range(args.n_samples):

            if args.use_normal or args.start_natural:
                use_normal = normal_data[i:i + 1]

            # Get optimal point based on local set
            x_opt_, loss_ = gen_optimal(
                fe_model,
                X_train_1 + X_train_2, Y_train,
                (3, 224, 224), 1,
                args.steps, args.step_size,
                args.latent_focus,
                use_normal=use_normal)
            x_opt.append(x_opt_)
            losses.append(loss_)

        if args.use_best:
            best_id = np.argmin(losses)
            x_opt = x_opt[best_id:best_id+1]

        # x_opt = ch.cat(x_opt, 0)
        x_opt = normal_data

        x_use = x_opt

    x_use = x_use.cuda()
    clf = None
    # Get threshold on train data
    if args.use_dt:
        focus_layers = [int(x) for x in args.dt_layers.split(",")]
        clf, train_acc = train_dt(
            fe_model, X_train_1, X_train_2, x_use,
            focus_layers, depth=2)
    else:
        # Plot performance for train models
        reprs_0_use, reprs_1_use = get_patterns(
            fe_model, X_train_1, X_train_2,
            x_use, normal_data, args)

        threshold = get_threshold(reprs_0_use, reprs_1_use)
        train_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    return x_use, normal_data, threshold, train_acc, clf


def main(args):
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/" % (args.first))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/" % (args.second))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/" % (args.first))
    test_dir_2 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/" % (args.second))

    # Modify model behavior if using generation mode
    if args.use_natural:
        latent_focus = None
        fake_relu = False
    else:
        latent_focus = args.latent_focus
        fake_relu = True

    n_models = args.n_models
    X_train_1 = get_all_models(
        train_dir_1, n_models // 2, latent_focus, fake_relu=fake_relu)
    X_train_2 = get_all_models(
        train_dir_2, n_models // 2, latent_focus, fake_relu=fake_relu)
    Y_train = [0.] * len(X_train_1) + [1.] * len(X_train_2)
    Y_train = ch.from_numpy(np.array(Y_train)).cuda()

    test_models_use = 100
    X_test_1 = get_all_models(
        test_dir_1, test_models_use // 2, latent_focus, fake_relu=fake_relu)
    X_test_2 = get_all_models(
        test_dir_2, test_models_use // 2, latent_focus, fake_relu=fake_relu)
    Y_test = [0.] * len(X_test_1) + [1.] * len(X_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()

    # Get feature extractor
    fe_model = get_pre_processor().cuda()

    x_use_1, normal_data, threshold, train_acc_1, clf_1 = specific_case(
        X_train_1, X_train_2, Y_train, float(args.first), args)
    x_use_2, normal_data, threshold, train_acc_2, clf_2 = specific_case(
          X_train_1, X_train_2, Y_train, float(args.second), args)
    print("Train accuracies:", train_acc_1, train_acc_2)

    if train_acc_1 > train_acc_2:
        x_use = x_use_1
        clf = clf_1
    else:
        x_use = x_use_2
        clf = clf_2

    if args.use_dt:
        focus_layers = [int(x) for x in args.dt_layers.split(",")]
        feat_0 = combined_features(fe_model, X_test_1, x_use, focus_layers)
        feat_1 = combined_features(fe_model, X_test_2, x_use, focus_layers)
        x = list(feat_0) + list(feat_1)
        y = [0] * len(feat_0) + [1] * len(feat_1)
        test_acc = clf.score(x, y)
    else:
        # Plot performance for test models
        reprs_0_use, reprs_1_use = get_patterns(
            fe_model, X_test_1, X_test_2, x_use,
            normal_data, args)

        # Get threshold on test data
        test_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    print("Test accuracy: %.3f" % test_acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--latent_focus', type=int, default=0)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=1e2)
    parser.add_argument('--first', help="Ratio for D_0", default="0.5")
    parser.add_argument('--second', help="Ratio for D_1")
    parser.add_argument('--use_normal', action="store_true",
                        help="Use normal data for init instead of noise")
    parser.add_argument('--align', action="store_true",
                        help="Look at relative change in activation trends")
    parser.add_argument('--use_best', action="store_true",
                        help="Use lowest-loss example instead of all of them")
    parser.add_argument('--use_natural', action="store_true",
                        help="Pick from actual images")
    parser.add_argument('--use_dt', action="store_true",
                        help="Train small decision tree based on activation values")
    parser.add_argument('--dt_layers', default="0,1",
                        help="layers to use features for (if and when training DT)")
    parser.add_argument('--start_natural', action="store_true",
                        help="Start with natural images, but better criteria")
    args = parser.parse_args()

    if args.start_natural and args.use_normal:
        raise ValueError(
            "Can't start with nautral images to optimize and not use them")

    flash_utils(args)

    main(args)
