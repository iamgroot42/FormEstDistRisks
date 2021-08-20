import torch as ch
from tqdm import tqdm
import argparse
from utils import flash_utils
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from data_utils import SUPPORTED_PROPERTIES
from model_utils import get_model, BASE_MODELS_DIR
from kornia.geometry.transform import resize
from data_utils import CelebaWrapper
from robustness.tools.vis_tools import show_image_row
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def combined_features(models, x, layers):
    # Get activations for specified layers for both models
    reprs = []
    for layer in layers:
        repr = get_differences(models, x, layer, reduce=False)
        reprs.append(repr.flatten())

    # Create feature vectors using these layers
    f_vecs = np.array(reprs).T
    return f_vecs


def train_dt(models_0, models_1, x_opt, layers, depth=2):
    fvecs_0 = combined_features(models_0, x_opt.cuda(), layers)
    fvecs_1 = combined_features(models_1, x_opt.cuda(), layers)

    x = list(fvecs_0) + list(fvecs_1)
    y = [0] * len(fvecs_0) + [1] * len(fvecs_1)

    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(x, y)
    return clf, clf.score(x, y)


def get_differences(models, x_use, latent_focus, reduce=True):
    # View resulting activation distribution for current models
    reprs = ch.stack([m(x_use, latent=latent_focus).detach()
                      for m in models], 0)
    # Count number of neuron activations
    reprs = (1. * ch.sum(reprs > 0, 2))
    if reduce:
        reprs = ch.mean(reprs, 1)
    reprs = reprs.cpu().numpy()
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


def ordered_samples(models_0, models_1, loader, args):
    diffs_0, diffs_1, inputs = [], [], []
    for tup in loader:
        x = tup[0]
        inputs.append(x)
        x = x.cuda()
        reprs_0 = get_differences(models_0, x, args.latent_focus, reduce=False)
        reprs_1 = get_differences(models_1, x, args.latent_focus, reduce=False)
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


def gen_optimal(models, labels, sample_shape, n_samples,
                n_steps, step_size, latent_focus, upscale,
                use_normal=None, constrained=False,):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    if upscale > 1:
        actual_shape = (sample_shape[0], sample_shape[1] //
                        upscale, sample_shape[2] // upscale)
        x_rand_data = ch.rand(*((n_samples,) + actual_shape)).cuda()
        x_eff = resize(x_rand_data, (sample_shape[1], sample_shape[2]))
        print(models[0](x_eff, latent=latent_focus).shape[1:])
    else:
        if use_normal is None:
            x_rand_data = ch.rand(*((n_samples,) + sample_shape)).cuda()
        else:
            x_rand_data = use_normal.clone().cuda()
        print(models[0](x_rand_data, latent=latent_focus).shape[1:])

    x_rand_data_start = x_rand_data.clone().detach()

    iterator = tqdm(range(n_steps))
    # Focus on latent=4 for now
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)

        if upscale > 1:
            x_use = resize(x_rand, (sample_shape[1], sample_shape[2]))
        else:
            x_use = x_rand

        # Get representations from all models
        reprs = ch.stack([m(x_use, latent=latent_focus) for m in models], 0)
        reprs_z = ch.mean(reprs[labels == 0], 2)
        reprs_o = ch.mean(reprs[labels == 1], 2)
        # const = 2.
        const = 1.
        const_neg = 0.5
        loss = ch.mean((const - reprs_z) ** 2) + ch.mean((const_neg + reprs_o) ** 2)
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
            x_intermediate = x_rand_data - step_size * grad[0]
            if constrained:
                shape = x_rand_data.shape
                difference = (x_rand_data_start - x_intermediate)
                difference = difference.view(difference.shape[0], -1)
                eps = 0.5
                difference_norm = eps * ch.norm(difference, p=2, dim=0, keepdim=True)
                difference_norm = difference_norm.view(*shape)
                # difference = difference.renorm(p=2, dim=0, maxnorm=eps)
                x_rand_data = x_rand_data_start - difference_norm
            else:
                x_rand_data = x_intermediate
            x_rand_data = ch.clamp(x_rand_data, -1, 1)

    return x_rand.clone().detach(), (l1 + l2).item()


def get_all_models(dir, n_models, latent_focus, fake_relu, shuffle=False):
    models = []
    files = os.listdir(dir)
    if shuffle:
        np.random.permutation(files)
    files = files[:n_models]
    for pth in tqdm(files):
        m = get_model(os.path.join(dir, pth), fake_relu=fake_relu,
                      latent_focus=latent_focus)
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


def specific_case(X_train_1, X_train_2, Y_train, ratio, args):
    # Get some normal data for estimates of activation values
    ds = CelebaWrapper(args.filter, ratio, "adv")
    if args.use_normal:
        _, test_loader = ds.get_loaders(args.n_samples, eval_shuffle=False)
        normal_data = next(iter(test_loader))[0]
    else:
        _, test_loader = ds.get_loaders(100, eval_shuffle=True)
        normal_data = next(iter(test_loader))[0].cuda()

    if args.use_natural:
        x_use = ordered_samples(X_train_1, X_train_2, test_loader, args)
    else:
        if args.start_natural:
            normal_data = ordered_samples(
                X_train_1, X_train_2, test_loader, args)
            print("Starting with natural data")

        x_opt, losses = [], []
        for i in range(args.n_samples):
            # Get optimal point based on local set
            x_opt_, loss_ = gen_optimal(
                                X_train_1 + X_train_2, Y_train,
                                (3, 218, 178), 1,
                                args.steps, args.step_size,
                                args.latent_focus, args.upscale,
                                use_normal=normal_data[i:i+1] if (args.use_normal or args.start_natural) else None,
                                constrained=args.constrained)
            x_opt.append(x_opt_)
            losses.append(loss_)

        if args.use_best:
            best_id = np.argmin(losses)
            x_opt = x_opt[best_id:best_id+1]

        # x_opt = ch.cat(x_opt, 0)
        x_opt = normal_data

        if args.upscale:
            x_use = resize(x_opt, (218, 178))
        else:
            x_use = x_opt

    x_use = x_use.cuda()
    clf = None
    # Get threshold on train data
    if args.use_dt:
        focus_layers = [int(x) for x in args.dt_layers.split(",")]
        clf, train_acc = train_dt(
            X_train_1, X_train_2, x_use, focus_layers, depth=2)
    else:
        # Plot performance for train models
        reprs_0_use, reprs_1_use = get_patterns(
            X_train_1, X_train_2, x_use, normal_data, args)

        threshold = get_threshold(reprs_0_use, reprs_1_use)
        train_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

    return x_use.cpu(), normal_data.cpu(), threshold, train_acc, clf


def main(args):
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.first))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.second))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.first))
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/%s/" %
                              (args.filter, args.second))

    # Modify model behavior if using generation mode
    if args.use_natural:
        latent_focus = None
        fake_relu = False
    else:
        latent_focus = args.latent_focus
        fake_relu = True

    X_train_1 = get_all_models(
        train_dir_1, args.n_models, latent_focus, fake_relu,
        shuffle=True)
    X_train_2 = get_all_models(
        train_dir_2, args.n_models, latent_focus, fake_relu,
        shuffle=True)
    Y_train = [0.] * len(X_train_1) + [1.] * len(X_train_2)
    Y_train = ch.from_numpy(np.array(Y_train)).cuda()

    # Load test models
    n_test_models = 100
    X_test_1 = get_all_models(
        test_dir_1, n_test_models, latent_focus, fake_relu)
    X_test_2 = get_all_models(
        test_dir_2, n_test_models, latent_focus, fake_relu)
    Y_test = [0.] * len(X_test_1) + [1.] * len(X_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()

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

    x_use = x_use.cuda()

    if args.use_dt:
        focus_layers = [int(x) for x in args.dt_layers.split(",")]
        feat_0 = combined_features(X_test_1, x_use, focus_layers)
        feat_1 = combined_features(X_test_2, x_use, focus_layers)
        x = list(feat_0) + list(feat_1)
        y = [0] * len(feat_0) + [1] * len(feat_1)
        test_acc = clf.score(x, y)
    else:
        # Plot performance for test models
        reprs_0_use, reprs_1_use = get_patterns(
            X_test_1, X_test_2, x_use, normal_data, args)

        # Get threshold on test data
        test_acc = get_acc(reprs_0_use, reprs_1_use, threshold)

        # Plot differences
        plt.plot(np.arange(len(X_test_1)), sorted(
            reprs_0_use), label=r'Trained on $D_0$')
        plt.plot(np.arange(len(X_test_2)), sorted(
            reprs_1_use), label=r'Trained on $D_1$')
        plt.legend()
        plt.savefig("./on_test_distr.png")
        plt.clf()

    print("Test accuracy: %.3f" % test_acc)

    # Bring from (-1, 1) to (0, 1) normalization
    x_use_show = (x_use.cpu() + 1) / 2
    # Plot images
    show_image_row([x_use_show], filename='./see_samples.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Celeb-A')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--latent_focus', type=int, default=4)
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        default="Male", choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=1e2)
    parser.add_argument('--first', help="Ratio for D_0", default="0.5")
    parser.add_argument('--second', help="Ratio for D_1", type=str)
    parser.add_argument('--use_normal', action="store_true",
                        help="Use normal data for init instead of noise")
    parser.add_argument('--align', action="store_true",
                        help="Look at relative change in activation trends")
    parser.add_argument('--use_best', action="store_true",
                        help="Use lowest-loss example instead of all of them")
    parser.add_argument('--upscale', type=int,
                        default=1, help="optimize and upscale")
    parser.add_argument('--use_natural', action="store_true",
                        help="Pick from actual images")
    parser.add_argument('--use_dt', action="store_true",
                        help="Train small decision tree based on activation values")
    parser.add_argument('--dt_layers', default="1,2",
                        help="layers to use features for (if and when training DT)")
    parser.add_argument('--constrained', action="store_true",
                        help="Constrain amount of noise added")
    parser.add_argument('--start_natural', action="store_true",
                        help="Start with natural images, but better criteria")
    args = parser.parse_args()

    if args.start_natural and args.use_normal:
        raise ValueError("Can't start with nautral images to optimize and not use them")	

    flash_utils(args)

    if args.use_normal and args.upscale > 1:
        raise ValueError("Cannot have upscale and normal data init together")
    main(args)
