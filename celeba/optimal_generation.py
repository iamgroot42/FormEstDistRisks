import torch as ch
import torch.nn as nn
from tqdm import tqdm
import argparse
from utils import flash_utils
import os
import numpy as np
from data_utils import SUPPORTED_PROPERTIES
from model_utils import get_model, BASE_MODELS_DIR
from kornia.geometry.transform import resize
from robustness.tools.vis_tools import show_image_row
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def gen_optimal(models, labels, sample_shape, n_samples,
                n_steps, step_size, latent_focus, upscale):
    # Generate set of query points such that
    # Their outputs (or simply activations produced by model)
    # Help an adversary differentiate between models
    # Trained on different distributions

    loss_fn = nn.MSELoss()

    if upscale:
        actual_shape = (sample_shape[0], sample_shape[1] //
                        upscale, sample_shape[2] // upscale)
        x_rand_data = ch.rand(*((n_samples,) + actual_shape)).cuda()
        x_eff = resize(x_rand_data, (sample_shape[1], sample_shape[2]))
        print(models[0](x_eff, latent=latent_focus).shape[1:])
    else:
        x_rand_data = ch.rand(*((n_samples,) + sample_shape)).cuda()
        print(models[0](x_rand_data, latent=latent_focus).shape[1:])

    iterator = tqdm(range(n_steps))
    # Focus on latent=4 for now
    for i in iterator:
        x_rand = ch.autograd.Variable(x_rand_data.clone(), requires_grad=True)

        if upscale:
            x_use = resize(x_rand, (sample_shape[1], sample_shape[2]))
        else:
            x_use = x_rand

        # Get representations from all models
        reprs = ch.stack([m(x_use, latent=latent_focus) for m in models], 0)
        # reprs_z = ch.mean(reprs[labels == 0], 0)
        # reprs_o = ch.mean(reprs[labels == 1], 0)
        reprs_z = ch.mean(reprs[labels == 0], 2)
        reprs_o = ch.mean(reprs[labels == 1], 2)
        const = 2.
        const_neg = 0.5
        # const = 3.
        # const_neg = 1.
        # loss = ch.mean((reprs_z) ** 2) + ch.mean((const - reprs_o) ** 2)
        # loss = ch.mean((const_neg + reprs_z) ** 2) + ch.mean((const - reprs_o) ** 2)
        loss = ch.mean((const - reprs_z) ** 2) + ch.mean((const_neg + reprs_o) ** 2)
        # reprs_z = ch.mean(1. * ch.mean(reprs[labels == 0], 0), 1)
        # reprs_o = ch.mean(1. * ch.mean(reprs[labels == 1], 0), 1)
        # loss = loss_fn(reprs_z, reprs_o)
        grad = ch.autograd.grad(loss, [x_rand])

        with ch.no_grad():
            zero_acts = ch.sum(1. * (reprs[labels == 0] > 0), 2)
            one_acts = ch.sum(1. * (reprs[labels == 1] > 0), 2)
            # l1 = ch.mean((reprs_z) ** 2)
            # l1 = ch.mean((const_neg + reprs_z) ** 2)
            # l2 = ch.mean((const - reprs_o) ** 2)
            l1 = ch.mean((const - reprs_z) ** 2)
            l2 = ch.mean((const_neg + reprs_o) ** 2)
            iterator.set_description("Loss: %.3f | ZA: %.1f | OA: %.1f | Loss(1): %.3f | Loss(2): %.3f" % (
                loss.item(), zero_acts.mean(), one_acts.mean(), l1, l2))

        with ch.no_grad():
            # x_rand_data += (step_size * grad[0])
            x_rand_data -= (step_size * grad[0])
            x_rand_data = ch.clamp(x_rand_data, -1, 1)

    return x_rand.clone().detach()


def get_all_models(dir, n_models):
    models = []
    files = os.listdir(dir)[:n_models]
    for pth in tqdm(files):
        m = get_model(os.path.join(dir, pth), fake_relu=True)
        models.append(m)
    return models


def main(args):
    train_dir_1 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.first))
    train_dir_2 = os.path.join(
        BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, args.second))
    test_dir_1 = os.path.join(
        BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, args.first))
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/%s/" %
                              (args.filter, args.second))

    X_train_1 = get_all_models(train_dir_1, args.n_models)
    X_train_2 = get_all_models(train_dir_2, args.n_models)
    Y_train = [0.] * len(X_train_1) + [1.] * len(X_train_2)
    Y_train = ch.from_numpy(np.array(Y_train)).cuda()

    X_test_1 = get_all_models(test_dir_1, args.n_models)
    X_test_2 = get_all_models(test_dir_2, args.n_models)
    Y_test = [0.] * len(X_test_1) + [1.] * len(X_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()

    x_opt = []
    for i in range(args.n_samples):
        # Get optimal point based on local set
        x_opt_ = gen_optimal(X_train_1 + X_train_2, Y_train,
                             (3, 218, 178), 1,
                             args.steps, args.step_size,
                             args.latent_focus,
                             args.upscale)
        x_opt.append(x_opt_)
    x_opt = ch.cat(x_opt, 0)

    if args.upscale:
        x_use = resize(x_opt, (218, 178))
    else:
        x_use = x_opt

    # View resulting activation distribution for current models
    reprs_0 = ch.stack([m(x_use, latent=args.latent_focus)
                        for m in X_train_1], 0)
    reprs_1 = ch.stack([m(x_use, latent=args.latent_focus)
                        for m in X_train_2], 0)
    # Count number of neuron activations
    reprs_0 = ch.mean(1. * ch.sum(reprs_0 > 0, 2), 1).cpu().numpy()
    reprs_1 = ch.mean(1. * ch.sum(reprs_1 > 0, 2), 1).cpu().numpy()
    # Plot differences
    plt.plot(np.arange(len(X_train_1)), sorted(
        reprs_0), label=r'Trained on $D_0$')
    plt.plot(np.arange(len(X_train_2)), sorted(
        reprs_1), label=r'Trained on $D_1$')
    plt.legend()
    plt.savefig("./on_train_distr.png")
    plt.clf()

    # Test on unseen models
    reprs_0 = ch.stack([m(x_use, latent=args.latent_focus)
                        for m in X_test_1], 0)
    reprs_1 = ch.stack([m(x_use, latent=args.latent_focus)
                        for m in X_test_2], 0)
    # Count number of neuron activations
    reprs_0 = ch.mean(1. * ch.sum(reprs_0 > 0, 2), 1).cpu().numpy()
    reprs_1 = ch.mean(1. * ch.sum(reprs_1 > 0, 2), 1).cpu().numpy()
    # Plot differences
    plt.plot(np.arange(len(X_test_1)), sorted(
        reprs_0), label=r'Trained on $D_0$')
    plt.plot(np.arange(len(X_test_2)), sorted(
        reprs_1), label=r'Trained on $D_1$')
    plt.legend()
    plt.savefig("./on_test_distr.png")
    plt.clf()

    # Plot images
    x_opt_show = (x_opt.cpu() + 1) / 2
    show_image_row([x_opt_show], filename='./see_samples.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--latent_focus', type=int, default=4)
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        default="Male", choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--n_models', type=int, default=20)
    parser.add_argument('--step_size', type=float, default=1e-2)
    parser.add_argument('--first', help="Ratio for D_0", default="0.5")
    parser.add_argument('--second', help="Ratio for D_1")
    parser.add_argument('--upscale', type=int,
                        default=1, help="optimize and upscale")
    args = parser.parse_args()

    flash_utils(args)
    main(args)
