import utils
import seaborn as sns
import pandas as pd

import numpy as np
from tqdm import tqdm
from joblib import load
import os

import torch.nn as nn
import torch.optim as optim
import torch as ch

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def params_to_gpu(params):
    for i in range(len(params)):
        params[i] = [x.clone().cuda() for x in params[i]]


@ch.no_grad()
def acc_fn(x, y):
    return ch.sum((y == (x >= 0)))


def prepare_batched_data(X):
    inputs = [[] for _ in range(len(X[0]))]
    for x in X:
        for i, l in enumerate(x):
            inputs[i].append(l)

    inputs = [ch.stack(x, 0) for x in inputs]
    return inputs


def get_outputs(model, X, no_grad=False):

    with ch.set_grad_enabled(not no_grad):
        outputs = model(X)

    return outputs[:, 0]


def train_meta_pin(train_data, lr=1e-3, epochs=20, verbose=True, gpu=False, test=None):
    # model = utils.PermInvModel([90, 32, 16, 8], dropout=0.5)
    # model = utils.PermInvModel([90, 32], dropout=0.5)
    model = utils.PermInvModel([42, 32, 16, 8])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    if gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        model = nn.DataParallel(model.cuda())

    params, y = train_data

    if test is not None:
        params_test, y_test = test

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator)

    # Start training
    for e in iterator:
        # Clear gradients
        model.train()
        optimizer.zero_grad()

        # Get model output
        outputs = get_outputs(model, params)
        # Compute loss
        loss = loss_fn(outputs, y.float())

        # Take gradient step
        loss.backward()
        optimizer.step()

        # Compute train loss, accuracy
        num_samples = outputs.shape[0]
        loss = loss.item() * num_samples
        running_acc = acc_fn(outputs, y)

        # Compute test accuracy
        model.eval()
        t_acc = acc_fn(get_outputs(model, params_test, no_grad=True), y_test)
        t_acc = 100 * float(t_acc) / len(y_test)

        # Log information
        if verbose:
            iterator.set_description(
                "Epoch %d : [Train] Loss: %.5f "
                "[Train] Accuacy: %.2f | [Test] Accuracy: %.2f" % (
                    e + 1, loss / num_samples,
                    100 * running_acc / num_samples, t_acc))

    # Set back to evaluation mode and return
    model.eval()
    return model


def get_models(folder_path, label):
    models_in_folder = os.listdir(folder_path)
    # np.random.shuffle(models_in_folder)
    w, labels = [], []
    for path in tqdm(models_in_folder):
        clf = load(os.path.join(folder_path, path))

        # Extract model parameters
        weights = [ch.from_numpy(x) for x in clf.coefs_]
        biases = [ch.from_numpy(x) for x in clf.intercepts_]
        processed = [ch.cat((w, ch.unsqueeze(b, 0)), 0).float().T
                     for (w, b) in zip(weights, biases)]    

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    w = np.array(w, dtype=object)
    labels = ch.from_numpy(labels)

    return w, labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='',
                        help='path to folder of models')
    parser.add_argument('--test_path', type=str, default='',
                        help='path to folder of models')
    parser.add_argument('--sample', type=int, default=0,
                        help='# models (per label) to use for meta-classifier')
    parser.add_argument('--ntimes', type=int, default=5,
                        help='number of repetitions for multimode')
    parser.add_argument('--save_prefix', default="./loaded_census_models/",
                        help='default basepath to store loaded model weights in')
    parser.add_argument('--filter', choices=["sex", "race"],
                        help='name for subfolder to save/load data from')
    parser.add_argument('--gpu', action="store_true",
                        help='use GPU for training PIM?')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    # d_0 = "0.5"
    # d_0 = "0.0"
    d_0 = "0.5"
    # targets = filter(lambda x: x != "0.5", os.listdir(args.path))
    # targets = ["0.25", "0.4"]
    # targets = ["0.2", "0.3", "0.4"]
    # targets = ["0.8"]
    # targets = ["0.75"]
    # targets = ["0.87"]
    targets = ["0.2"]

    # Load up positive-label test data
    pos_w_test, pos_labels_test = get_models(os.path.join(
        args.test_path, d_0), 1)

    # Load up positive-label train data
    pos_w, pos_labels = get_models(os.path.join(
        args.path, d_0), 1)

    data = []
    columns = [
        "Ratio of dataset satisfying property",
        "Accuracy on unseen models"
    ]
    for tg in targets:

        # Load up negative-label train data
        neg_w, neg_labels = get_models(os.path.join(
            args.path, tg), 0)

        # Load up negative-label test data
        neg_w_test, neg_labels_test = get_models(os.path.join(
            args.test_path, tg), 0)

        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))

        if args.gpu:
            params_to_gpu(X_te)

        # Batch layer-wise inputs
        print("Batching data: hold on")
        X_te = prepare_batched_data(X_te)

        Y_te = ch.cat((pos_labels_test, neg_labels_test))
        if args.gpu:
            Y_te = Y_te.cuda()

        for _ in range(args.ntimes):
            # Random shuffles
            rs = np.random.permutation(min(len(neg_labels), len(pos_labels)))[
                :args.sample]

            # Pick data accordingly
            pp_x, pp_y = pos_w[rs], pos_labels[rs]
            np_x, np_y = neg_w[rs], neg_labels[rs]

            # Combine them together
            X_tr = np.concatenate((pp_x, np_x))
            Y_tr = ch.cat((pp_y, np_y))
            if args.gpu:
                Y_tr = Y_tr.cuda()

            if args.gpu:
                params_to_gpu(X_tr)

            # Batch layer-wise inputs
            print("Batching data: hold on")
            X_tr = prepare_batched_data(X_tr)

            # Train meta-classifier on this
            # Record performance on test data
            clf = train_meta_pin((X_tr, Y_tr),
                                 lr=1e-3,
                                #  epochs=200,
                                 epochs=500,
                                 gpu=args.gpu,
                                 test=(X_te, Y_te))
            acc = acc_fn(get_outputs(clf, X_te, no_grad=True), Y_te)
            if args.gpu:
                acc = acc.cpu()
            acc = acc.numpy()
            data.append([float(tg), acc / len(Y_te)])
            print("Test accuracy: %.3f" % data[-1][1])

    # Construct dataframe for boxplots
    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x="Ratio of dataset satisfying property",
        y="Accuracy on unseen models",
        data=df)

    # plt.ylim(0.45, 1.0)
    # plt.title(" ".join(args.plot_title.split('_')))
    sns_plot.figure.savefig("../visualize/here_meta.png")


# Race recreate:
# 99.5, 99.4, 99.6, 99.9, 98.7, 98.1, 99.2, 99.9, 100, 99.6

# Sex recreate:
# 64.9, 62.55, 58.95, 61.15, 62.95, 60.3, 61.6, 62.95, 63.05, 62.2
