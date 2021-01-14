import utils
import seaborn as sns
import pandas as pd

import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from joblib import load
import os

import torch.nn as nn
import torch.optim as optim
import torch as ch

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def acc_fn(x, y):
    return ch.sum((y == (x >= 0)))


def get_outputs(model, X, no_grad=False):
    outputs = []
    for x in X:
        if no_grad:
            with ch.no_grad():
                outputs.append(model(x)[:, 0])
        else:
            outputs.append(model(x)[:, 0])
    outputs = ch.cat(outputs, 0)
    return outputs


def train_meta_pin(train_data, lr=1e-3, epochs=20, verbose=False):
    model = utils.PermInvModel([90, 60, 30, 30])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    params, y = train_data
    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator)

    # Start training
    model.train()
    for e in iterator:
        outputs = get_outputs(model, params)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y.float())

        loss.backward()
        optimizer.step()

        num_samples = outputs.shape[0]
        loss = loss.item() * num_samples
        running_acc = acc_fn(outputs, y)

        if verbose:
            iterator.set_description("Epoch %d : [Train] Loss: %.5f "
                                     "Accuacy: %.2f" % (
                                         e, loss / num_samples,
                                         100 * running_acc / num_samples))

    # Set back to evaluation mode and return
    model.eval()
    return model


def get_models(folder_path, label, collect_all=False):
    models_in_folder = os.listdir(folder_path) #[:50]
    # np.random.shuffle(models_in_folder)
    w, labels = [], []
    for path in tqdm(models_in_folder):
        clf = load(os.path.join(folder_path, path))

        # Look at weights linked to 'sex:Female' as well as 'sex:Male'
        _, _, cols = ci.load_data()

        # Look at initial layer weights, biases
        if collect_all:
            # Extract model parameters
            weights = [ch.from_numpy(x) for x in clf.coefs_]
            biases = [ch.from_numpy(x) for x in clf.intercepts_]
            processed = [ch.cat((w, ch.unsqueeze(b, 0)), 0).float().T
                         for (w, b) in zip(weights, biases)]
        else:
            processed = clf.coefs_[0]
            processed = np.concatenate((
                # np.mean(processed, 1),
                # np.mean(processed ** 2, 1),
                # np.std(processed, 1),
                np.mean(processed, 1),
                np.mean(processed ** 2, 1),
                np.std(processed, 1),
            ))

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    if collect_all:
        w = np.array(w, dtype=object)
        labels = ch.from_numpy(labels)
    else:
        w = np.array(w)

    return w, labels


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='',
                        help='path to folder of models')
    parser.add_argument('--test_path', type=str, default='',
                        help='path to folder of models')
    parser.add_argument('--sample', type=int, default=0,
                        help='how many models to use for meta-classifier')
    parser.add_argument('--ntimes', type=int, default=5,
                        help='number of repetitions for multimode')
    parser.add_argument('--plot_title', type=str, default="",
                        help='desired title for plot, sep by _')
    parser.add_argument('--collect_all', type=bool, default=False,
                        help='use all layer weights, like PIN?')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Census Income dataset
    ci = utils.CensusIncome("./census_data/")

    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    d_0 = "0.5"
    targets = filter(lambda x: x != "0.5", os.listdir(args.path))
    # targets = list(targets)[:1]

    # Load up positive-label train data
    pos_w, pos_labels = get_models(os.path.join(
        args.test_path, "0.5"), 1, collect_all=args.collect_all)

    # Load up positive-label test data
    pos_w_test, pos_labels_test = get_models(os.path.join(
        args.path, "0.5"), 1, collect_all=args.collect_all)

    data = []
    columns = [
        "Ratio of dataset satisfying property",
        "Accuracy on unseen models"
    ]
    for tg in targets:
        # Load up negative-label train data
        neg_w, neg_labels = get_models(os.path.join(
            args.path, tg), 0, collect_all=args.collect_all)

        # Load up negative-label test data
        neg_w_test, neg_labels_test = get_models(os.path.join(
            args.test_path, tg), 0, collect_all=args.collect_all)

        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))
        if args.collect_all:
            Y_te = ch.cat((pos_labels_test, neg_labels_test))
        else:
            Y_te = np.concatenate((pos_labels_test, neg_labels_test))

        for _ in range(args.ntimes):
            # Random shuffles
            rs = np.random.permutation(len(pos_labels))[:args.sample]

            # Pick data accordingly
            pp_x, pp_y = pos_w[rs], pos_labels[rs]
            np_x, np_y = neg_w[rs], neg_labels[rs]

            # Combine them together
            X_tr = np.concatenate((pp_x, np_x))
            if args.collect_all:
                Y_tr = ch.cat((pp_y, np_y))
            else:
                Y_tr = np.concatenate((pp_y, np_y))

            # Train meta-classifier on this
            # Record performance on test data
            if args.collect_all:
                clf = train_meta_pin((X_tr, Y_tr), lr=1e-3, epochs=50)
                acc = acc_fn(get_outputs(clf, X_te, no_grad=True),
                             Y_te).numpy() / len(Y_te)
                data.append([float(tg), acc])
            else:
                clf = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=1000)
                clf.fit(X_tr, Y_tr)
                data.append([float(tg), clf.score(X_te, Y_te)])

    # Construct dataframe for boxplots
    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x="Ratio of dataset satisfying property",
        y="Accuracy on unseen models",
        data=df)

    # plt.ylim(0.45, 1.0)
    # plt.title(" ".join(args.plot_title.split('_')))
    sns_plot.figure.savefig("../visualize/alpha_varying_census_meta.png")
