import utils
import data_utils
from model_utils import get_models_path, load_model
import seaborn as sns
import pandas as pd

import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
import os

import torch.nn as nn
import torch.optim as optim
import torch as ch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def params_to_gpu(params):
    for i in range(len(params)):
        params[i] = [x.clone().cuda() for x in params[i]]


def get_outputs(model, X, no_grad=False):

    with ch.set_grad_enabled(not no_grad):
        outputs = model(X)

    return outputs[:, 0]


def train_meta_pin(train_data, lr=1e-3, epochs=20, verbose=True, gpu=False):
    model = utils.PermInvModel([90, 32, 16, 8])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    if gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
        model = nn.DataParallel(model.cuda())

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
        running_acc = utils.acc_fn(outputs, y)

        if verbose:
            iterator.set_description("Epoch %d : [Train] Loss: %.5f "
                                     "Accuacy: %.2f" % (
                                         e + 1, loss / num_samples,
                                         100 * running_acc / num_samples))

    # Set back to evaluation mode and return
    model.eval()
    return model


# Load models from directory, return feature representations for meta-classifier
def get_models(folder_path, label):
    models_in_folder = os.listdir(folder_path)
    # np.random.shuffle(models_in_folder)
    w, labels = [], []
    for path in tqdm(models_in_folder):
        clf = load_model(os.path.join(folder_path, path))

        # Look at weights linked to 'sex:Female' as well as 'sex:Male'
        _, _, cols = ci.load_data()

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


def conditional_load(path1, path2):
    if os.path.exists(path1) and os.path.exists(path2):
        w = np.load(path1, allow_pickle=True)
        labels = np.load(path2)
        return w, labels
    else:
        return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=0,
                        help='# models (per label) to use for meta-classifier')
    parser.add_argument('--ntimes', type=int, default=5,
                        help='number of repetitions for multimode')
    parser.add_argument('--filename', type=str, default="graph",
                        help='desired title for plot, sep by _')
    parser.add_argument('--save_prefix', default="./loaded_census_models/",
                        help='default basepath to store loaded model weights in')
    parser.add_argument('--filter', choices=data_utils.SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--gpu', action="store_true",
                        help='use GPU for training PIM?')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Set dark background
    plt.style.use('dark_background')

    # Census Income dataset
    prefix = os.path.join(args.save_prefix, args.filter)
    ci = data_utils.CensusIncome()

    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    targets = filter(lambda x: x != "0.5", os.listdir(
        get_models_path(args.filter, "adv")))

    # Load up positive-label test data
    tup = conditional_load(os.path.join(
        prefix, "test_0.5_W.npy"),
        os.path.join(prefix, "test_0.5_labels.npy"))
    if tup is not None:
        pos_w_test, pos_labels_test = tup
        pos_labels_test = ch.tensor(pos_labels_test)
        print("Loaded from memory!")
    else:
        pos_w_test, pos_labels_test = get_models(get_models_path(
            args.filter, "victim", "0.5"), 1)
        # Save for later use
        np.save(os.path.join(prefix, "test_0.5_W"), pos_w_test)
        np.save(os.path.join(prefix, "test_0.5_labels"), pos_labels_test)

    # Load up positive-label train data
    tup = conditional_load(os.path.join(
        prefix, "train_0.5_W.npy"),
        os.path.join(prefix, "train_0.5_labels.npy"))
    if tup is not None:
        pos_w, pos_labels = tup
        pos_labels = ch.tensor(pos_labels)
        print("Loaded from memory!")
    else:
        pos_w, pos_labels = get_models(get_models_path(
            args.filter, "adv", "0.5"), 1)
        # Save for later use
        np.save(os.path.join(prefix, "train_0.5_W"), pos_w)
        np.save(os.path.join(prefix, "train_0.5_labels"), pos_labels)

    data = []
    columns = [
        "Ratio of females in dataset that model is trained on",
        "Meta-classifier accuracy (%) differentiating between models"
        # "Ratio of dataset satisfying property",
        # "Accuracy on unseen models"
    ]
    for tg in targets:

        # Load up negative-label train data
        tup = conditional_load(os.path.join(
            prefix, "train_%s_W.npy" % tg),
            os.path.join(prefix, "train_%s_labels.npy" % tg))
        if tup is not None:
            neg_w, neg_labels = tup
            neg_labels = ch.tensor(neg_labels)
            print("Loaded from memory!")
        else:
            neg_w, neg_labels = get_models(get_models_path(
                args.filter, "adv", tg), 0)
            # Save for later use
            np.save(os.path.join(prefix, "train_%s_W" % tg), neg_w)
            np.save(os.path.join(prefix, "train_%s_labels" %
                                 tg), neg_labels)

        # Load up negative-label test data
        tup = conditional_load(os.path.join(
            prefix, "test_%s_W.npy" % tg),
            os.path.join(prefix, "test_%s_labels.npy" % tg))
        if tup is not None:
            neg_w_test, neg_labels_test = tup
            neg_labels_test = ch.tensor(neg_labels_test)
            print("Loaded from memory!")
        else:
            neg_w_test, neg_labels_test = get_models(get_models_path(
                args.filter, "victim", tg), 0)
            # Save for later use
            np.save(os.path.join(prefix, "test_%s_W" % tg), neg_w_test)
            np.save(os.path.join(prefix, "test_%s_labels" %
                                 tg), neg_labels_test)

        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))

        if args.gpu:
            params_to_gpu(X_te)
            # Batch layer-wise inputs
        print("Batching data: hold on")
        X_te = utils.prepare_batched_data(X_te)

        if args.collect_all:
            Y_te = ch.cat((pos_labels_test, neg_labels_test))
            if args.gpu:
                Y_te = Y_te.cuda()
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
                if args.gpu:
                    Y_tr = Y_tr.cuda()
            else:
                Y_tr = np.concatenate((pp_y, np_y))

            if args.gpu:
                params_to_gpu(X_tr)

            # Batch layer-wise inputs
            print("Batching data: hold on")
            X_tr = utils.prepare_batched_data(X_tr)

            # Train meta-classifier on this
            # Record performance on test data
            if args.collect_all:
                clf = train_meta_pin((X_tr, Y_tr), lr=1e-3,
                                     epochs=200, gpu=args.gpu)
                acc = utils.acc_fn(get_outputs(clf, X_te, no_grad=True), Y_te)
                if args.gpu:
                    acc = acc.cpu()
                acc = acc.numpy()
                data.append([float(tg), acc / len(Y_te)])
                print("Test accuracy: %.3f" % data[-1][1])
            else:
                clf = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=1000)
                clf.fit(X_tr, Y_tr)
                data.append([float(tg), clf.score(X_te, Y_te)])

    # Add dividing line
    plt.axvline(x=2.5, color='w', linewidth=1.0, linestyle='--')

    # Construct dataframe for boxplots
    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0],
        y=columns[1],
        data=df)

    # plt.ylim(0.45, 1.0)
    # plt.title(" ".join(args.plot_title.split('_')))
    sns_plot.figure.savefig("../visualize/%s.png" % args.filename)
