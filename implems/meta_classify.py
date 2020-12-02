import numpy as np
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import utils


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='census',
                        help='which dataset to work on (census/mnist/celeba)')
    parser.add_argument('--path1', type=str, default='',
                        help='path to first folder of models')
    parser.add_argument('--path2', type=str, default='',
                        help='path to second folder of models')
    parser.add_argument('--sample', type=int, default=0,
                        help='how many models to use for meta-classifier')
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.dataset == 'census':
        # Census Income dataset
        if args.sample < 1: raise ValueError(
            "At least one model must be used!")

        paths = [args.path1, args.path2]
        ci = utils.CensusIncome("./census_data/")

        w, b = [], []
        labels = []

        for i, path_seg in enumerate(paths):
            models_in_folder = os.listdir(path_seg)
            np.random.shuffle(models_in_folder)
            models_in_folder = models_in_folder[:args.sample]
            for path in models_in_folder:
                clf = load(os.path.join(path_seg, path))

                # Look at initial layer weights, biases
                processed = clf.coefs_[0]
                processed = np.concatenate(
                    (np.mean(processed, 1), np.mean(processed ** 2, 1)))
                w.append(processed)
                b.append(clf.intercepts_[0])
                labels.append(i)

        w = np.array(w)
        b = np.array(w)
        labels = np.array(labels)

        clf = MLPClassifier(hidden_layer_sizes=(30, 30), max_iter=500)
        X_train, X_test, y_train, y_test = train_test_split(
            w, labels, test_size=0.7)
        clf.fit(X_train, y_train)
        print("Meta-classifier performance on train data: %.2f" % clf.score(X_train, y_train))
        print("Meta-classifier performance on test data: %.2f" % clf.score(X_test, y_test))


        # Plot score distributions on test data
        labels = ['Trained on $D_0$', 'Trained on $D_1$']
        params = {'mathtext.default': 'regular'}
        plt.rcParams.update(params)
        zeros = np.nonzero(y_test == 0)[0]
        ones = np.nonzero(y_test == 1)[0]

        score_distrs = clf.predict_proba(X_test)[:, 1]

        plt.hist(score_distrs[zeros], 100, label=labels[0], alpha=0.9)
        plt.hist(score_distrs[ones], 100, label=labels[1], alpha=0.9)

        plt.title("Metal-classifier score prediction distributions for unseen models")
        plt.xlabel("Meta-classifier $Pr$[trained on $D_1$]")
        plt.ylabel("Number of models")
        plt.legend()

        plt.savefig("../visualize/census_meta_scores.png")

    else:
        raise ValueError("Support for this dataset not added yet")
