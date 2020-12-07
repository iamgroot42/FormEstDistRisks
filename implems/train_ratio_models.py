from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from joblib import dump
import os
import numpy as np
import utils


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savepath', type=str, default='',
                        help='folder where trained model(s) should be saved')
    parser.add_argument('--filter', type=str, default='',
                        help='while filter to use (sex/race/income/none)')
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='what ratio of the new sampled dataset should be true')
    parser.add_argument('--num', type=int, default=20,
                        help='how many classifiers to train?')
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help='split original data into two (ratio for second)')
    parser.add_argument('--on_first', type=bool, default=None,
                        help='train on first split?')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='print out per-classifier stats?')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Census Income dataset
    ci = utils.CensusIncome("./census_data/")
    ratio = args.ratio

    if args.filter == "sex":
        def data_filter(df): return utils.filter(
            df, lambda x: x['sex:Female'] == 1, ratio, args.verbose)  # 0.65
    elif args.filter == "race":
        def data_filter(df): return utils.filter(
            df, lambda x: x['race:White'] == 0,  ratio, args.verbose)  # 1.0
    elif args.filter == "income":
        def data_filter(df): return utils.filter(
            df, lambda x: x['income'] == 1, ratio, args.verbose)  # 0.5
    elif args.filter == "none":
        data_filter = None
    else:
        raise ValueError("Invalid filter requested")

    iterator = range(1, args.num + 1)
    if not args.verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        if args.verbose:
            print("Training classifier %d" % i)

        # Sample to qualify ratio, ultimately coming from fixed split
        # Ensures non-overlapping data for target and adversary
        # All the while allowing variations in dataset locally
        (x_tr, y_tr), (x_te, y_te), cols = ci.load_data(data_filter,
                                                    first=args.on_first,
                                                    test_ratio=args.split_ratio)
        clf = MLPClassifier(hidden_layer_sizes=(60, 30, 30), max_iter=200)
        clf.fit(x_tr, y_tr.ravel())
        train_acc = 100 * clf.score(x_tr, y_tr.ravel())
        test_acc = 100 * clf.score(x_te, y_te.ravel())
        if args.verbose:
            print("Classifier %d : Train acc %.2f , Test acc %.2f\n" %
                  (i, train_acc, test_acc))

        dump(clf, os.path.join(args.savepath,  str(i) + "_%.2f" % test_acc))
