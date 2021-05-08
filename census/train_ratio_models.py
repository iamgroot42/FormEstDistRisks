from tqdm import tqdm
import os
import data_utils
import model_utils
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
    parser.add_argument('--split', choices=["adv", "victim", "all"],
                        help='which split of data to use')
    parser.add_argument('--verbose', action="store_true",
                        help='print out per-classifier stats?')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='number of iterations to train MLP for')
    parser.add_argument('--subsample_ratio', type=float, default=1.,
                        help='use sample of data')
    args = parser.parse_args()
    utils.flash_utils(args)

    # Get appropriate data filter
    data_filter = data_utils.get_default_filter(
        args.filter, args.ratio, args.verbose)

    # Census Income dataset
    ds = data_utils.CensusWrapper(data_filter)

    iterator = range(1, args.num + 1)
    if not args.verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        if args.verbose:
            print("Training classifier %d" % i)

        # Sample to qualify ratio, ultimately coming from fixed split
        # Ensures non-overlapping data for target and adversary
        # All the while allowing variations in dataset locally
        (x, y), _, cols = ds.load_data(data_filter,
                                       split=args.split,
                                       test_ratio=args.split_ratio,
                                       sample_ratio=args.subsample_ratio)

        clf = model_utils.get_model(max_iter=args.max_iter)
        clf.fit(x, y.ravel())
        train_acc = 100 * clf.score(x, y.ravel())
        test_acc = 100 * clf.best_validation_score_
        if args.verbose:
            print("Classifier %d : Train acc %.2f , Test acc %.2f\n" %
                  (i, train_acc, test_acc))

        save_path = os.path.join(args.savepath,  str(i) + "_%.2f" % test_acc)
        model_utils.save_model(clf, save_path)
