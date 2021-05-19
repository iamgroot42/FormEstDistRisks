import utils
from data_utils import SUPPORTED_PROPERTIES
from model_utils import get_models_path, get_model_representations
import seaborn as sns
import pandas as pd
import argparse
import numpy as np
import torch as ch

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


# WHY ARE RESULTS SUDDENLY SO EFFING BAD?!?!?!?
# NOTHING MAKES SENSE!!


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample', type=int, default=700,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=50,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--ntimes', type=int, default=5,
                        help='number of repetitions for multimode')
    parser.add_argument('--filename', type=str, default="graph",
                        help='desired title for plot, sep by _')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--regression', action="store_true")
    args = parser.parse_args()
    utils.flash_utils(args)

    binary = True

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    # targets = filter(lambda x: x != "0.5", os.listdir(
    #     get_models_path(args.filter, "adv")))
    targets = ["0.8"]

    # Load up positive-label test, test data
    pos_w, pos_labels, _ = get_model_representations(
        get_models_path(args.filter, "adv", "0.5"), 1)
    pos_w_test, pos_labels_test, dims = get_model_representations(
        get_models_path(args.filter, "victim", "0.5"), 1)

    data = []
    columns = [
        "Ratio of females in dataset that model is trained on",
        "Meta-classifier accuracy (%) differentiating between models"
        # "Ratio of dataset satisfying property",
        # "Accuracy on unseen models"
    ]
    for tg in targets:

        # Load up negative-label train, test data
        neg_w, neg_labels, _ = get_model_representations(
                get_models_path(args.filter, "adv", tg), 0)
        neg_w_test, neg_labels_test, _ = get_model_representations(
            get_models_path(args.filter, "victim", tg), 0)

        # Generate test set
        X_te = np.concatenate((pos_w_test, neg_w_test))
        Y_te = ch.cat((pos_labels_test, neg_labels_test)).cuda()

        print("Batching data: hold on")
        X_te = utils.prepare_batched_data(X_te)

        for _ in range(args.ntimes):
            # Random shuffles
            shuffled_1 = np.random.permutation(len(pos_labels))
            pp_x = pos_w[shuffled_1[:args.train_sample]]
            pp_y = pos_labels[shuffled_1[:args.train_sample]]

            shuffled_2 = np.random.permutation(len(neg_labels))
            np_x = neg_w[shuffled_2[:args.train_sample]]
            np_y = neg_labels[shuffled_2[:args.train_sample]]

            # Combine them together
            X_tr = np.concatenate((pp_x, np_x))
            Y_tr = ch.cat((pp_y, np_y))

            val_data = None
            if args.val_sample > 0:
                pp_val_x = pos_w[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_x = neg_w[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                pp_val_y = pos_labels[
                    shuffled_1[
                        args.train_sample:args.train_sample+args.val_sample]]
                np_val_y = neg_labels[
                    shuffled_2[
                        args.train_sample:args.train_sample+args.val_sample]]

                # Combine them together
                X_val = np.concatenate((pp_val_x, np_val_x))
                Y_val = ch.cat((pp_val_y, np_val_y))

                # Batch layer-wise inputs
                print("Batching data: hold on")
                X_val = utils.prepare_batched_data(X_val)
                Y_val = Y_val.float()

                val_data = (X_val, Y_val)

            metamodel = utils.PermInvModel(dims)
            metamodel = metamodel.cuda()
            metamodel = ch.nn.DataParallel(metamodel)

            # Float data
            Y_tr = Y_tr.float()
            Y_te = Y_te.float()

            # Batch layer-wise inputs
            print("Batching data: hold on")
            X_tr = utils.prepare_batched_data(X_tr)

            clf, tacc = utils.train_meta_model(
                         metamodel,
                         (X_tr, Y_tr),
                         (X_te, Y_te),
                         epochs=200, binary=binary,
                         regression=args.regression,
                         lr=1e-3, batch_size=args.batch_size,
                         val_data=val_data,
                         combined=True,
                         eval_every=10, gpu=True)

            data.append([float(tg), tacc])
            print("Test accuracy: %.3f" % data[-1][1])

    # Add dividing line
    plt.axvline(x=2.5, color='w', linewidth=1.0, linestyle='--')

    # Construct dataframe for boxplots
    df = pd.DataFrame(data, columns=columns)
    sns_plot = sns.boxplot(
        x=columns[0],
        y=columns[1],
        data=df)

    # plt.ylim(0.45, 1.0)
    sns_plot.figure.savefig("../visualize/%s.png" % args.filename)
