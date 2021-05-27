import utils
from data_utils import SUPPORTED_PROPERTIES
from model_utils import get_models_path, get_model_representations
import argparse
import numpy as np
import torch as ch
import os
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def epoch_strategy(tg, args):
    if args.filter == "race":
        return args.epochs if tg not in ["0.6", "0.7", "0.8"] else 70
    else:
        return args.epochs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sample', type=int, default=800,
                        help='# models (per label) to use for training')
    parser.add_argument('--val_sample', type=int, default=0,
                        help='# models (per label) to use for validation')
    parser.add_argument('--batch_size', type=int, default=1000)
    # Sex: 1000 epochs, 1e-3
    # Race: 500* epochs, 1e-3
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Number of epochs to train meta-classifier")
    parser.add_argument('--ntimes', type=int, default=10,
                        help='number of repetitions for multimode')
    parser.add_argument('--filter', choices=SUPPORTED_PROPERTIES,
                        help='name for subfolder to save/load data from')
    args = parser.parse_args()
    utils.flash_utils(args)

    d_0 = "0.5"
    # Look at all folders inside path
    # One by one, run 0.5 v/s X experiments
    targets = filter(lambda x: x != d_0, os.listdir(
        get_models_path(args.filter, "adv")))

    # Load up positive-label test, test data
    pos_w, pos_labels, _ = get_model_representations(
        get_models_path(args.filter, "adv", d_0), 1)
    pos_w_test, pos_labels_test, dims = get_model_representations(
        get_models_path(args.filter, "victim", d_0), 1)

    data = []
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

            metamodel = utils.PermInvModel(dims, dropout=0.5)
            metamodel = metamodel.cuda()
            metamodel = ch.nn.DataParallel(metamodel)

            # Float data
            Y_tr = Y_tr.float()
            Y_te = Y_te.float()

            # Batch layer-wise inputs
            print("Batching data: hold on")
            X_tr = utils.prepare_batched_data(X_tr)

            # Train PIM
            clf, tacc = utils.train_meta_model(
                         metamodel,
                         (X_tr, Y_tr), (X_te, Y_te),
                         epochs=epoch_strategy(tg, args),
                         binary=True, lr=1e-3,
                         regression=False,
                         batch_size=args.batch_size,
                         val_data=val_data, combined=True,
                         eval_every=10, gpu=True)

            data.append([float(tg), tacc])
            print("Test accuracy: %.3f" % data[-1][1])

    # Print data
    for tup in data:
        print(tup)
