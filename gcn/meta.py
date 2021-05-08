from data_utils import ArxivNodeDataset
import torch as ch
import argparse
import numpy as np
from model_utils import get_model_features
from utils import PermInvModel, train_meta_model


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--degrees', default="9,10,11,12,13,14,15,16,17")
    parser.add_argument('--regression', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    args = parser.parse_args()
    print(args)

    # Get dataset ready (only need meta-data from this object)
    ds = ArxivNodeDataset('adv')

    degrees = args.degrees.split(",")
    binary = len(degrees) == 2

    # Directories where saved models are stored
    train_dirs = ["models/adv/deg" + x for x in degrees]
    test_dirs = ["models/victim/deg" + x for x in degrees]

    # Load models, convert to features
    train_vecs, test_vecs = [], []
    for trd, ted in zip(train_dirs, test_dirs):
        dims, vecs_train = get_model_features(
            trd, ds, args, max_read=700)
            # trd, ds, args, max_read=50)
        _, vecs_test = get_model_features(
            ted, ds, args, max_read=840)
            # ted, ds, args, max_read=50)

        train_vecs.append(vecs_train)
        test_vecs.append(vecs_test)

    # Ready train, test data
    Y_train, Y_test = [], []
    X_train, X_test = [], []
    for i, (vtr, vte) in enumerate(zip(train_vecs, test_vecs)):
        i_ = i
        if args.regression:
            i_ = float(degrees[i_])
        elif binary:
            i_ = float(i_)
        Y_train.append([i_] * len(vtr))
        Y_test.append([i_] * len(vte))

        X_train += vtr
        X_test += vte

    Y_train = ch.from_numpy(np.concatenate(Y_train))
    Y_test = ch.from_numpy(np.concatenate(Y_test))

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if binary or args.regression:
        Y_train = Y_train.float()
        Y_test = Y_test.float()

    if args.gpu:
        Y_train = Y_train.cuda()
        Y_test = Y_test.cuda()

    # First experiment: shuffle labels and use those to train
    # np.random.shuffle(Y_train)

    # Second experiment- run as a n-class classification problem
    # Cells added/modified above

    # Set seed for weight init
    ch.manual_seed(2021)

    # Train meta-classifier model
    if binary or args.regression:
        metamodel = PermInvModel(dims)
    else:
        metamodel = PermInvModel(dims, n_classes=len(degrees))

    # Split across GPUs if flag specified
    if args.gpu:
        metamodel = metamodel.cuda()

    metamodel = train_meta_model(metamodel,
                                 (X_train, Y_train),
                                 (X_test, Y_test),
                                 # epochs=40,
                                 epochs=70,
                                 binary=binary,
                                 regression=args.regression,
                                 # lr=0.01,
                                 lr=0.001,
                                 batch_size=args.batch_size,
                                 eval_every=5)

    # Sav emeta-model
    ch.save(metamodel.state_dict(), "./metamodel.pth")


if __name__ == "__main__":
    main()
