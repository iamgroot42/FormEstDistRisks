from data_utils import ArxivNodeDataset
import torch as ch
import argparse
import numpy as np
import os
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_sample', type=int, default=700)
    parser.add_argument('--val_sample', type=int, default=50)
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
    train_dirs = [os.path.join(BASE_MODELS_DIR, "adv", "deg" + x)
                  for x in degrees]
    test_dirs = [os.path.join(BASE_MODELS_DIR, "victim", "deg" + x)
                 for x in degrees]

    # Load models, convert to features
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    for i, (trd, ted) in enumerate(zip(train_dirs, test_dirs)):
        dims, vecs_train = get_model_features(
            trd, ds, args, max_read=args.train_sample + args.val_sample)
        _, vecs_test = get_model_features(
            ted, ds, args, max_read=1000)

        # Split train into train+val
        vecs_val = vecs_train[:args.val_sample]
        vecs_train = vecs_train[args.val_sample:]

        X_train += vecs_train
        X_val += vecs_val
        X_test += vecs_test

        # Prepare labels too
        i_ = i
        if args.regression:
            i_ = float(degrees[i_])
        elif binary:
            i_ = float(i_)

        Y_train.append([i_] * len(vecs_train))
        Y_val.append([i_] * len(vecs_val))
        Y_test.append([i_] * len(vecs_test))

    X_train = np.array(X_train, dtype='object')
    X_val = np.array(X_val, dtype='object')
    X_test = np.array(X_test, dtype='object')

    Y_train = ch.from_numpy(np.concatenate(Y_train))
    Y_val = ch.from_numpy(np.concatenate(Y_val))
    Y_test = ch.from_numpy(np.concatenate(Y_test))

    if binary or args.regression:
        Y_train = Y_train.float()
        Y_val = Y_val.float()
        Y_test = Y_test.float()

    if args.gpu:
        Y_train = Y_train.cuda()
        Y_val = Y_val.cuda()
        Y_test = Y_test.cuda()

    # First experiment: shuffle labels and use those to train
    # np.random.shuffle(Y_train)

    # Second experiment- run as a n-class classification problem
    # Cells added/modified above

    # Train meta-classifier model
    if binary or args.regression:
        metamodel = PermInvModel(dims)
    else:
        metamodel = PermInvModel(dims, n_classes=len(degrees))

    # Split across GPUs if flag specified
    if args.gpu:
        metamodel = metamodel.cuda()

    metamodel, test_loss = train_meta_model(
                            metamodel,
                            (X_train, Y_train),
                            (X_test, Y_test),
                            epochs=200, binary=binary,
                            regression=args.regression,
                            # lr=0.001, batch_size=args.batch_size,
                            lr=0.01, batch_size=args.batch_size,
                            eval_every=10,
                            val_data=(X_val, Y_val),
                            gpu=args.gpu)

    print("[Test] Loss: %.4f" % test_loss)

    # Save meta-model
    ch.save(metamodel.state_dict(), "./metamodel_%.3f.pth" % test_loss)
