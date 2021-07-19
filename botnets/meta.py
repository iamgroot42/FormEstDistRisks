import torch as ch
import argparse
import numpy as np
import os
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model, prepare_batched_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Botnets-dataset (GCN)')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--train_sample', type=int, default=800)
    parser.add_argument('--val_sample', type=int, default=0)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_feat', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--regression', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--first_n', type=int, default=4,
                        help="Only consider first N layers")
    args = parser.parse_args()
    print(args)

    # Directories where saved models are stored
    binary = True
    train_dirs = [os.path.join(BASE_MODELS_DIR, "adv", x) for x in ["0", "1"]]
    # test_dirs = [os.path.join(BASE_MODELS_DIR, "victim", x) for x in ["0", "1"]]
    test_dirs = train_dirs

    # Load models, convert to features
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    for i, (trd, ted) in enumerate(zip(train_dirs, test_dirs)):
        dims, vecs_train = get_model_features(
            trd, args, max_read=args.train_sample + args.val_sample)
        _, vecs_test = get_model_features(
            ted, args, max_read=1000)

        # Split train into train+val
        vecs_train = vecs_train[args.val_sample:]

        X_train += vecs_train
        X_test += vecs_test
        if args.val_sample > 0:
            vecs_val = vecs_train[:args.val_sample]
            X_val += vecs_val

        # Prepare labels too
        i_ = i
        if args.regression:
            i_ = float(degrees[i_])
        elif binary:
            i_ = float(i_)

        Y_train.append([i_] * len(vecs_train))
        Y_test.append([i_] * len(vecs_test))

        if args.val_sample > 0:
            Y_val.append([i_] * len(vecs_val))

    X_train = np.array(X_train, dtype='object')
    X_test = np.array(X_test, dtype='object')

    Y_train = ch.from_numpy(np.concatenate(Y_train))
    Y_test = ch.from_numpy(np.concatenate(Y_test))

    print("Batching data: hold on")
    X_train = prepare_batched_data(X_train)
    X_test = prepare_batched_data(X_test)

    if args.val_sample > 0:
        Y_val = ch.from_numpy(np.concatenate(Y_val))
        X_val = np.array(X_val, dtype='object')
        X_val = prepare_batched_data(X_val)

    if binary or args.regression:
        Y_train = Y_train.float()
        Y_test = Y_test.float()
        if args.val_sample > 0:
            Y_val = Y_val.float()

    if args.gpu:
        Y_train = Y_train.cuda()
        Y_test = Y_test.cuda()
        if args.val_sample > 0:
            Y_val = Y_val.cuda()

    # Train meta-classifier model
    if binary or args.regression:
        metamodel = PermInvModel(dims)
    else:
        metamodel = PermInvModel(dims, n_classes=len(degrees))

    # Split across GPUs if flag specified
    if args.gpu:
        metamodel = metamodel.cuda()
        if args.parallel:
            metamodel = ch.nn.DataParallel(metamodel)

    if args.val_sample > 0:
        val_data = (X_val, Y_val)
    else:
        val_data = None

    metamodel, test_loss = train_meta_model(
        metamodel, (X_train, Y_train), (X_test, Y_test),
        epochs=args.iters, binary=binary, regression=args.regression,
        lr=0.01, batch_size=args.batch_size, eval_every=10,
        combined=True, val_data=val_data, gpu=args.gpu)

    print("[Test] Loss: %.4f" % test_loss)

    # Save meta-model
    ch.save(metamodel.state_dict(), "./metamodel_new_%.3f.pth" % test_loss)
