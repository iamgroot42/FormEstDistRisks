import torch as ch
import argparse
import numpy as np
import os
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model, prepare_batched_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Botnets-dataset (GCN)')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--train_sample', type=int, default=400)
    parser.add_argument('--val_sample', type=int, default=500)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_feat', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--parallel', action="store_true")
    parser.add_argument('--residual_modification', action="store_true",
                        help="modification in weight parsing for residual connections")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Only consider first N layers")
    parser.add_argument('--start_n', type=int, default=0,
                        help="Only consider starting from this layer")
    args = parser.parse_args()
    print(args)

    # Directories where saved models are stored
    train_dirs = [os.path.join(BASE_MODELS_DIR, "adv", x) for x in ["0", "1"]]
    test_dirs = [os.path.join(BASE_MODELS_DIR, "victim", x) for x in ["0", "1"]]

    # Load models, convert to features
    X_train, X_val, X_test = [], [], []
    Y_train, Y_val, Y_test = [], [], []
    for i, (trd, ted) in enumerate(zip(train_dirs, test_dirs)):
        dims, vecs_train = get_model_features(
            trd, args, max_read=args.train_sample + args.val_sample,
            residual_modification=args.residual_modification)
        _, vecs_test = get_model_features(
            ted, args, max_read=1000,
            residual_modification=args.residual_modification)

        # Split train into train+val
        vecs_train = vecs_train[args.val_sample:]

        X_train += vecs_train
        X_test += vecs_test
        if args.val_sample > 0:
            vecs_val = vecs_train[:args.val_sample]
            X_val += vecs_val

        # Prepare labels too
        i_ = float(i)

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

    Y_train = Y_train.float()
    Y_test = Y_test.float()
    if args.val_sample > 0:
        Y_val = Y_val.float()

    if args.gpu:
        Y_train = Y_train.cuda()
        Y_test = Y_test.cuda()
        if args.val_sample > 0:
            Y_val = Y_val.cuda()

    if args.val_sample > 0:
        val_data = (X_val, Y_val)
    else:
        val_data = None

    # Train meta-classifier model
    metamodel = PermInvModel(dims)

    # Split across GPUs if flag specified
    if args.gpu:
        metamodel = metamodel.cuda()
        if args.parallel:
            metamodel = ch.nn.DataParallel(metamodel)

    metamodel, test_loss = train_meta_model(
        metamodel, (X_train, Y_train), (X_test, Y_test),
        epochs=args.iters, binary=True, regression=False,
        lr=0.001, batch_size=args.batch_size, eval_every=10,
        combined=True, val_data=val_data, gpu=args.gpu)

    print("[Test] Accuracy: %.4f" % test_loss)
