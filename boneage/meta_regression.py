import torch as ch
import numpy as np
import os
import argparse
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model


def load_stuff(model_dir, args):
    dims, vecs = get_model_features(model_dir, first_n=args.first_n)
    return dims, np.array(vecs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_sample', type=int, default=700)
    parser.add_argument('--val_sample', type=int, default=50)
    parser.add_argument('--first_n', type=int, default=3,
                        help="Only consider first N layers")
    args = parser.parse_args()
    print(args)

    # Keep track of ratios to be used
    ratios = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"]
    train_dirs = [
        os.path.join(BASE_MODELS_DIR, "victim/%s/" % z) for z in ratios]
    test_dirs = [
        os.path.join(BASE_MODELS_DIR, "adv/%s/" % z) for z in ratios]

    X_train, X_test = [], []
    X_val, Y_val = [], []
    Y_train, Y_test = [], []
    for ratio in ratios:
        # Load model weights, convert to features
        train_dir = os.path.join(BASE_MODELS_DIR, "victim/%s/" % ratio)
        test_dir = os.path.join(BASE_MODELS_DIR, "victim/%s/" % ratio)
        dims, vecs_train = load_stuff(train_dir, args)
        _, vecs_test = load_stuff(test_dir, args)

        # Create train/val split
        shuffled = np.random.permutation(len(vecs_train))
        vecs_train_use = vecs_train[shuffled[:args.train_sample]]
        vecs_val = vecs_train[
            shuffled[args.train_sample:args.train_sample+args.val_sample]]

        # Keep collecting data...
        X_train.append(vecs_train_use)
        X_val.append(vecs_val)
        X_test.append(vecs_test)

        # ...and labels
        Y_train += [float(ratio)] * len(vecs_train_use)
        Y_val += [float(ratio)] * len(vecs_val)
        Y_test += [float(ratio)] * len(vecs_test)

    # Prepare for PIM
    X_train = np.concatenate(X_train)
    X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)

    Y_train = ch.from_numpy(np.array(Y_train)).float().cuda()
    Y_val = ch.from_numpy(np.array(Y_val)).float().cuda()
    Y_test = ch.from_numpy(np.array(Y_test)).float().cuda()

    # Train meta-classifier model
    metamodel = PermInvModel(dims)
    metamodel = metamodel.cuda()

    _, tloss = train_meta_model(
                     metamodel,
                     (X_train, Y_train),
                     (X_test, Y_test),
                     epochs=150, binary=True,
                     lr=0.001, batch_size=args.batch_size,
                     val_data=(X_val, Y_val),
                     regression=True,
                     eval_every=10, gpu=True)
    print("Test loss %.4f" % (tloss))

    # Save meta-model
    ch.save(metamodel.state_dict(), "./metamodel_%d_%.3f.pth" %
            (args.first_n, tloss))

    # i=3: 0.011, 0.0026, 0.0044,
    # i=2: 0.0038, 0.0022, 0.0023,
    # i=1: 0.0028, 0.0042, 0.0034,
