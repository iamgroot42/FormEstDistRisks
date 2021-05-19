import torch as ch
import numpy as np
import os
import argparse
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--n_tries', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1200)
    parser.add_argument('--train_sample', type=int, default=700)
    parser.add_argument('--val_sample', type=int, default=50)
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Only consider first N layers")
    parser.add_argument('--first')
    parser.add_argument('--second')
    args = parser.parse_args()
    print(args)

    train_dir_1 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % args.first)
    train_dir_2 = os.path.join(BASE_MODELS_DIR, "victim/%s/" % args.second)
    test_dir_1 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % args.first)
    test_dir_2 = os.path.join(BASE_MODELS_DIR, "adv/%s/" % args.second)
    features_avail = False

    # Load models, convert to features
    dims, vecs_train_1 = get_model_features(train_dir_1, first_n=args.first_n)
    _, vecs_train_2 = get_model_features(train_dir_2, first_n=args.first_n)

    _, vecs_test_1 = get_model_features(test_dir_1, first_n=args.first_n)
    _, vecs_test_2 = get_model_features(test_dir_2, first_n=args.first_n)

    vecs_train_1 = np.array(vecs_train_1)
    vecs_train_2 = np.array(vecs_train_2)

    Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
    Y_test = ch.from_numpy(np.array(Y_test)).cuda()
    X_test = vecs_test_1 + vecs_test_2
    X_test = np.array(X_test)

    accs = []
    for i in range(args.n_tries):

        shuffled_1 = np.random.permutation(len(vecs_train_1))
        vecs_train_1_use = vecs_train_1[shuffled_1[:args.train_sample]]

        shuffled_2 = np.random.permutation(len(vecs_train_2))
        vecs_train_2_use = vecs_train_2[shuffled_2[:args.train_sample]]

        val_data = None
        if args.val_sample > 0:
            vecs_val_1 = vecs_train_1[
                shuffled_1[
                    args.train_sample:args.train_sample+args.val_sample]]
            vecs_val_2 = vecs_train_2[
                shuffled_2[
                    args.train_sample:args.train_sample+args.val_sample]]
            X_val = np.concatenate((vecs_val_1, vecs_val_2))

            Y_val = [0.] * len(vecs_val_1) + [1.] * len(vecs_val_2)
            Y_val = ch.from_numpy(np.array(Y_val)).cuda()
            val_data = (X_val, Y_val)

        # Ready train, test data
        Y_train = [0.] * len(vecs_train_1_use) + [1.] * len(vecs_train_2_use)
        Y_train = ch.from_numpy(np.array(Y_train)).cuda()
        X_train = np.concatenate((vecs_train_1_use, vecs_train_2_use))

        # Train meta-classifier model
        metamodel = PermInvModel(dims)
        metamodel = metamodel.cuda()

        _, test_acc = train_meta_model(
            metamodel,
            (X_train, Y_train),
            (X_test, Y_test),
            epochs=80, binary=True,
            lr=0.001, batch_size=args.batch_size,
            val_data=val_data,
            eval_every=10, gpu=True)
        accs.append(test_acc)
        print("Run %d: %.2f" % (i+1, test_acc))

    print(accs)


# Test accuracies
# Ratio | i = 1 | i = 2 | i = 3
# 0.2 [98.7, 99.7, 99.75, 99.6, 99.3] [99.75, 98.6, 99.1, 98.8, 99.8]
# 0.3 [92.7, 95.75, 95.65, 83.35, 93.7] [49.5, 95.9, 91.85, 92.45, 92.75]
# 0.4 [58.45, 76.35, 64.35, 50.0, 73.55] [50.05, 84.3, 61.65, 69.9, 50.4] ?
# 0.6 [55.75, 84.3, 52.95, 74.95, 52.65] [50.4, 77.45, 59.35, 56.95, 64.95] [57.3, 71.9, 51.15, 64.7, 61.55]
# 0.7 [88.2, 85.45, 89.75, 88.35, 88.75] [78.05, 85.25, 73.15, 79.65, 82.1] 
# 0.8 [98.4, 98.55, 96.4, 97.65, 98.8] [99.0, 97.2, 96.95, 95.2, 99.25]
