import torch as ch
import numpy as np
import os
import argparse
from model_utils import get_model_features, BASE_MODELS_DIR
from utils import PermInvModel, train_meta_model, flash_utils, test_meta
from data_utils import SUPPORTED_RATIOS


def load_stuff(model_dir, args):
    max_read = 5 if args.testing else None
    dims, vecs = get_model_features(model_dir, first_n=args.first_n,
                                    shift_to_gpu=False, max_read=max_read)
    return dims, np.array(vecs, dtype=object)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Boneage')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_sample', type=int, default=700)
    parser.add_argument('--val_sample', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--testing', action="store_true", help="Testing mode")
    parser.add_argument('--first_n', type=int, default=3,
                        help="Only consider first N layers")
    parser.add_argument('--eval_only', action="store_true",
                        help="Only loading up model and evaluating")
    parser.add_argument('--model_path', help="Path to saved model")
    args = parser.parse_args()
    flash_utils(args)

    if args.testing:
        num_train, num_val = 3, 2
    else:
        num_train, num_val = args.train_sample, args.val_sample

    X_train, X_test = [], []
    X_val, Y_val = [], []
    Y_train, Y_test = [], []
    num_per_dist = None
    for ratio in SUPPORTED_RATIOS:
        # Load model weights, convert to features
        train_dir = os.path.join(BASE_MODELS_DIR, "victim/%s/" % ratio)
        test_dir = os.path.join(BASE_MODELS_DIR, "victim/%s/" % ratio)
        if not args.eval_only:
            dims, vecs_train = load_stuff(train_dir, args)
            # Create train/val split
            shuffled = np.random.permutation(len(vecs_train))
            vecs_train_use = vecs_train[shuffled[:num_train]]
            vecs_val = vecs_train[
                shuffled[num_train:num_train+num_val]]

            X_train.append(vecs_train_use)
            X_val.append(vecs_val)
            Y_train += [float(ratio)] * len(vecs_train_use)
            Y_val += [float(ratio)] * len(vecs_val)

        dims, vecs_test = load_stuff(test_dir, args)
        X_test.append(vecs_test)
        Y_test += [float(ratio)] * len(vecs_test)

        # Make sure same number of models read per distribution
        if num_per_dist:
            assert num_per_dist == len(vecs_test)
        num_per_dist = len(vecs_test)

    # Prepare for PIM
    if not args.eval_only:
        X_train = np.concatenate(X_train)
        X_val = np.concatenate(X_val)
    X_test = np.concatenate(X_test)

    if not args.eval_only:
        Y_train = ch.from_numpy(np.array(Y_train)).float()
        Y_val = ch.from_numpy(np.array(Y_val)).float()
    Y_test = ch.from_numpy(np.array(Y_test)).float()

    # Train meta-classifier model
    metamodel = PermInvModel(dims)
    if args.eval_only:
        # Load model
        metamodel.load_state_dict(ch.load(args.model_path))
        # Evaluate
        metamodel = metamodel.cuda()
        loss_fn = ch.nn.MSELoss(reduction='none')
        _, losses, preds = test_meta(
            metamodel, loss_fn, X_test, Y_test.cuda(),
            args.batch_size, None,
            binary=True, regression=True, gpu=True,
            element_wise=True, get_preds=True)
        y_np = Y_test.numpy()
        losses = losses.numpy()
        print("Mean loss: %.4f" % np.mean(losses))
        # Get all unique ratios in GT, and their average losses from model
        ratios = np.unique(y_np)
        losses_dict = {}
        for ratio in ratios:
            losses_dict[ratio] = np.mean(losses[y_np == ratio])
        print(losses_dict)
        # Conctruct a matrix where every (i, j) entry is the accuracy
        # for ratio[i] v/s ratio [j], where whichever ratio is closer to the 
        # ratios is considered the "correct" one
        # Assume equal number of models per ratio, stored in order of
        # SUPPORTED_RATIOS
        acc_mat = np.zeros((len(ratios), len(ratios)))
        for i in range(acc_mat.shape[0]):
            for j in range(i + 1, acc_mat.shape[0]):
                # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
                gt_z = (Y_test[num_per_dist * i:num_per_dist * (i + 1)].numpy() == float(ratios[j]))
                gt_o = (Y_test[num_per_dist * j:num_per_dist * (j + 1)].numpy() == float(ratios[j]))
                # Get relevant preds
                pred_z = preds[num_per_dist * i:num_per_dist * (i + 1)]
                pred_o = preds[num_per_dist * j:num_per_dist * (j + 1)]
                pred_z = (pred_z >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
                pred_o = (pred_o >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
                # Compute accuracies and store
                acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate((pred_z, pred_o), 0)
                acc_mat[i, j] = np.mean(acc)
        print(acc_mat)
    else:
        metamodel = metamodel.cuda()

        batch_size = 10 if args.testing else args.batch_size
        _, tloss = train_meta_model(
                        metamodel,
                        (X_train, Y_train),
                        (X_test, Y_test),
                        epochs=args.epochs, binary=True,
                        lr=0.001, batch_size=batch_size,
                        val_data=(X_val, Y_val),
                        regression=True,
                        eval_every=10, gpu=True)
        print("Test loss %.4f" % (tloss))

        # Save meta-model
        ch.save(metamodel.state_dict(), "./metamodel_%d_%.3f.pth" %
                (args.first_n, tloss))
