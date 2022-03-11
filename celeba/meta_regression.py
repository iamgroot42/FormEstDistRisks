import torch as ch
import numpy as np
import os
import argparse
import utils
from data_utils import SUPPORTED_PROPERTIES, SUPPORTED_RATIOS
from model_utils import get_model_features, BASE_MODELS_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Celeb-A')
    parser.add_argument('--batch_size', type=int, default=150)
    parser.add_argument('--train_sample', type=int, default=700)
    parser.add_argument('--val_sample', type=int, default=50)
    parser.add_argument('--testing', action="store_true", help="Testing mode")
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        default="Male", choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--focus', choices=["fc", "conv", "combined"],
                        required=True, help="Which layer paramters to use")
    parser.add_argument('--eval_only', action="store_true",
                        help="Only loading up model and evaluating")
    parser.add_argument('--model_path', help="Path to saved model")
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.testing:
        num_train, num_val = 3, 2
        n_models = 5
        SUPPORTED_RATIOS = SUPPORTED_RATIOS[:3]
    else:
        num_train, num_val = args.train_sample, args.val_sample
        n_models = 1000

    X_train, X_test = [], []
    X_val, Y_val = [], []
    Y_train, Y_test = [], []
    num_per_dist = None
    for ratio in SUPPORTED_RATIOS:
        train_dir = os.path.join(
            BASE_MODELS_DIR, "adv/%s/%s/" % (args.filter, ratio))
        test_dir = os.path.join(
            BASE_MODELS_DIR, "victim/%s/%s/" % (args.filter, ratio))

        # Load models, convert to features
        if not args.eval_only:
            _, vecs_train_and_val = get_model_features(
                train_dir, max_read=n_models,
                focus=args.focus,
                shift_to_gpu=False)
            vecs_train_and_val = np.array(vecs_train_and_val, dtype='object')

            # Shuffle and divide train data into train and val
            shuffled = np.random.permutation(len(vecs_train_and_val))
            vecs_train = vecs_train_and_val[shuffled[:num_train]]
            vecs_val = vecs_train_and_val[shuffled[num_train:num_train+num_val]]

            Y_train += [float(ratio)] * len(vecs_train)
            Y_val += [float(ratio)] * len(vecs_val)

            X_train.append(vecs_train)
            X_val.append(vecs_val)

        dims, vecs_test = get_model_features(
            test_dir, max_read=n_models,
            focus=args.focus,
            shift_to_gpu=False)        
        vecs_test = np.array(vecs_test, dtype='object')
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
        Y_train = ch.from_numpy(np.array(Y_train)).float()
        Y_val = ch.from_numpy(np.array(Y_val)).float()

    X_test = np.concatenate(X_test)
    Y_test = ch.from_numpy(np.array(Y_test)).float()

    # Batch layer-wise inputs
    print("Batching data: hold on")
    if not args.eval_only:
        X_train = utils.prepare_batched_data(X_train)
        X_val = utils.prepare_batched_data(X_val)
    X_test = utils.prepare_batched_data(X_test)

    # Train meta-classifier model
    if args.focus == "conv":
        dim_channels, dim_kernels = dims
        metamodel = utils.PermInvConvModel(
            dim_channels, dim_kernels)
    elif args.focus == "fc":
        metamodel = utils.PermInvModel(dims)
    else:
        dims_conv, dims_fc = dims
        dim_channels, dim_kernels, middle_dim = dims_conv
        metamodel = utils.FullPermInvModel(
            dims_fc, middle_dim, dim_channels, dim_kernels,)

    if args.eval_only:
        # Load model
        metamodel.load_state_dict(ch.load(args.model_path))
        # Evaluate
        metamodel = metamodel.cuda()
        loss_fn = ch.nn.MSELoss(reduction='none')
        _, losses, preds = utils.test_meta(
                        metamodel, loss_fn, X_test, Y_test.cuda(),
                        args.batch_size, None,
                        binary=True, regression=True, gpu=True,
                        combined=True, element_wise=True,
                        get_preds=True)
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
        # Train PIM
        batch_size = 10 if args.testing else args.batch_size
        _, tloss = utils.train_meta_model(
            metamodel,
            (X_train, Y_train), (X_test, Y_test),
            epochs=args.epochs,
            binary=True, lr=args.lr,
            regression=True,
            batch_size=batch_size,
            val_data=(X_val, Y_val), combined=True,
            eval_every=10, gpu=True)
        print("Test loss %.4f" % (tloss))

        # Save meta-model
        ch.save(metamodel.state_dict(),
                "./log/meta/regression_models/%s_%.3f.pth" % (args.filter, tloss))
