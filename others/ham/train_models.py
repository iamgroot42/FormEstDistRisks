import data_utils
import pandas as pd
import torch as ch
import utils
import os


if __name__ == "__main__":
    import sys

    target_prop = sys.argv[1]
    target_ratio = float(sys.argv[2])
    num_models = int(sys.argv[3])
    split = int(sys.argv[4])
    savepath_prefix = sys.argv[5]

    batch_size = 250 * 8 * 4
    # Change the last '4' to the number of GPUs that are almost free

    def filter(x): return x[target_prop] == 1

    basedir = "/p/adversarialml/as9rw/datasets/ham10000/splits/"

    # Make sure directory exists
    utils.ensure_dir_exists(savepath_prefix)

    # Get DF
    df_train = pd.read_csv(os.path.join(
        basedir, "split_%d/train.csv" % (split)))
    df_val = pd.read_csv(os.path.join(
        basedir, "split_%d/val.csv" % (split)))

    # Load features
    features = {}
    features["train"] = ch.load(os.path.join(
        basedir, "split_%d/features_train.pt" % (split)))
    features["val"] = ch.load(os.path.join(
        basedir, "split_%d/features_val.pt" % (split)))

    for i in range(num_models):

        df_train_processed = utils.heuristic(
            df_train, filter, target_ratio,
            960, class_imbalance=2.0, n_tries=50)

        df_val_processed = utils.heuristic(
            df_val, filter, target_ratio,
            250, class_imbalance=2.0, n_tries=50)

        print("%d train samples, %d val samples" %
              (len(df_train_processed), len(df_val_processed)))

        # Define augmentation
        model = data_utils.HamModel(1024)
        model = model.cuda()

        # Process datasets and get features
        ds = data_utils.HamWrapper(
            df_train_processed,
            df_val_processed,
            features)

        train_loader, val_loader = ds.get_loaders(batch_size, shuffle=False)

        # Train model
        vloss, vacc = utils.train(model, (train_loader, val_loader),
                                  lr=2e-3, epoch_num=20,
                                  weight_decay=0.02, verbose=False)
        print("Validation accuracy: %.3f" % vacc)

        # Save model
        ch.save(model.state_dict(), os.path.join(
            savepath_prefix, "%d_%.3f.pth" % (i+1, vacc)))

        print("[Status] %d/%d" % (i+1, num_models))
