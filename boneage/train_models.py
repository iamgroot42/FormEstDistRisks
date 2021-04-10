import data_utils
import pandas as pd
import torch as ch
import utils
import os


if __name__ == "__main__":
    import sys

    target_ratio = float(sys.argv[1])
    num_models = int(sys.argv[2])
    split = int(sys.argv[3])
    savepath_prefix = sys.argv[4]

    batch_size = 256 * 32

    def filter(x): return x["gender"] == 1

    # Make sure directory exists
    utils.ensure_dir_exists(savepath_prefix)

    # Get DF
    df_train = pd.read_csv("./data/split_%d/train.csv" % (split))
    df_val = pd.read_csv("./data/split_%d/val.csv" % (split))

    # Load features
    features = {}
    features["train"] = ch.load("./data/split_%d/features_train.pt" % (split))
    features["val"] = ch.load("./data/split_%d/features_val.pt" % (split))

    # Subsample to make sure dataset sizes are comparable across ratios
    # All while maintaining class balance
    n_train, n_test = 700, 200
    if split == 1:
        n_train, n_test = 1400, 400

    for i in range(num_models):

        df_train_processed = utils.heuristic(
            df_train, filter, target_ratio,
            n_train, class_imbalance=1.0, n_tries=300)

        df_val_processed = utils.heuristic(
            df_val, filter, target_ratio,
            n_test, class_imbalance=1.0, n_tries=300)

        # Define model
        model = data_utils.BoneModel(1024)
        model = model.cuda()

        # Process datasets and get features
        ds = data_utils.BoneWrapper(
            df_train, df_val,
            features=features)

        train_loader, val_loader = ds.get_loaders(batch_size, shuffle=False)

        # Train model
        vloss, vacc = utils.train(model, (train_loader, val_loader),
                                  lr=2e-3, epoch_num=20,
                                  weight_decay=0.02,
                                  verbose=False)

        # Save model
        ch.save(model.state_dict(), os.path.join(
            savepath_prefix, "%d_%.3f.pth" % (i+1, vacc)))

        print("[Status] %d/%d" % (i+1, num_models))
