import data_utils
import pandas as pd
import torch as ch
import numpy as np
import utils
import os


def get_same_ratio(df, filter, ratio):
    trials = 0
    while trials < 1e3:
        df_processed = utils.filter(df, filter, ratio, verbose=False)
        df_processed.reset_index(inplace=True, drop=True)
        # Only accept a split that does not vary
        # label ratio too much
        if (np.abs(df_processed["label"].mean() - df["label"].mean()) <= 0.05):
            return df_processed
        trials += 1

    raise ValueError("Could not split while preserving label ratios!")


if __name__ == "__main__":
    import sys

    target_prop = sys.argv[1]
    target_ratio = float(sys.argv[2])
    num_models = int(sys.argv[3])
    split = int(sys.argv[4])
    savepath_prefix = sys.argv[5]

    batch_size = 250

    def filter(x): return x[target_prop] == 1

    # Get DF
    df_train = pd.read_csv("./data/split_%d/train.csv" % (split))
    df_val = pd.read_csv("./data/split_%d/val.csv" % (split))

    # Load features
    features = {}
    features["train"] = ch.load("./data/split_%d/features_train.pt" % (split))
    features["val"] = ch.load("./data/split_%d/features_val.pt" % (split))

    for i in range(num_models):

        df_train_processed = get_same_ratio(df_train, filter, target_ratio)
        df_val_processed = get_same_ratio(df_val, filter, target_ratio)

        # Define augmentation
        model = data_utils.HamModel(1024)
        model = model.cuda()

        # Process datasets and get features
        ds = data_utils.HamWrapper(
            df_train, df_val,
            features)

        train_loader, val_loader = ds.get_loaders(batch_size)

        # Train model
        vloss, vacc = data_utils.train(model, (train_loader, val_loader),
                                       lr=5e-4, epoch_num=20,
                                       weight_decay=0.02, verbose=False)

        # Save model
        ch.save(model.state_dict(), os.path.join(
            savepath_prefix, "%d_%.3f.pth" % (i+1, vacc)))

        print("[Status] %d/%d" % (i+1, num_models))
