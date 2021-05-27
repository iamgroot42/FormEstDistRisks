from model_utils import BoneModel, save_model
from data_utils import BoneWrapper, get_df, get_features
import utils


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=["victim", "adv"])
    parser.add_argument('--batch_size', type=int, default=256*32)
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='what ratio of the new sampled dataset should be true')
    parser.add_argument('--num', type=int, default=1000,
                        help='how many classifiers to train?')
    args = parser.parse_args()
    utils.flash_utils(args)

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df(args.split)
    features = get_features(args.split)

    # Subsample to make sure dataset sizes are comparable across ratios
    # All while maintaining class balance
    n_train, n_test = 700, 200
    if args.split == "victim":
        n_train, n_test = 1400, 400

    for i in range(args.num):

        df_train_processed = utils.heuristic(
            df_train, filter, args.ratio,
            n_train, class_imbalance=1.0, n_tries=300)

        df_val_processed = utils.heuristic(
            df_val, filter, args.ratio,
            n_test, class_imbalance=1.0, n_tries=300)

        # Define model
        model = BoneModel(1024)
        model = model.cuda()

        # Process datasets and get features
        ds = BoneWrapper(
            df_train_processed,
            df_val_processed,
            features=features)

        # Shuffle data, since will be used for training now
        train_loader, val_loader = ds.get_loaders(
            args.batch_size, shuffle=True)

        # Train model
        vloss, vacc = utils.train(model, (train_loader, val_loader),
                                  lr=2e-3, epoch_num=20,
                                  weight_decay=0.02,
                                  verbose=False)

        # Save model
        save_model(model, "%d_%.3f.pth" % (i+1, vacc))

        print("[Status] %d/%d" % (i+1, args.num))
