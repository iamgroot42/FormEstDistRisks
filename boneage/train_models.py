from model_utils import BoneModel, BoneFullModel, save_model, check_if_exists, BASE_MODELS_DIR
from data_utils import BoneWrapper, get_df, get_features
import utils
import os


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=["victim", "adv"], required=True)
    parser.add_argument('--full_model', action="store_true",
                        help="Train E2E model")
    parser.add_argument('--verbose', action="store_true",
                        help="Train model-wise epoch stats")
    parser.add_argument('--ratio', type=float, default=0.5,
                        help='what ratio of the new sampled dataset should be true')
    parser.add_argument('--num', type=int, default=1000,
                        help='how many classifiers to train?')
    args = parser.parse_args()
    utils.flash_utils(args)

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df(args.split)

    # Load pre-processed features if model not end to end
    features = None
    if not args.full_model:
        features = get_features(args.split)

    # Subsample to make sure dataset sizes are comparable across ratios
    # All while maintaining class balance
    n_train, n_test = 700, 200
    if args.split == "victim":
        n_train, n_test = 1400, 400

    for i in range(args.num):
        # Check if model exists- skip if it does
        if check_if_exists(i+1, args.split, args.ratio, args.full_model):
            continue

        df_train_processed = utils.heuristic(
            df_train, filter, args.ratio,
            n_train, class_imbalance=1.0, n_tries=300)

        df_val_processed = utils.heuristic(
            df_val, filter, args.ratio,
            n_test, class_imbalance=1.0, n_tries=300)

        # Define model
        if args.full_model:
            model = BoneFullModel()
            batch_size = 40
            learning_rate = 2e-5
            epochs = 8
            weight_decay = 0.05
            augment = True
        else:
            model = BoneModel(1024)
            batch_size = 256*32
            learning_rate = 2e-3
            epochs = 20
            weight_decay = 0.02
            augment = False
        model = model.cuda()

        # Process datasets and get features
        ds = BoneWrapper(
            df_train_processed,
            df_val_processed,
            features=features,
            augment=augment)

        # Shuffle data, since will be used for training now
        train_loader, val_loader = ds.get_loaders(
            batch_size, shuffle=True)

        vloss, vacc = utils.train(model, (train_loader, val_loader),
                                  lr=learning_rate,
                                  epoch_num=epochs,
                                  weight_decay=weight_decay,
                                  verbose=args.verbose)

        # Make sure directory exists
        model_dir_path = os.path.join(BASE_MODELS_DIR, args.split, str(args.ratio))
        if not os.path.exists(model_dir_path):
            os.makedirs(model_dir_path)
        # Save model
        save_model(model, args.split, args.ratio,
                   "%d_%.3f.pth" %
                   (i+1, vacc),
                   full_model=args.full_model)

        print("[Status] %d/%d" % (i+1, args.num))
