from model_utils import load_model, get_model_folder_path, get_models
from data_utils import BoneWrapper, get_df, get_features, SUPPORTED_RATIOS
import torch.nn as nn
import numpy as np
import utils
from tqdm import tqdm
import os


def get_accs(val_loader, models):
    accs = []

    criterion = nn.BCEWithLogitsLoss().cuda()
    for model in tqdm(models):
        # Shift model to GPU
        model = model.cuda()

        vloss, vacc = utils.validate_epoch(
            val_loader, model, criterion, verbose=False)
        accs.append(vacc)

        # Bring back to CPU (save GPU memory)
        model = model.cpu()
    return np.array(accs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256*32)
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2", required=True,
                        choices=SUPPORTED_RATIOS)
    parser.add_argument('--victim_full', action="store_true",
                        help="Use full BoneAge Densenet models for victim models")
    parser.add_argument('--testing', action="store_true",
                        help="Testing mode")
    parser.add_argument('--total_models', type=int, default=100,
                        help="Total number of models adversary uses for attack")
    args = parser.parse_args()
    utils.flash_utils(args)

    if args.testing:
        total_models = 3
        n_test_models = 3
    else:
        total_models = args.total_models
        n_test_models = 1000

    def filter(x): return x["gender"] == 1

    # Ready data
    df_train, df_val = get_df("adv")
    features = get_features("adv")

    # Get data with ratio
    df_1 = utils.heuristic(
        df_val, filter, float(args.ratio_1),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    df_2 = utils.heuristic(
        df_val, filter, float(args.ratio_2),
        cwise_sample=10000,
        class_imbalance=1.0, n_tries=300)

    # Prepare data loaders
    ds_1 = BoneWrapper(
        df_1, df_1, features=features)
    ds_2 = BoneWrapper(
        df_2, df_2, features=features)
    loaders = [
        ds_1.get_loaders(args.batch_size, shuffle=False)[1],
        ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]
    # If victim model is full, get image-level dataset for same df
    if args.victim_full:
        ds_1_full = BoneWrapper(df_1, df_1)
        ds_2_full = BoneWrapper(df_2, df_2)
        loaders_full = [
            ds_1_full.get_loaders(args.batch_size, shuffle=False)[1],
            ds_2_full.get_loaders(args.batch_size, shuffle=False)[1]
        ]

    # Load victim models
    models_victim_1 = get_models(get_model_folder_path(
        "victim", args.ratio_1, full_model=args.victim_full),
        n_models=n_test_models,
        full_model=args.victim_full)
    models_victim_2 = get_models(get_model_folder_path(
        "victim", args.ratio_2, full_model=args.victim_full),
        n_models=n_test_models,
        full_model=args.victim_full)

    # Load adv models
    models_1 = get_models(get_model_folder_path(
        "adv", args.ratio_1), total_models // 2)
    models_2 = get_models(get_model_folder_path(
        "adv", args.ratio_2), total_models // 2)

    allaccs_1, allaccs_2 = [], []
    vic_accs, adv_accs = [], []
    for i, loader in enumerate(loaders):
        accs_1 = get_accs(loader, models_1)
        accs_2 = get_accs(loader, models_2)

        # # Look at [0, 100]
        accs_1 *= 100
        accs_2 *= 100

        tracc, threshold, rule = utils.find_threshold_acc(accs_1, accs_2)
        print("[Adversary] Threshold based accuracy: %.2f at threshold %.2f" %
              (100 * tracc, threshold))
        adv_accs.append(tracc)

        # Compute accuracies on this data for victim
        if args.victim_full:
            accs_victim_1 = get_accs(loaders_full[i], models_victim_1)
            accs_victim_2 = get_accs(loaders_full[i], models_victim_2)
        else:
            accs_victim_1 = get_accs(loader, models_victim_1)
            accs_victim_2 = get_accs(loader, models_victim_2)

        # Look at [0, 100]
        accs_victim_1 *= 100
        accs_victim_2 *= 100

        # Threshold based on adv models
        combined = np.concatenate((accs_victim_1, accs_victim_2))
        classes = np.concatenate(
            (np.zeros_like(accs_victim_1), np.ones_like(accs_victim_2)))
        specific_acc = utils.get_threshold_acc(
            combined, classes, threshold, rule)
        print("[Victim] Accuracy at specified threshold: %.2f" %
              (100 * specific_acc))
        vic_accs.append(specific_acc)

        # Collect all accuracies for basic baseline
        allaccs_1.append(accs_victim_1)
        allaccs_2.append(accs_victim_2)

    adv_accs = np.array(adv_accs)
    vic_accs = np.array(vic_accs)

    # Basic baseline: look at model performance on test sets from both G_b
    # Predict b for whichever b it is higher
    allaccs_1 = np.array(allaccs_1).T
    allaccs_2 = np.array(allaccs_2).T

    preds_1 = (allaccs_1[:, 0] > allaccs_1[:, 1])
    preds_2 = (allaccs_2[:, 0] < allaccs_2[:, 1])
    basic_baseline_acc = (np.mean(preds_1) + np.mean(preds_2)) / 2

    print("[Results] %s v/s %s" % (args.ratio_1, args.ratio_2))
    print("Loss-Test accuracy: %.3f" % (100 * basic_baseline_acc))

    # Threshold baseline: look at model performance on test sets from both G_b
    # and pick the better one
    print("Threshold-Test accuracy: %.3f" %
          (100 * vic_accs[np.argmax(adv_accs)]))
