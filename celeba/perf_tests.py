from model_utils import get_model, BASE_MODELS_DIR
from data_utils import CelebaWrapper, SUPPORTED_PROPERTIES
import torch.nn as nn
import numpy as np
import utils
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_models(folder_path, n_models=1000):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = get_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def get_accs(val_loader, models):
    accs = []

    criterion = nn.BCEWithLogitsLoss().cuda()
    for model in tqdm(models):
        model = model.cuda()

        vloss, vacc = utils.validate_epoch(
            val_loader, model, criterion, verbose=False)
        accs.append(vacc)
        # accs.append(vloss)

    return np.array(accs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--filter', help='alter ratio for this attribute',
                        required=True, choices=SUPPORTED_PROPERTIES)
    parser.add_argument('--task', default="Smiling",
                        choices=SUPPORTED_PROPERTIES,
                        help='task to focus on')
    parser.add_argument('--ratio_1', help="ratio for D_1", default="0.5")
    parser.add_argument('--ratio_2', help="ratio for D_2")
    parser.add_argument('--total_models', type=int, default=100)
    args = parser.parse_args()
    utils.flash_utils(args)

    # Get data with ratio
    print("Preparing data")
    ds_1 = CelebaWrapper(args.filter, float(
        args.ratio_1), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)
    ds_2 = CelebaWrapper(args.filter, float(
        args.ratio_2), "adv", cwise_samples=(int(1e6), int(1e6)),
        classify=args.task)

    # Get loaders
    loaders = [
        ds_1.get_loaders(args.batch_size, shuffle=False)[1],
        ds_2.get_loaders(args.batch_size, shuffle=False)[1]
    ]

    # Load victim models
    print("Loading models")
    models_victim_1 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.filter, args.ratio_1))
    models_victim_2 = get_models(os.path.join(
        BASE_MODELS_DIR, "victim", args.filter, args.ratio_2))

    # Load adv models
    total_models = args.total_models
    models_1 = get_models(os.path.join(
        BASE_MODELS_DIR, "adv", args.filter, args.ratio_1), total_models // 2)
    models_2 = get_models(os.path.join(
        BASE_MODELS_DIR, "adv", args.filter, args.ratio_2), total_models // 2)

    allaccs_1, allaccs_2 = [], []
    vic_accs, adv_accs = [], []
    for loader in loaders:
        accs_1 = get_accs(loader, models_1)
        accs_2 = get_accs(loader, models_2)

        # Look at [0, 100]
        accs_1 *= 100
        accs_2 *= 100

        print("Number of samples: %d" % total_models)

        tracc, threshold, rule = utils.find_threshold_acc(accs_1, accs_2)
        print("[Adversary] Threshold based accuracy: %.2f at threshold %.2f" %
              (100 * tracc, threshold))
        adv_accs.append(tracc)

        # Compute accuracies on this data for victim
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
    print("Loss-test accuracy: %.3f" % (100 * basic_baseline_acc))

    # Threshold baseline: look at model performance on test sets from both G_b
    # and pick the better one
    print("Threshold-test baseline accuracy: %.3f" %
          (100 * vic_accs[np.argmax(adv_accs)]))

    plt.plot(np.arange(len(accs_1)), np.sort(accs_1))
    plt.plot(np.arange(len(accs_2)), np.sort(accs_2))
    plt.savefig("./quick_see.png")
