from data_utils import BotNetWrapper

import torch as ch
import argparse
from model_utils import get_model, BASE_MODELS_DIR, epoch
import os
from utils import find_threshold_acc, get_threshold_acc
from tqdm import tqdm
import numpy as np


def load_models(model_dir, args, max_read=None):
    iterator = os.listdir(model_dir)
    if max_read is not None:
        iterator = np.random.permutation(iterator)[:max_read]

    models = []
    for mpath in tqdm(iterator):
        # Load model
        model = get_model(args)
        model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        model.eval()
        models.append(model)
    return models


@ch.no_grad()
def get_model_scores(models, ds, args):
    _, test_loader = ds.get_loaders(args.batch_size, shuffle=False)

    lossees, f1s = [], []
    for model in tqdm(models):
        loss, f1 = epoch(model, test_loader, args.gpu,
                         optimizer=None, verbose=False)
        lossees.append(loss)
        f1s.append(f1)

    return np.array(lossees), np.array(f1s)


def main():
    parser = argparse.ArgumentParser(description='Botnets-dataset (GCN)')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--train_sample', type=int, default=800)
    parser.add_argument('--val_sample', type=int, default=0)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--n_feat', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--n_use', type=int, default=100,
                        help="Number of models adversary uses to estimate trends")
    args = parser.parse_args()
    print(args)

    # Get datasets ready
    ds_1 = BotNetWrapper(split="adv", prop_val=0)
    ds_2 = BotNetWrapper(split="adv", prop_val=1)

    # Directories where saved models are stored
    dir_victim_1 = os.path.join(BASE_MODELS_DIR, "victim", "0")
    dir_victim_2 = os.path.join(BASE_MODELS_DIR, "victim", "1")
    dir_1 = os.path.join(BASE_MODELS_DIR, "adv", "0")
    dir_2 = os.path.join(BASE_MODELS_DIR, "adv", "1")

    # Load victim models
    models_victim_1 = load_models(dir_victim_1, args)
    models_victim_2 = load_models(dir_victim_2, args)

    # Load adv models
    models_1 = load_models(dir_1, args, args.n_use // 2)
    models_2 = load_models(dir_2, args, args.n_use // 2)

    adv_f1s, f_accs = [], []
    loders = [ds_1, ds_2]
    allaccs_1, allaccs_2 = [], []
    for j, loader in enumerate(loders):

        # Get model predictions
        _, f1_1 = get_model_scores(models_1, loader, args)
        _, f1_2 = get_model_scores(models_2, loader, args)

        tr_f1, threshold, rule = find_threshold_acc(
            f1_1, f1_2, granularity=0.001)
        print("[Adversary] Threshold based accuracy: %.2f at threshold %.3f" %
              (100 * tr_f1, threshold))
        adv_f1s.append(tr_f1)

        # Compute accuracies on this data for victim
        _, f1_victim_1 = get_model_scores(models_victim_1, loader, args)
        _, f1_victim_2 = get_model_scores(models_victim_2, loader, args)

        # Threshold based on adv models
        combined = np.concatenate((f1_victim_1, f1_victim_2))
        classes = np.concatenate(
            (np.zeros_like(f1_victim_1), np.ones_like(f1_victim_2)))
        specific_acc = get_threshold_acc(combined, classes, threshold, rule)
        print("[Victim] Accuracy at specified threshold: %.3f" %
              (100 * specific_acc))
        f_accs.append(100 * specific_acc)

        # Collect all accuracies for basic baseline
        allaccs_1.append(f1_victim_1)
        allaccs_2.append(f1_victim_2)

    # Basic baseline: look at model performance on test sets from both G_b
    # Predict b for whichever b it is higher
    allaccs_1 = np.array(allaccs_1)
    allaccs_2 = np.array(allaccs_2)

    preds_1 = (allaccs_1[0, :] > allaccs_1[1, :])
    preds_2 = (allaccs_2[0, :] <= allaccs_2[1, :])

    basic_baseline_acc = (np.mean(preds_1) + np.mean(preds_2)) / 2
    print("[Victim] Threshold-test accuracy: %.2f" % f_accs[np.argmin(adv_f1s)])
    print("[Victim] Loss-test accuracy: %.3f" % (100 * basic_baseline_acc))


if __name__ == "__main__":
    main()
