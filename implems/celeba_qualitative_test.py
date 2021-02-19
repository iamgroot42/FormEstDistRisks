import numpy as np
import utils
import torch as ch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib as mpl
# mpl.rcParams['figure.dpi'] = 200


def get_stats(mainmodel, dataloader, target_prop, return_preds=False):
    stats = []
    all_stats = []
    all_preds = [] if return_preds else None
    for (x, y) in dataloader:

        y_ = y[:, target_prop].cuda()

        preds = mainmodel(x.cuda()).detach()[:, 0]
        incorrect = ((preds >= 0) != y_)
        stats.append(y[incorrect].cpu().numpy())
        if return_preds:
            all_preds.append(preds.cpu().numpy())
            all_stats.append(y.cpu().numpy())

    stats = np.concatenate(stats)
    all_preds = np.concatenate(all_preds)
    all_stats = np.concatenate(all_stats)

    return stats, all_preds, all_stats


if __name__ == "__main__":

    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    folder_paths = [
        "/p/adversarialml/as9rw/celeb_models/50_50/split_2/all/none/",    
        "/p/adversarialml/as9rw/celeb_models/50_50/split_2/male/none/"
    ]

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/"
        "50_50/all/split_2/test",
        transform=transform)

    target_prop = attrs.index("Smiling")
    all_cfms = []
    successes, failures = [], []

    for UPFOLDER in folder_paths:
        model_preds = []
        model_stats = []

        for FOLDER in tqdm(os.listdir(UPFOLDER)):
            print(FOLDER)
            wanted_model = [x for x in os.listdir(
                os.path.join(UPFOLDER, FOLDER)) if x.startswith("15_")][0]
            MODELPATH = os.path.join(UPFOLDER, FOLDER, wanted_model)

            # Load model
            model = utils.FaceModel(512,
                                    train_feat=True,
                                    weight_init=None,
                                    hidden=[64, 16]).cuda()
            model = nn.DataParallel(model)
            model.load_state_dict(ch.load(MODELPATH), strict=False)
            model.eval()

            cropped_dataloader = DataLoader(td, batch_size=256, shuffle=False)

            stats, preds, all_stats = get_stats(
                model, cropped_dataloader, target_prop, return_preds=True)
            model_stats.append(all_stats)
            model_preds.append(preds)

        model_preds = np.array(model_preds)
        model_stats = np.array(model_stats)

        yeslabel = np.nonzero(all_stats[:, target_prop] == 1)[0]
        nolabel = np.nonzero(all_stats[:, target_prop] == 0)[0]

        # Look at failure/success cases to identify any potential trends
        success, fail = [], []
        for mp in model_preds:
            success.append(np.nonzero((mp >= 0) == all_stats[:, target_prop])[0])
            fail.append(np.nonzero((mp >= 0) != all_stats[:, target_prop])[0])
        successes.append(success)
        failures.append(fail)

        # Print accuracies
        label_attr = attrs.index("Male")
        prop_ones = np.nonzero(all_stats[:, label_attr] == 1)[0]
        noprop_ones = np.nonzero(all_stats[:, label_attr] == 0)[0]
        accuracies_prop = [np.mean(((mp >= 0) == all_stats[:, target_prop])[
                                   prop_ones]) for mp in model_preds]
        accuracies_noprop = [np.mean(((mp >= 0) == all_stats[:, target_prop])[
                                     noprop_ones]) for mp in model_preds]
        print("Accuracy on data (P=1):", accuracies_prop)
        print("Accuracy on data (P=0):", accuracies_noprop)

        # Look at loss
        lossfn = nn.BCEWithLogitsLoss(reduction='none')
        cfms = []

        # Pick relevant samples
        label_prop = np.nonzero(all_stats[yeslabel, label_attr] == 1)[0]
        label_noprop = np.nonzero(all_stats[yeslabel, label_attr] == 0)[0]
        nolabel_prop = np.nonzero(all_stats[nolabel, label_attr] == 1)[0]
        nolabel_noprop = np.nonzero(all_stats[nolabel, label_attr] == 0)[0]

        for i in range(len(model_preds)):

            label_prop_losses = lossfn(ch.from_numpy(model_preds[i][yeslabel][label_prop]), ch.from_numpy(
                1. * all_stats[yeslabel, target_prop][label_prop]))
            label_noprop_losses = lossfn(ch.from_numpy(model_preds[i][yeslabel][label_noprop]), ch.from_numpy(
                1. * all_stats[yeslabel, target_prop][label_noprop]))

            nolabel_prop_losses = lossfn(ch.from_numpy(model_preds[i][nolabel][nolabel_prop]), ch.from_numpy(
                0. * all_stats[nolabel, target_prop][nolabel_prop]))
            nolabel_noprop_losses = lossfn(ch.from_numpy(model_preds[i][nolabel][nolabel_noprop]), ch.from_numpy(
                0. * all_stats[nolabel, target_prop][nolabel_noprop]))

            cfms.append([[ch.mean(label_prop_losses), ch.mean(label_noprop_losses)], [
                        ch.mean(nolabel_prop_losses), ch.mean(nolabel_noprop_losses)]])

        all_cfms.append(cfms)

    all_cfms = np.array(all_cfms)

    # Get property-wise losses per model
    prop_losses = (all_cfms[:, :, 0, 0] * len(label_prop) + all_cfms[:, :, 1, 0]
                   * len(label_noprop)) / (len(label_prop) + len(label_noprop))
    noprop_losses = (all_cfms[:, :, 0, 1] * len(nolabel_prop) + all_cfms[:, :, 1, 1]
                     * len(nolabel_noprop)) / (len(nolabel_prop) + len(nolabel_noprop))

    print("Loss on data (P=1):", prop_losses)
    print("Loss on data (P=0):", noprop_losses)
    exit(0)

    # Compute overlap of success, failure cases across models
    print("Success/Failure case analysis")
    for i in range(len(successes)):
        for k in range(len(successes[0])):
            for j in range(len(successes)):
                for l in range(len(successes[0])):
                    # Skip same model type, same model
                    if i == j and k == l:
                        continue

                    s1, s2 = set(successes[i][k]), set(successes[j][l])
                    f1, f2 = set(successes[i][k]), set(successes[j][l])
                    success_overlap = len(
                        s1.intersection(s2)) / len(s1.union(s2))
                    failure_overlap = len(
                        f1.intersection(f2)) / len(f1.union(f2))

                    # print("(%d,%d) (%d,%d) : Success Overlap: %f" % (i, k, j, l, success_overlap))
                    print("(%d,%d) (%d,%d) : Failure Overlap: %f" %
                          (i, k, j, l, failure_overlap))
        print()
