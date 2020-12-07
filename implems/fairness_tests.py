import utils
import implem_utils

from tqdm import tqdm
import os
import numpy as np
import torch as ch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
from aif360.datasets import BinaryLabelDataset

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def collect_data_for_folders(folder_paths, plot_metric=None):
    for index, UPFOLDER in enumerate(folder_paths):
        all_metrics = []
        for pf in UPFOLDER:
            for MODELPATHSUFFIX in tqdm(os.listdir(pf)):
                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                # Load model
                model = utils.FaceModel(512,
                                        train_feat=True,
                                        weight_init=None,
                                        hidden=[64, 16]).cuda()
                model = nn.DataParallel(model)
                model.load_state_dict(ch.load(MODELPATH), strict=False)
                model.eval()

                # Get predictions
                preds = implem_utils.get_predictions(model, all_x, batch_size)
                preds = ch.sigmoid(preds).numpy()

                # Get model's predictions
                clasf_dataset = aif_dataset.copy(deepcopy=True)
                clasf_dataset.scores = preds

                # Reset label values as well
                thresh = 0.5
                fav_inds = clasf_dataset.scores > thresh
                clasf_dataset.labels[fav_inds] = clasf_dataset.favorable_label
                clasf_dataset.labels[~fav_inds] = clasf_dataset.unfavorable_label

                metrics = implem_utils.compute_metrics(aif_dataset,
                                                       clasf_dataset,
                                                       unprivileged_groups=[{"Attractive": 0}],
                                                       privileged_groups=[{"Attractive": 1}])

                all_metrics.append(metrics)

        if plot_metric is not None:
            ys = sorted([x[plot_metric] for x in all_metrics])
            xs = np.arange(len(ys))
            plt.plot(xs, ys,
                     label=str(index+1),
                     marker='o',
                     linestyle='dashed')

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    args = parser.parse_args()
    utils.flash_utils(args)

    batch_size = args.bs

    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]

    folder_paths = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/augment_none/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/",
        ]
    ]

    blind_test_models = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/"
        ]
    ]

    # Use existing dataset instead
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    # Read all data and get ready to store as df
    # To be used by aif360
    cropped_dataloader = DataLoader(td,
                                    batch_size=batch_size,
                                    shuffle=False)

    all_x, all_y = utils.load_all_loader_data(cropped_dataloader)
    data = {}
    for i, colname in enumerate(attrs):
        data[colname] = all_y[:, i]
    df = pd.DataFrame(data=data)
    aif_dataset = BinaryLabelDataset(df=df,
                                     label_names=["Smiling"],
                                     protected_attribute_names=["Attractive"])

    # Collect all metrics for given folders
    picked_focus = [
        "False discovery rate difference",
        "False omission rate difference",
        "Disparate impact",
        "Equal opportunity difference",
        "Average odds difference",
        "Theil index",
        "Statistical parity difference"
    ]
    all_metrics = collect_data_for_folders(folder_paths)
    # plt.xlabel("Models (sorted by metric)")
    # plt.ylabel(picked_focus)
    # plt.savefig("../visualize/fairness_celeba_%s.png" % ("_".join(picked_focus.split(" "))))
    # exit(0)

    # Train meta-classifier
    x_meta, y_meta = [], []
    x_meta = [list(mm.values()) for mm in all_metrics]
    y_meta = [0] * (len(x_meta) // 2) + [1] * (len(x_meta) // 2)

    clf = RandomForestClassifier(max_depth=3,
                                 n_estimators=25)  # random_state=42)
    clf.fit(x_meta, y_meta)
    print("Meta classifier score:", clf.score(x_meta, y_meta))

    # all_metrics_test = collect_data_for_folders(folder_paths)
    all_metrics_test = collect_data_for_folders(blind_test_models)
    x_meta_test = [list(mm.values()) for mm in all_metrics_test]
    y_meta_test = np.array([0] * (len(x_meta_test) // 2) + [1] * (len(x_meta_test) // 2))
    scores = clf.predict_proba(x_meta_test)[:, 1]
    print(clf.score(x_meta_test, y_meta_test))

    zero_scores = sorted(scores[y_meta_test == 0])
    plt.plot(np.arange(len(zero_scores)), zero_scores, label="0", marker='o')
    one_scores = sorted(scores[y_meta_test == 1])
    plt.plot(np.arange(len(one_scores)), one_scores, label="1", marker='o')
    #  linestyle='dashed')

    plt.xlabel("Models (sorted by meta-classifier score)")
    plt.ylabel("Probability of model being trained on D1")
    plt.savefig("../visualize/fairness_celeba_metaclf.png")
