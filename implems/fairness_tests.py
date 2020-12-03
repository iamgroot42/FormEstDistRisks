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
    # data = {"features": all_x}
    data = {}
    for i, colname in enumerate(attrs):
        data[colname] = all_y[:, i]
    df = pd.DataFrame(data=data)
    aif_dataset = BinaryLabelDataset(df=df,
                                     label_names=["Smiling"],
                                     protected_attribute_names=["Attractive"])

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
                # print("For model %s" % MODELPATH)
                # for k, v in metrics.items():
                #     print(k, v)
                # print("\n\n")

                all_metrics.append(metrics)

        picked_focus = "Disparate impact"
        # picked_focus = "Equal opportunity difference"
        # picked_focus = "Average odds difference"
        # picked_focus = "Theil index"
        # picked_focus = "Statistical parity difference"
        # ys = sorted([x[picked_focus] for x in all_metrics])
        # xs = np.arange(len(ys))
        # plt.plot(xs, ys,
        #          label=str(index+1),
        #          marker='o',
        #          linestyle='dashed')

    # Train meta-classifier (because why not?)
    x_meta, y_meta = [], []
    for mm in all_metrics:
        x_meta.append(list(mm.values()))
    y_meta = [0] * (len(x_meta) // 2) + [1] * (len(x_meta) // 2)

    clf = RandomForestClassifier(max_depth=4,)  # random_state=42)
    clf.fit(x_meta, y_meta)
    print("Meta classifier score:", clf.score(x_meta, y_meta))

    # plt.xlabel("Models (sorted by metric)")
    # plt.ylabel(picked_focus)
    # plt.savefig("../visualize/fairness_celeba_%s.png" % ("_".join(picked_focus.split(" "))))

    for index, UPFOLDER in enumerate(blind_test_models):
        scores = []
        for pf in UPFOLDER:
            for MODELPATHSUFFIX in os.listdir(pf):
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

                meta_test = [list(metrics.values())]
                scores.append(clf.predict_proba(meta_test)[0, 1])

        ys = sorted(scores)
        xs = np.arange(len(ys))
        plt.plot(xs, ys,
                 label=str(index+1),
                 marker='o',
                 linestyle='dashed')

    plt.xlabel("Models (sorted by meta-classifier score)")
    plt.ylabel("Probability of model being trained on D1")
    plt.savefig("../visualize/fairness_celeba_metaclf.png")
