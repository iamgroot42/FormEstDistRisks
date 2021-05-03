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


def collect_data_for_models(folder_paths,
                            all_x,
                            batch_size,
                            aif_dataset,
                            plot_metric=None,
                            sample=None):
    all_metrics = []
    for index, UPFOLDER in enumerate(folder_paths):
        for pf in UPFOLDER:
            modelList = os.listdir(pf)
            if sample:
                modelList = np.random.permutation(modelList)[:sample]
            for MODELPATHSUFFIX in tqdm(modelList):
                MODELPATH = os.path.join(pf, MODELPATHSUFFIX)
                # Only consider last three models per folder
                if not ("13_" in MODELPATHSUFFIX or "14_" in MODELPATHSUFFIX or "15_" in MODELPATHSUFFIX):
                    continue

                # Load model
                model = utils.FaceModel(512,
                                        train_feat=True,
                                        weight_init=None,
                                        hidden=[64, 16]).cuda()
                model = nn.DataParallel(model)
                model.load_state_dict(ch.load(MODELPATH), strict=False)
                model.eval()

                # Get predictions
                preds = implem_utils.get_predictions(
                    model, all_x, batch_size)
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
                                                       unprivileged_groups=[
                                                           {"Male": 0}],
                                                       privileged_groups=[
                                                           {"Male": 1}])

                all_metrics.append(metrics)

        if plot_metric is not None:
            ys = sorted([x[plot_metric] for x in all_metrics])
            xs = np.arange(len(ys))
            plt.plot(xs, ys,
                     label=str(index+1),
                     marker='o',
                     linestyle='dashed')
            plt.savefig("../visualize/metric_%s.png" % plot_metric)
            plt.clf()

    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=1024, help='batch size')
    parser.add_argument('--sample', type=int, default=None, help='number of models to sample per folder')
    parser.add_argument('--path1', type=str, default='celeba', help='path to models trained on D0')
    parser.add_argument('--path2', type=str, default='celeba', help='path to models trained on D1')
    parser.add_argument('--testpath1', type=str, default='celeba', help='path to unseen models trained on D0')
    parser.add_argument('--testpath2', type=str, default='celeba', help='path to unseen models trained on D1')
    args = parser.parse_args()
    utils.flash_utils(args)

    batch_size = args.bs
    constants = utils.Celeb()
    ds = constants.get_dataset()

    attrs = constants.attr_names
    inspect_these = ["Attractive", "Male", "Young"]
    common_prefix = "/p/adversarialml/as9rw/celeb_models/50_50"

    folder_paths = [
        [
            common_prefix + "/split_2/all/casia/1/",
            common_prefix + "/split_2/all/casia/2/",
            common_prefix + "/split_2/all/casia/3/",
            common_prefix + "/split_2/all/none/1/",
            common_prefix + "/split_2/all/none/2/",
            common_prefix + "/split_2/all/none/3/",
            common_prefix + "/split_2/all/vggface/1/",
            common_prefix + "/split_2/all/vggface/2/",
            common_prefix + "/split_2/all/vggface/3/",
        ],
        [
            common_prefix + "/split_2/male/casia/1/",
            common_prefix + "/split_2/male/casia/2/",
            common_prefix + "/split_2/male/casia/3/",
            common_prefix + "/split_2/male/none/1/",
            common_prefix + "/split_2/male/none/2/",
            common_prefix + "/split_2/male/none/3/",
            common_prefix + "/split_2/male/vggface/1/",
            common_prefix + "/split_2/male/vggface/2/",
            common_prefix + "/split_2/male/vggface/3/",
        ]
    ]

    blind_test_models = [
        [
            common_prefix + "/split_1/all/casia/1/",
            common_prefix + "/split_1/all/casia/2/",
            common_prefix + "/split_1/all/casia/3/",
            common_prefix + "/split_1/all/none/1/",
            common_prefix + "/split_1/all/none/2/",
            common_prefix + "/split_1/all/none/3/",
            common_prefix + "/split_1/all/vggface/1/",
            common_prefix + "/split_1/all/vggface/2/",
            common_prefix + "/split_1/all/vggface/3/",
        ],
        [
            common_prefix + "/split_1/male/casia/1/",
            common_prefix + "/split_1/male/casia/2/",
            common_prefix + "/split_1/male/casia/3/",
            common_prefix + "/split_1/male/none/1/",
            common_prefix + "/split_1/male/none/2/",
            common_prefix + "/split_1/male/none/3/",
            common_prefix + "/split_1/male/vggface/1/",
            common_prefix + "/split_1/male/vggface/2/",
            common_prefix + "/split_1/male/vggface/3/",
        ]
    ]

    # Use existing dataset instead
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        # "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/50_50/all/split_2/test",
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
                                     protected_attribute_names=["Male"])

    # Collect all metrics for given folders
    all_metrics = collect_data_for_models(folder_paths,
                                          all_x,
                                          batch_size,
                                          aif_dataset,
                                          sample=args.sample)

    # Train meta-classifier
    x_meta, y_meta = [], []
    x_meta = [list(mm.values()) for mm in all_metrics]
    y_meta = [0] * (len(x_meta) // 2) + [1] * (len(x_meta) // 2)

    clf = RandomForestClassifier(max_depth=3,
                                 n_estimators=25)  # random_state=42)
    clf.fit(x_meta, y_meta)
    print("Meta classifier score:", clf.score(x_meta, y_meta))

    all_metrics_test = collect_data_for_models(blind_test_models,
                                               all_x,
                                               batch_size,
                                               aif_dataset,
                                               sample=args.sample)
    x_meta_test = [list(mm.values()) for mm in all_metrics_test]
    y_meta_test = np.array(
        [0] * (len(x_meta_test) // 2) + [1] * (len(x_meta_test) // 2))
    scores = clf.predict_proba(x_meta_test)[:, 1]
    print(clf.score(x_meta_test, y_meta_test))

    zero_scores = sorted(scores[y_meta_test == 0])
    plt.plot(np.arange(len(zero_scores)), zero_scores, label="0", marker='o')
    one_scores = sorted(scores[y_meta_test == 1])
    plt.plot(np.arange(len(one_scores)), one_scores, label="1", marker='o')
    #  linestyle='dashed')

    plt.xlabel("Models (sorted by meta-classifier score)")
    plt.ylabel("Probability of model being trained on D1")
    plt.title("Fairness-based meta-classifier for CelebA")
    plt.savefig("../visualize/fairness_celeba_metaclf.png")
