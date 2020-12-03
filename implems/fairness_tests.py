import utils
import implem_utils

from tqdm import tqdm
import os
import numpy as np
import torch as ch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import train_test_split
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
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/augment_none/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/",
        ]
    ]

    blind_test_models = [
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/10_0.928498243559719.pth",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/10_0.9093969555035128.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/20_0.9006555723651034.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/15_0.9073793914943687.pth",

        ],
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/augment_vggface/10_0.9240681998413958.pth",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/attractive/vggface/10_0.8992862807295797.pth",

            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/20_0.8947974217311234.pth",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/15_0.9120626151012892.pth",
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
                if not("3_" in MODELPATHSUFFIX or "8_" in MODELPATHSUFFIX or "13_" in MODELPATHSUFFIX): continue
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
        ys = sorted([x[picked_focus] for x in all_metrics])
        xs = np.arange(len(ys))
        plt.plot(xs, ys, label=str(index+1))

    plt.savefig("../visualize/fairness_celeba.png")
