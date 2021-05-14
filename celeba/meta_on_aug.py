from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch as ch
import os
import torch.nn as nn
import numpy as np
import utils
import implem_utils
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def convert_for_meta_model(paths, all_augdata,
                           target_prop, prop_index,
                           plotThem=None):
    drops, labels = [], []
    for i, mtype in enumerate(paths):
        for mdir in mtype:
            mlist = os.listdir(mdir)
            for path in tqdm(mlist):

                # Don't consider sub-optimal models
                model_acc = float(path.split("_")[1].rsplit(".", 1)[0])
                if model_acc < 0.9:
                    continue

                MODELPATH = os.path.join(mdir, path)
                model = utils.FaceModel(512,
                                        train_feat=True,
                                        weight_init=None).cuda()
                model = nn.DataParallel(model)
                model.load_state_dict(ch.load(MODELPATH), strict=False)
                model.eval()

                def model_fn(z):
                    return model(z)[:, 0]

                aug_diffs = []
                for augdata in all_augdata:
                    noprop, prop = implem_utils.get_robustness_shifts(model_fn,
                                                                      augdata,
                                                                      target_prop,
                                                                      prop_index)
                    # Take note of difference in performance drops
                    # Use relative drop ratio
                    drop_prop = (prop[0] - prop[1]) / prop[0]
                    drop_noprop = (noprop[0] - noprop[1]) / noprop[0]
                    aug_diffs.append(drop_noprop - drop_prop)

                drops.append(aug_diffs)
                labels.append(i)

                if plotThem is not None:
                    plt.plot(plotThem, aug_diffs,
                             label=str(i), color='C' + str(i))

    if plotThem is not None:
        plt.legend()
        plt.savefig("../visualize/all_models_translate.png")
        print("Saved patterns for all model checkpoints")
    return np.array(drops), np.array(labels)


if __name__ == "__main__":
    batch_size = 1000

    paths = [
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/none/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_vggface/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/all/64_16/augment_none/"
        ],
        [
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/none/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/attractive/64_16/none/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/augment_vggface/",
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_2/male/64_16/augment_none/"
        ]
    ]

    # Use existing dataset instead
    constants = utils.Celeb()
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)
    dataloader = DataLoader(td,
                            batch_size=batch_size,
                            shuffle=True)

    attrs = constants.attr_names
    target_prop = attrs.index("Smiling")
    # Look at examples that satisfy particular property
    inspect_these = ["Attractive", "Male", "Young"]

    # Generate and save augmented data for use
    # degrees = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    degrees = [20, 30, 40, 50, 60]
    jitter_vals = [0.5, 1, 2, 3, 4, 5, 6]
    translate_vals = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
    erase_vals = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    noprop_scores, prop_scores = [], []
    all_augdata = []
    for deg in degrees:
    # for jv in jitter_vals:
    # for tv in translate_vals:
    # for ev in erase_vals:
        augdata = implem_utils.collect_augmented_data(dataloader, deg=deg)
        # augdata = implem_utils.collect_augmented_data(dataloader,
                                                    #   erase_scale=(ev-0.01, ev))
                                                    #   translate=(0, tv))
                                                      # jitter=(0, 0, 0, jv))
        all_augdata.append(augdata)

    drops, labels = convert_for_meta_model(paths,
                                           all_augdata,
                                           target_prop,
                                           attrs.index(inspect_these[1]),
                                           plotThem=degrees)
    exit(0)

    # Use siamese approach
    # Tell if patterns from two models are identical or not
    # For unseen models at inference time, can use existing models
    # As template data
    X, Y = [], []
    for i in range(0, labels.shape[0]):
        for j in range(i+1, labels.shape[0]):
            Y.append(labels[i] == labels[j])
            X.append(np.abs(drops[i] - drops[j]))

    # Split into training and validation data for meta-classifier
    x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.3)
    # x_tr, x_te, y_tr, y_te = train_test_split(drops, labels, test_size=0.3)
    clf = RandomForestClassifier(max_depth=4)
    # clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=500)
    clf.fit(x_tr, y_tr)
    print("Accuracy on training data: %.2f" % clf.score(x_tr, y_tr))
    print("Accuracy on validation data: %.2f" % clf.score(x_te, y_te))

    # Evaluate model on unseen models
    unseen_paths = [
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/augment_vggface/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/all/vggface/"
        ],
        [
            # "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/male/augment_vggface/",
            "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/split_1/male/vggface/"
        ]
    ]

    drops, labels = convert_for_meta_model(unseen_paths,
                                           all_augdata,
                                           target_prop,
                                           attrs.index(inspect_these[1]))
    X, Y = [], []
    for i in range(0, labels.shape[0]):
        for j in range(i+1, labels.shape[0]):
            Y.append(labels[i] == labels[j])
            X.append(np.abs(drops[i] - drops[j]))

    print("Accuracy on unseen models: %.2f" % clf.score(X, Y))
    # print("Accuracy on unseen models: %.2f" % clf.score(drops, labels))
