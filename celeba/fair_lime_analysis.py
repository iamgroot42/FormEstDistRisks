import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch as ch
from torchvision import transforms

from lime import lime_image

import model_utils
import data_utils
import utils
import os
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt 
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def raw_255_image(z):
    z_ = z.numpy().transpose(1, 2, 0)
    z_ = (z_ + 1) / 2
    return (z_ * 255).astype(np.uint8)


def batch_predict(model, images):
    batch = ch.stack(tuple(transform(i) for i in images), dim=0)
    logits = model(batch.cuda()).detach()
    probs = ch.sigmoid(logits)
    probs = ch.stack([1 - probs[:, 0], probs[:, 0]], 1)
    return probs.cpu().numpy()


def get_scores(model, verbose=True):
    explainer = lime_image.LimeImageExplainer()

    scores = []
    labels = []

    def model_batch_predict(x):
        return batch_predict(model, x)

    iterator = enumerate(cropped_dataloader)
    if verbose:
        iterator = tqdm(iterator, total=len(cropped_dataloader))
    for i, (x, y) in iterator:
        x_raw = [raw_255_image(x_) for x_ in x]
        labels.append(y.numpy())

        for img_t in x_raw:
            explanation = explainer.explain_instance(img_t,
                                                     model_batch_predict,
                                                     top_labels=1,
                                                     hide_color=0,
                                                     num_samples=200,
                                                     progress_bar=False)

            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                        positive_only=True,
                                                        num_features=5,
                                                        hide_rest=False)

            temp_pos = temp * (np.expand_dims(mask, 2))
            temp_neg = temp * (1 - np.expand_dims(mask, 2))
            test_pred = model_batch_predict([temp.astype(np.uint8),
                                            temp_pos.astype(np.uint8),
                                            temp_neg.astype(np.uint8)])
            scores.append(test_pred[:, 1])

        if i == 20:
            break

    return np.stack(scores, 0), np.concatenate(labels, 0)


def get_data_for_meta(model_paths,
                      attrs,
                      how_many=None,
                      target_prop="Male"):
    X, Y = [], []
    for i, model_cluster in enumerate(model_paths):
        for model_dir in model_cluster:
            model_list = os.listdir(model_dir)
            if how_many is not None:
                model_list = np.random.permutation(model_list)[:how_many]
            for model_path in tqdm(model_list):
                MODELPATH = os.path.join(model_dir, model_path)
                model = model_utils.get_model(MODELPATH)
                scores, labels = get_scores(model, verbose=False)
                where_prop = np.nonzero(labels[:, attrs.index(target_prop)])[0]
                where_noprop = np.nonzero(
                    1 - labels[:, attrs.index(target_prop)])[0]
                X.append([
                    np.mean(scores[where_prop, 0] - scores[where_prop, 2]),
                    np.mean(scores[where_noprop, 0] - scores[where_noprop, 2])
                ])
                Y.append(i)
    return np.array(X), np.array(Y)


if __name__ == "__main__":
    import sys
    MODELPATH = sys.argv[1]

    ds = data_utils.CelebaWrapper("all", "victim")
    attrs = ds.attr_names
    _, cropped_dataloader = ds.get_loaders(5)

    if 1 == 2:
        model = model_utils.get_model(MODELPATH, use_prefix=False)

        scores, labels = get_scores(model)
        where_prop = np.nonzero(labels[:, attrs.index("Male")])[0]
        where_noprop = np.nonzero(1 - labels[:, attrs.index("Male")])[0]

        print("When property satisfied:")
        print(np.mean(scores[where_prop, 0] - scores[where_prop, 1]))
        print(np.mean(scores[where_prop, 0] - scores[where_prop, 2]))

        print("When property not satisfied:")
        print(np.mean(scores[where_noprop, 0] - scores[where_noprop, 1]))
        print(np.mean(scores[where_noprop, 0] - scores[where_noprop, 2]))

        prop_order = np.argsort(scores[where_prop, 1] - scores[where_prop, 2])
        noprop_order = np.argsort(
            scores[where_noprop, 1] - scores[where_noprop, 2])

        plt.plot(np.arange(where_prop.shape[0]),
                 scores[where_prop[prop_order], 1] - scores[where_prop[prop_order], 2],
                 label='prop=True')
        plt.plot(np.arange(where_noprop.shape[0]),
                 scores[where_noprop[noprop_order], 1] - scores[where_noprop[noprop_order], 2],
                 label='prop=False')

        plt.legend()
        plt.savefig("../visualize/lime_score_distrs.png")

    else:
        folder_paths = [
            ["adv/all/64_16/augment_none/", "adv/all/64_16/none/"],
            ["adv/male/64_16/augment_none/", "adv/male/64_16/none/"]
        ]
        X_train, Y_train = get_data_for_meta(folder_paths,
                                             attrs, how_many=10,
                                             target_prop="Male")

        clf = RandomForestClassifier(max_depth=3,
                                     n_estimators=10)
        clf.fit(X_train, Y_train)
        print("On train data:", clf.score(X_train, Y_train))
        blind_test_models = [
            ["adv/all/64_16/augment_vggface/", "victim/all/vggface/"],
            ["adv/male/64_16/augment_vggface/", "victim/male/vggface/"]
        ]
        X_test, Y_test = get_data_for_meta(blind_test_models,
                                           attrs, how_many=10,
                                           target_prop="Male")
        print("On unseen models:", clf.score(X_test, Y_test))
        print("Probabilities:", clf.predict_proba(X_test)[:, 1])
