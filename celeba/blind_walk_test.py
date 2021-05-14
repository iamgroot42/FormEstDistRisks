import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import torch as ch
from torchvision import transforms
from sklearn.neural_network import MLPClassifier

import utils
import os

import matplotlib as mpl
import seaborn as sns
import pandas as pd
mpl.rcParams['figure.dpi'] = 200


def get_walk_distances(model, X, Y, norm, num_steps=50):
    def noise(x, norm):
        # params for noise magnitudes
        uni, std, scale = (0.005, 0.005, 0.01)
        if norm == 1:
            noisy = ch.from_numpy(np.random.laplace(loc=0.0,
                                                    scale=scale,
                                                    size=x.shape)).float()
        elif norm == 2:
            noisy = ch.normal(0, std, size=x.shape)
        elif norm == np.inf:
            noisy = ch.empty_like(x).uniform_(-uni, uni)
        return noisy.cuda()

    def preprocess_for_model(x):
        return (x - 0.5) / 0.5

    mag = 1
    delta = noise(X, norm)
    delta_base = delta.clone()
    delta.data = ch.min(ch.max(delta.detach(), -X), 1-X)
    X_r, delta_r, remaining = None, None, None
    with ch.no_grad():
        for t in range(num_steps):
            if t > 0:
                preds = model(preprocess_for_model(X_r + delta_r))
                new_remaining = ((preds[:, 0] >= 0) == Y[remaining])
                remaining[remaining] = new_remaining
            else:
                preds = model(preprocess_for_model(X + delta))
                remaining = ((preds[:, 0] >= 0) == Y)

            # If all boundaries reached, terminate
            if remaining.sum() == 0:
                break

            # Only query the data points that have still not flipped
            X_r = X[remaining]
            delta_r = delta[remaining]
            preds = model(preprocess_for_model(X_r + delta_r))
            # Move by one more step for points still in their original class
            mag += 1
            delta_r = delta_base[remaining]*mag
            # clip X+delta_r[remaining] to [0,1]
            delta_r.data = ch.min(ch.max(delta_r.detach(), -X_r), 1-X_r)
            delta[remaining] = delta_r.detach()

    # print(f"Number of steps = {t+1} | Failed to convert = {remaining.sum().item()}")
    return delta


def get_model(path):
    model = utils.FaceModel(512,
                            train_feat=True,
                            weight_init=None,
                            hidden=[64, 16]).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(path), strict=False)
    model.eval()
    return model


def get_distance_representaion(model, x, y):
    norms = [1, 2, np.inf]
    delta_vec = []
    for norm in norms:
        deltas = get_walk_distances(model, x, y, norm)
        diff = (x - deltas).cpu().numpy().reshape(x.shape[0], -1)
        delta_vec.append(np.linalg.norm(diff, ord=norm, axis=1))
    return np.stack(delta_vec, 1)


def get_distances(model, data_loader, attrs, verbose=True):
    iterator = enumerate(data_loader)
    if verbose:
        iterator = tqdm(iterator, total=len(data_loader))
    distances = []
    labels = []
    for i, (x, y) in iterator:
        x, y_ = x.cuda(), y[:, attrs.index("Smiling")].cuda()
        # dist_repr = get_distance_representaion(model, x, y_)
        # distances.append(dist_repr)
        labels.append(y.numpy())
        break

    labels = np.concatenate(labels, 0)
    print(np.mean(labels[:, attrs.index("Male")]))
    exit(0)
    distances = np.concatenate(distances)
    return distances, labels


def plot_meta_per_model(model_paths,
                        attrs,
                        data_loader,
                        target_prop="Male"):

    for ii, model_cluster in enumerate(model_paths):
        for j, model_dir in enumerate(model_cluster):
            model_list = os.listdir(model_dir)
            for model_path in tqdm(model_list):
                MODELPATH = os.path.join(model_dir, model_path)
                model = get_model(MODELPATH)

                # Get distance vectors for model
                scores, labels = get_distances(model,
                                               data_loader,
                                               attrs,
                                               verbose=False)
                where_prop = \
                    np.nonzero(labels[:, attrs.index(target_prop)])[0]
                where_noprop = \
                    np.nonzero(1 - labels[:, attrs.index(target_prop)])[0]

                # Look at distances (per norm) for property/not-property
                # Plot distributions to se if they are indistinguishable
                for i, n in enumerate(["1", "2", "inf"]):
                    # plt.clf()
                    # plt.plot(np.arange(where_prop.shape[0]),
                    #          scores[where_prop, i],
                    #          label="P=1",
                    #          marker='o',
                    #          markersize=5)
                    # plt.plot(np.arange(where_noprop.shape[0]),
                    #          scores[where_noprop, i],
                    #          label="P=0",
                    #          marker='o',
                    #          markersize=5)

                    columns = ["distance", "Property"]
                    data = []
                    for sc in scores[where_prop, i]:
                        data.append([sc, "P=1"])
                    for sc in scores[where_noprop, i]:
                        data.append([sc, "P=0"])
                    df = pd.DataFrame(data, columns=columns)
                    snsp = sns.displot(data=df,
                                       x=columns[0],
                                       hue=columns[1],
                                       kind="kde")
                    if n == "inf":
                        snsp.set(xlim=(0.6, 1.5))
                        snsp.set(ylim=(0, 9.5))
                    snsp.savefig("../visualize/norm_%d_%d_%s_image.png" % (ii, j, n))

                    # plt.hist(scores[where_prop, i],
                    #          bins=50,
                    #          alpha=0.7,
                    #          label="P=1")
                    # plt.hist(scores[where_noprop, i],
                    #          bins=50,
                    #          alpha=0.7,
                    #          label="P=0")
                    # plt.legend()
                    # plt.title("Norm %s" % n)
                    # plt.savefig("../visualize/norm_%d_%d_%s_image.png" % (ii, j, n))

                print("Saved three distribution graphs!")
                break
    exit(0)


def distance_features(model_paths, x, y):
    X, Y = [], []
    for i, model_cluster in enumerate(model_paths):
        for j, model_dir in enumerate(model_cluster):
            model_list = os.listdir(model_dir)
            for model_path in tqdm(model_list):
                MODELPATH = os.path.join(model_dir, model_path)
                model = get_model(MODELPATH)

                repr = get_distance_representaion(model, x, y)
                X.append(repr)
                Y.append(i)

    return np.array(X), np.array(Y)


if __name__ == "__main__":
    constants = utils.Celeb()
    ds = constants.get_dataset()
    transform = transforms.Compose([transforms.ToTensor()])

    def get_input_tensors(img):
        # unsqeeze converts single image to batch of 1
        return transform(img).unsqueeze(0)

    td = utils.CelebACustomBinary(
        "/p/adversarialml/as9rw/datasets/celeba_raw_crop/splits/70_30/all/split_2/test",
        transform=transform)

    attrs = constants.attr_names
    cropped_dataloader = DataLoader(td,
                                    batch_size=1000,
                                    shuffle=False)
                                    # shuffle=True)

    common_prefix = "/u/as9rw/work/fnb/implems/celeba_models_split/70_30/"
    folder_paths = [
        [
            common_prefix + "split_2/all/64_16/augment_none/",
            common_prefix + "split_2/all/64_16/none/",
        ],
        [
            common_prefix + "split_2/male/64_16/augment_none/",
            common_prefix + "split_2/male/64_16/none/",
        ]
    ]

    # Plot distribution
    plot_meta_per_model(folder_paths,
                        attrs,
                        cropped_dataloader)

    # Get a shuffled set of data points
    x, y = next(iter(cropped_dataloader))
    x, y = x.cuda(), y[:, attrs.index("Smiling")].cuda()

    X_train, Y_train = distance_features(folder_paths, x, y)
    X_train = X_train.reshape((X_train.shape[0], -1))
    clf = MLPClassifier(hidden_layer_sizes=(30, 30))
    clf.fit(X_train, Y_train)

    print("Training accuracy : %.2f" % clf.score(X_train, Y_train))
    blind_test_models = [
        [
            common_prefix + "split_2/all/64_16/augment_vggface/",
            common_prefix + "split_1/all/vggface/"
        ],
        [
            common_prefix + "split_2/male/64_16/augment_vggface/",
            common_prefix + "split_1/male/vggface/"
        ]
    ]

    X_test, Y_test = distance_features(blind_test_models, x, y)
    X_test = X_test.reshape((X_test.shape[0], -1))
    print("On unseen models: %2f" % clf.score(X_test, Y_test))
    print("Probabilities:", clf.predict_proba(X_test)[:, 1])
