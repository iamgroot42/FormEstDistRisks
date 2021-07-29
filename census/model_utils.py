import numpy as np
from tqdm import tqdm
import torch as ch
import os
from joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_census/50_50_new"


def layer_output(data, MLP, layer=0, get_all=False):
    L = data.copy()
    all = []
    for i in range(layer):
        L = ACTIVATIONS['relu'](
            np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
        if get_all:
            all.append(L)
    if get_all:
        return all
    return L


# Load models from directory, return feature representations
def get_model_representations(folder_path, label, first_n=np.inf, start_n=0):
    models_in_folder = os.listdir(folder_path)
    # np.random.shuffle(models_in_folder)
    w, labels = [], []
    for path in tqdm(models_in_folder):
        clf = load_model(os.path.join(folder_path, path))

        # Extract model parameters
        weights = [ch.from_numpy(x) for x in clf.coefs_]
        dims = [w.shape[0] for w in weights]
        biases = [ch.from_numpy(x) for x in clf.intercepts_]
        processed = [ch.cat((w, ch.unsqueeze(b, 0)), 0).float().T
                     for (w, b) in zip(weights, biases)]

        # Use parameters only from first N layers
        # and starting from start_n
        if first_n != np.inf:
            processed = processed[start_n:first_n]
            dims = dims[start_n:first_n]

        w.append(processed)
        labels.append(label)

    labels = np.array(labels)

    w = np.array(w, dtype=object)
    labels = ch.from_numpy(labels)

    return w, labels, dims


def get_model(max_iter=40,
              hidden_layer_sizes=(32, 16, 8),):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter)
    return clf


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, split, property)
    return os.path.join(BASE_MODELS_DIR, split, property, value)
