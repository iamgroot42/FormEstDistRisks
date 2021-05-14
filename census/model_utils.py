import numpy as np
from tqdm import tqdm
import os
from joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_census/75_25"


def layer_output(data, MLP, layer=0):
    L = data.copy()
    for i in range(layer):
        L = ACTIVATIONS['relu'](
            np.matmul(L, MLP.coefs_[i]) + MLP.intercepts_[i])
    return L


def extract_model_features(paths, example_mode, multimode, sample):
    w, b = [], []
    labels = []
    for i, path_seg in tqdm(enumerate(paths)):
        models_in_folder = os.listdir(path_seg)
        np.random.shuffle(models_in_folder)
        if not multimode:
            models_in_folder = models_in_folder[:sample]
        for path in models_in_folder:
            clf = load(os.path.join(path_seg, path))
            # Look at initial layer weights, biases

            if example_mode is not None:
                preds = layer_output(example_mode, clf, layer=3)
                for pred in preds:
                    w.append(pred)
                    labels.append(i)
            else:
                processed = clf.coefs_[0]
                processed = np.concatenate(
                    (np.mean(processed, 1), np.mean(processed ** 2, 1)))
                w.append(processed)
                b.append(clf.intercepts_[0])
                labels.append(i)

    w = np.array(w)
    b = np.array(w)
    labels = np.array(labels)
    return (w, b), labels


def get_model(max_iter=200,
              early_stopping=True,
              hidden_layer_sizes=(32, 16, 8),
              validation_fraction=0.25):
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter,
                        early_stopping=early_stopping,
                        validation_fraction=validation_fraction)
    return clf


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, split, property)
    return os.path.join(BASE_MODELS_DIR, split, property, value)
