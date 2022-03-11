import numpy as np
from tqdm import tqdm
from typing import List
import torch as ch
import torch.nn as nn
import os
from utils import check_if_inside_cluster, make_affinity_features
from joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network._base import ACTIVATIONS


BASE_MODELS_DIR = "<PATH_TO_MODELS>"
ACTIVATION_DIMS = [32, 16, 8, 1]


class PortedMLPClassifier(nn.Module):
    def __init__(self):
        super(PortedMLPClassifier, self).__init__()
        layers = [
            nn.Linear(in_features=42, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: ch.Tensor,
                latent: int = None,
                get_all: bool = False,
                detach_before_return: bool = True,
                on_cpu: bool = False):
        """
        Args:
            x: Input tensor of shape (batch_size, 42)
            latent: If not None, return only the latent representation. Else, get requested latent layer's output
            get_all: If True, return all activations
            detach_before_return: If True, detach the latent representation before returning it
            on_cpu: If True, return the latent representation on CPU
        """
        if latent is None and not get_all:
            return self.layers(x)

        if latent not in [0, 1, 2] and not get_all:
            raise ValueError("Invald interal layer requested")

        if latent is not None:
            # First three hidden layers correspond to outputs of
            # Model layers 1, 3, 5
            latent = (latent * 2) + 1
        valid_for_all = [1, 3, 5, 6]

        latents = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Append activations for all layers (post-activation only)
            if get_all and i in valid_for_all:
                if detach_before_return:
                    if on_cpu:
                        latents.append(x.detach().cpu())
                    else:
                        latents.append(x.detach())
                else:
                    if on_cpu:
                        latents.append(x.cpu())
                    else:
                        latents.append(x)
            if i == latent:
                if on_cpu:
                    return x.cpu()
                else:
                    return x

        return latents


def port_mlp_to_ch(clf):
    """
        Extract weights from MLPClassifier and port
        to PyTorch model.
    """
    nn_model = PortedMLPClassifier()
    i = 0
    for (w, b) in zip(clf.coefs_, clf.intercepts_):
        w = ch.from_numpy(w.T).float()
        b = ch.from_numpy(b).float()
        nn_model.layers[i].weight = nn.Parameter(w)
        nn_model.layers[i].bias = nn.Parameter(b)
        i += 2  # Account for ReLU as well

    nn_model = nn_model.cuda()
    return nn_model


def convert_to_torch(clfs):
    """
        Port given list of MLPClassifier models to
        PyTorch models
    """
    return np.array([port_mlp_to_ch(clf) for clf in clfs], dtype=object)


def layer_output(data, MLP, layer=0, get_all=False):
    """
        For a given model and some data, get output for each layer's activations < layer.
        If get_all is True, return all activations unconditionally.
    """
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
def get_model_representations(folder_path, label, first_n=np.inf,
                              n_models=1000, start_n=0,
                              fetch_models: bool = False,
                              shuffle: bool = True,
                              models_provided: bool = False):
    """
        If models_provided is True, folder_path will actually be a list of models.
    """
    if models_provided:
        models_in_folder = folder_path
    else:
        models_in_folder = os.listdir(folder_path)

    if shuffle:
        # Shuffle
        np.random.shuffle(models_in_folder)

    # Pick only N models
    models_in_folder = models_in_folder[:n_models]

    w, labels, clfs = [], [], []
    for path in tqdm(models_in_folder):
        if models_provided:
            clf = path
        else:
            clf = load_model(os.path.join(folder_path, path))
        if fetch_models:
            clfs.append(clf)

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

    if fetch_models:
        return w, labels, dims, clfs
    return w, labels, dims


def get_model(max_iter=40,
              hidden_layer_sizes=(32, 16, 8),):
    """
        Create new MLPClassifier model
    """
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=max_iter)
    return clf


def get_models(folder_path, n_models=1000, shuffle=True):
    """
        Load models from given directory.
    """
    paths = os.listdir(folder_path)
    if shuffle:
        paths = np.random.permutation(paths)
    paths = paths[:n_models]

    models = []
    for mpath in tqdm(paths):
        model = load_model(os.path.join(folder_path, mpath))
        models.append(model)
    return models


def save_model(clf, path):
    dump(clf, path)


def load_model(path):
    return load(path)


def get_models_path(property, split, value=None):
    if value is None:
        return os.path.join(BASE_MODELS_DIR, property, split)
    return os.path.join(BASE_MODELS_DIR,  property, split, value)


def get_model_activation_representations(
        models: List[PortedMLPClassifier],
        data, label, detach: bool = True,
        verbose: bool = True):
    w = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator)
    for model in iterator:
        activations = model(data, get_all=True,
                            detach_before_return=detach)
        # Skip last feature (logit)
        activations = activations[:-1]

        w.append([act.float() for act in activations])
    labels = np.array([label] * len(w))
    labels = ch.from_numpy(labels)

    # Make numpy object (to support sequence-based indexing)
    w = np.array(w, dtype=object)

    # Get dimensions of feature representations
    dims = [x.shape[1] for x in w[0]]

    return w, labels, dims


def make_activation_data(models_pos, models_neg, seed_data,
                         detach=True, verbose=True, use_logit=False):
    # Construct affinity graphs
    pos_model_scores = make_affinity_features(
        models_pos, seed_data,
        detach=detach, verbose=verbose,
        use_logit=use_logit)
    neg_model_scores = make_affinity_features(
        models_neg, seed_data,
        detach=detach, verbose=verbose,
        use_logit=use_logit)
    # Convert all this data to loaders
    X = ch.cat((pos_model_scores, neg_model_scores), 0)
    Y = ch.cat((ch.ones(len(pos_model_scores)),
                ch.zeros(len(neg_model_scores))))
    return X, Y
