import torch as ch
import torch.nn as nn
from torchvision import models
import numpy as np
from tqdm import tqdm
import torch.nn.utils.prune as prune
from torchvision.models import densenet121
from utils import get_weight_layers, ensure_dir_exists, BasicWrapper, FakeReluWrapper, check_if_inside_cluster
import os


BASE_MODELS_DIR = "<PATH_TO_MODELS>"


class BoneModel(nn.Module):
    def __init__(self,
                 n_inp: int,
                 fake_relu: bool = False,
                 latent_focus: int = None):
        if latent_focus is not None:
            if latent_focus not in [0, 1]:
                raise ValueError("Invalid interal layer requested")

        if fake_relu:
            act_fn = BasicWrapper
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus
        super(BoneModel, self).__init__()
        layers = [
            nn.Linear(n_inp, 128),
            FakeReluWrapper(inplace=True),
            nn.Linear(128, 64),
            FakeReluWrapper(inplace=True),
            nn.Linear(64, 1)
        ]

        mapping = {0: 1, 1: 3}
        if self.latent_focus is not None:
            layers[mapping[self.latent_focus]] = act_fn(inplace=True)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        if latent is None:
            return self.layers(x)

        if latent not in [0, 1]:
            raise ValueError("Invalid interal layer requested")

        # First, second hidden layers correspond to outputs of
        # Model layers 1, 3
        latent = (latent * 2) + 1

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == latent:
                return x


class BoneFullModel(nn.Module):
    def __init__(self,
                 fake_relu: bool = False,
                 latent_focus: int = None):
        # TODO: Implement latent focus
        # TODO: Implement fake_relu
        super(BoneFullModel, self).__init__()

        # Densenet
        self.model = densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, 1)

        # TODO: Implement fake_relu

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        # TODO: Implement latent functionality
        return self.model(x)


# Save model in specified directory
def save_model(model, split, ratio, prop_and_name, full_model=False):
    subfolder_prefix = os.path.join(split, str(ratio))
    if full_model:
        subfolder_prefix = os.path.join(subfolder_prefix, "full")

    # Make sure directory exists
    ensure_dir_exists(os.path.join(BASE_MODELS_DIR, subfolder_prefix))

    ch.save(model.state_dict(), os.path.join(
        BASE_MODELS_DIR, subfolder_prefix, prop_and_name))


# Load model from given directory
def load_model(path: str, fake_relu: bool = False,
               latent_focus: int = None,
               cpu: bool = False,
               full_model: bool = False):
    if full_model:
        model = BoneFullModel(fake_relu=fake_relu, latent_focus=latent_focus)
    else:
        model = BoneModel(1024, fake_relu=fake_relu, latent_focus=latent_focus)
    if cpu:
        model.load_state_dict(ch.load(path, map_location=ch.device("cpu")))
    else:
        model.load_state_dict(ch.load(path))

    model.eval()
    return model


# Get model path, given perameters
def get_model_folder_path(split, ratio, full_model=False):
    if full_model:
        return os.path.join(BASE_MODELS_DIR, split, ratio, "full")
    return os.path.join(BASE_MODELS_DIR, split, ratio)


# Function to extract model weights for all models in given directory
def get_model_features(model_dir, max_read=None, first_n=np.inf,
                       start_n=0, prune_ratio=None,
                       shift_to_gpu: bool = True,
                       fetch_models: bool = False,
                       models_provided: bool = False,
                       shuffle: bool = False):
    vecs, clfs = [], []

    if models_provided:
        iterator = model_dir
    else:
        iterator = os.listdir(model_dir)

    if shuffle:
        np.random.shuffle(iterator)

    encountered = 0

    for mpath in tqdm(iterator):
        if models_provided:
            model = mpath
        else:
            # Skip if path is directory
            if os.path.isdir(os.path.join(model_dir, mpath)):
                continue

            model = load_model(os.path.join(model_dir, mpath), cpu=not shift_to_gpu)

        if fetch_models:
            clfs.append(model)

        prune_mask = []
        # Prune weight layers, if requested
        if prune_ratio is not None:
            for layer in model.layers:
                if type(layer) == nn.Linear:
                    # Keep track of weight pruning mask
                    prune.l1_unstructured(
                        layer, name='weight', amount=prune_ratio)
                    prune_mask.append(layer.weight_mask.data.detach().cpu())

        # Get model params, shift to GPU
        dims, fvec = get_weight_layers(
            model, first_n=first_n, start_n=start_n,
            prune_mask=prune_mask)
        if shift_to_gpu:
            fvec = [x.cuda() for x in fvec]
        else:
            fvec = [x.cpu() for x in fvec]

        vecs.append(fvec)
        encountered += 1
        if encountered == max_read:
            break

    if fetch_models:
        return dims, vecs, clfs
    return dims, vecs


def get_pre_processor():
    # Load model
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Get rid of existing classification layer
    # Extract only features
    model.classifier = nn.Identity()
    return model


# Check with this model number exists
def check_if_exists(model_id, split, ratio, full_model=False):
    if full_model:
        model_check_path = os.path.join(
            BASE_MODELS_DIR, split, str(ratio), "full")
        # If this directory does not exist, we know adv models don't either
        if not os.path.exists(model_check_path):
            return False
    else:
        model_check_path = os.path.join(BASE_MODELS_DIR, split, str(ratio))
    # Return false if no directory exists
    if not os.path.exists(model_check_path):
        return False
    # If it does, check for models inside
    for model_name in os.listdir(model_check_path):
        if model_name.startswith("%d_" % model_id):
            return True
    return False


def get_models(folder_path, n_models: int = 1000,
               full_model: bool = False,
               cpu: bool = False,
               shuffle: bool = True):
    """
        Load all models from a given directory
    """
    paths = os.listdir(folder_path)
    if shuffle:
        paths = np.random.permutation(paths)
    encountered = 0

    models = []
    for mpath in tqdm(paths):
        # Skip if mpath is a directory
        if os.path.isdir(os.path.join(folder_path, mpath)):
            continue

        model = load_model(os.path.join(folder_path, mpath),
                           full_model=full_model, cpu=cpu)
        models.append(model)
        encountered += 1
        if encountered == n_models:
            break
    return models