import torch as ch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils import get_weight_layers, ensure_dir_exists
import os


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_boneage/"


class BoneModel(nn.Module):
    def __init__(self, n_inp):
        super(BoneModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_inp, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

    def forward(self, x, latent=None):
        if latent is None:
            return self.layers(x)

        if latent not in [0, 1]:
            raise ValueError("Invald interal layer requested")
        # First, second hidden layers correspond to outputs of
        # Model layers 1, 3
        latent = (latent * 2) + 1

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == latent:
                return x


# Save model in specified directory
def save_model(model, split, prop_and_name):
    savepath = os.path.join(split, prop_and_name)
    # Make sure cirectory exists
    ensure_dir_exists(savepath)
    ch.save(model.state_dict(), os.path.join(BASE_MODELS_DIR, savepath))


# Load model from given directory
def load_model(path):
    model = BoneModel(1024)
    model.load_state_dict(ch.load(path))
    model.eval()
    return model


# Get model path, given perameters
def get_model_folder_path(split, ratio):
    return os.path.join(BASE_MODELS_DIR, split, ratio)


# Function to extract model weights for all models in given directory
def get_model_features(model_dir, max_read=None, first_n=np.inf):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        model = load_model(os.path.join(model_dir, mpath))

        # Get model params, shift to GPU
        dims, fvec = get_weight_layers(model, first_n=first_n)
        fvec = [x.cuda() for x in fvec]

        vecs.append(fvec)

    return dims, vecs
