import torch as ch
import torch.nn as nn
from torchvision import models
import numpy as np
from tqdm import tqdm
from utils import get_weight_layers, ensure_dir_exists, BasicWrapper, FakeReluWrapper
import os


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_boneage/"


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
    # Make sure directory exists
    ensure_dir_exists(savepath)
    ch.save(model.state_dict(), os.path.join(BASE_MODELS_DIR, savepath))


# Load model from given directory
def load_model(path, fake_relu: bool = False, latent_focus: int = None):
    model = BoneModel(1024, fake_relu=fake_relu, latent_focus=latent_focus)
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


def get_pre_processor():
    # Load model
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    # Get rid of existing classification layer
    # Extract only features
    model.classifier = nn.Identity()
    return model
