import torch as ch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import normalize
from utils import ensure_dir_exists, get_weight_layers


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_celeba/75_25"


class MyAlexNet(nn.Module):
    def __init__(self, num_classes: int = 1) -> None:
        # 218,178
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 6 * 6, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: ch.Tensor) -> ch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = ch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(parallel=False):
    model = MyAlexNet().cuda()
    if parallel:
        model = nn.DataParallel(model)
    return model


def get_model(path, use_prefix=True, parallel=False):
    if use_prefix:
        path = os.path.join(BASE_MODELS_DIR, path)

    model = create_model(parallel=parallel)
    model.load_state_dict(ch.load(path), strict=False)

    if parallel:
        model = nn.DataParallel(model)

    model.eval()
    return model


def save_model(model, split, property, ratio, name, dataparallel=False):
    savepath = os.path.join(split, property, ratio, name)
    # Make sure directory exists
    ensure_dir_exists(savepath)
    if dataparallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    ch.save(state_dict, os.path.join(BASE_MODELS_DIR, savepath))


def get_latents(mainmodel, dataloader, method_type, to_normalize=False):
    all_stats = []
    all_latent = []
    for (x, y) in dataloader:

        # Not-last latent features needed
        if method_type < 0:
            latent = mainmodel(x.cuda(),
                               deep_latent=-method_type).detach()
            # latent = latent.view(latent.shape[0], -1)
        # Latent features needed
        elif method_type in [0, 1, 4, 5]:
            latent = mainmodel(x.cuda(), only_latent=True).detach()
        # Use logit scores
        elif method_type == 2:
            latent = mainmodel(x.cuda()).detach()[:, 0]
        # Use post-sigmoid outputs
        elif method_type == 3:
            latent = ch.sigmoid(mainmodel(x.cuda()).detach()[:, 0])

        all_latent.append(latent.cpu().numpy())
        all_stats.append(y.cpu().numpy())

    all_latent = np.concatenate(all_latent, 0)
    all_stats = np.concatenate(all_stats)

    # Normalize according to max entry?
    if to_normalize:
        # all_latent /= np.max(all_latent, 1, keepdims=True)
        all_latent = normalize(all_latent)

    if method_type == 5:
        all_latent = np.sort(all_latent, 1)

    return all_latent, all_stats


def get_features_for_model(dataloader, MODELPATH, method_type):
    # Load model
    model = get_model(MODELPATH)

    # Get latent representations
    lat, sta = get_latents(model, dataloader, method_type)
    return (lat, sta)


def extract_dl_model_weights(model):
    # Get sequential-layer weights
    attr_accessor = model.module
    weights, biases = [], []
    for layer in attr_accessor.dnn.modules():
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().cpu().numpy())
            biases.append(layer.bias.detach().cpu().numpy())

    feature_vec = []
    for w, b in zip(weights, biases):
        feature_vec.append(w)
        feature_vec.append(b)

    feature_vec = np.concatenate(feature_vec)
    return feature_vec


# Function to extract model weights for all models in given directory
def get_model_features(model_dir, max_read=None,
                       first_n=np.inf, conv_focus=False):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        model = get_model(os.path.join(model_dir, mpath))

        if conv_focus:
            dims, fvec = get_weight_layers(
                model.features, first_n=first_n, conv=True)
        else:
            dims, fvec = get_weight_layers(model.classifier, first_n=first_n)

        vecs.append(fvec)

    return dims, vecs
