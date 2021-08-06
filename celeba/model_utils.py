import torch as ch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import normalize
from utils import ensure_dir_exists, get_weight_layers, FakeReluWrapper, BasicWrapper


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_celeba/75_25"


class MyAlexNet(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 fake_relu: bool = False,
                 latent_focus: int = None) -> None:

        # expected input shape: 218,178
        if fake_relu:
            act_fn = BasicWrapper
        else:
            act_fn = nn.ReLU

        self.latent_focus = latent_focus

        super(MyAlexNet, self).__init__()
        layers = [
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            FakeReluWrapper(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            FakeReluWrapper(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            FakeReluWrapper(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            FakeReluWrapper(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            FakeReluWrapper(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]

        clf_layers = [
            nn.Linear(64 * 6 * 6, 64),
            FakeReluWrapper(inplace=True),
            nn.Linear(64, 32),
            FakeReluWrapper(inplace=True),
            nn.Linear(32, num_classes),
        ]

        mapping = {0: 1, 1: 4, 2: 7, 3: 9, 4: 11, 5: 1, 6: 3}
        if self.latent_focus is not None:
            if self.latent_focus < 5:
                layers[mapping[self.latent_focus]] = act_fn(inplace=True)
            else:
                clf_layers[mapping[self.latent_focus]] = act_fn(inplace=True)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:

        if latent is None:
            x = self.features(x)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            x = self.classifier(x)
            return x

        if latent not in list(range(7)):
            raise ValueError("Invald interal layer requested")

        # Pick activations just before previous layers
        # Since any intermediate functions that pool activations
        # Introduce invariance to further layers, so should be
        # Clubbed according to pooling

        if latent < 4:
            # Latent from Conv part of model
            mapping = {0: 2, 1: 5, 2: 7, 3: 9}
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == mapping[latent]:
                    return x.view(x.shape[0], -1)

        elif latent == 4:
            x = self.features(x)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            return x
        else:
            x = self.features(x)
            x = self.avgpool(x)
            x = ch.flatten(x, 1)
            for i, layer in enumerate(self.classifier):
                x = layer(x)
                if i == 2 * (latent - 5) + 1:
                    return x


def create_model(parallel=False, fake_relu=False, latent_focus=None):
    model = MyAlexNet(fake_relu=fake_relu, latent_focus=latent_focus).cuda()
    if parallel:
        model = nn.DataParallel(model)
    return model


def get_model(path, use_prefix=True, parallel=False, fake_relu=False, latent_focus=None):
    if use_prefix:
        path = os.path.join(BASE_MODELS_DIR, path)

    model = create_model(
        parallel=parallel, fake_relu=fake_relu, latent_focus=latent_focus)
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


# Function to extract model weights for all models in given directory
def get_model_features(model_dir, max_read=None,
                       start_n_conv=0, first_n_conv=np.inf,
                       start_n_fc=0, first_n_fc=np.inf,
                       conv_custom=None, fc_custom=None,
                       focus="all"):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        model = get_model(os.path.join(model_dir, mpath))

        if focus in ["all", "combined"]:
            dims_conv, fvec_conv = get_weight_layers(
                model.features, first_n=first_n_conv, conv=True,
                start_n=start_n_conv,
                custom_layers=conv_custom,
                include_all=focus == "combined")
            dims_fc, fvec_fc = get_weight_layers(
                model.classifier, first_n=first_n_fc,
                custom_layers=fc_custom,
                start_n=start_n_fc)

            vecs.append(fvec_conv + fvec_fc)
        elif focus == "conv":
            dims, fvec = get_weight_layers(
                model.features, first_n=first_n_conv,
                custom_layers=conv_custom,
                start_n=start_n_conv, conv=True)
            vecs.append(fvec)
        else:
            dims, fvec = get_weight_layers(
                model.classifier, first_n=first_n_fc,
                custom_layers=fc_custom,
                start_n=start_n_fc)
            vecs.append(fvec)

    if focus in ["all", "combined"]:
        return (dims_conv, dims_fc), vecs
    return dims, vecs
