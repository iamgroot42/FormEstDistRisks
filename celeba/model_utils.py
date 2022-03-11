import torch as ch
import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision.models import inception_v3, densenet121
from sklearn.preprocessing import normalize
from utils import check_if_inside_cluster, ensure_dir_exists, get_weight_layers, FakeReluWrapper, BasicWrapper


BASE_MODELS_DIR = "<PATH_TO_MODELS>"


class InceptionModel(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 fake_relu: bool = False,
                 latent_focus: int = None) -> None:
        super(InceptionModel, self).__init__()
        self.model = densenet121(num_classes=num_classes) #, aux_logits=False)

    def forward(self, x: ch.Tensor, latent: int = None) -> ch.Tensor:
        return self.model(x)


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


def create_model(parallel: bool = False, fake_relu: bool = False,
                 latent_focus=None, is_large: bool = False,
                 cpu: bool = False):
    """
        Create and return a model.
    """
    if is_large:
        model = InceptionModel(fake_relu=fake_relu, latent_focus=latent_focus)
    else:
        model = MyAlexNet(fake_relu=fake_relu, latent_focus=latent_focus)
    if not cpu:
        model = model.cuda()
    if parallel:
        model = nn.DataParallel(model)
    return model


def get_model(path, use_prefix=True, parallel: bool = False,
              fake_relu: bool = False, latent_focus=None,
              cpu: bool = False):
    if use_prefix:
        path = os.path.join(BASE_MODELS_DIR, path)

    model = create_model(
        parallel=parallel, fake_relu=fake_relu,
        latent_focus=latent_focus, cpu=cpu)

    if cpu:
        model.load_state_dict(
            ch.load(path, map_location=ch.device("cpu")), strict=False)
    else:
        model.load_state_dict(ch.load(path), strict=False)

    if parallel:
        model = nn.DataParallel(model)

    model.eval()
    return model


def get_models(folder_path, n_models: int = 1000, cpu: bool = False):
    paths = np.random.permutation(os.listdir(folder_path))[:n_models]

    models = []
    for mpath in tqdm(paths):
        # Folder for adv models (or others): skip
        if os.path.isdir(os.path.join(folder_path, mpath)):
            continue

        model = get_model(os.path.join(folder_path, mpath), cpu=cpu)
        models.append(model)
    return models


def save_model(model, split, property, ratio,
               name, dataparallel=False, is_adv=False,
               adv_folder_name="adv_train",
               is_large=False):
    if is_adv:
        subfolder_prefix = os.path.join(
            split, property, ratio, adv_folder_name)
    elif is_large:
        subfolder_prefix = os.path.join(split, property, ratio, "inception")
    else:
        subfolder_prefix = os.path.join(split, property, ratio)

    # Make sure directory exists
    ensure_dir_exists(os.path.join(BASE_MODELS_DIR, subfolder_prefix))

    if dataparallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    ch.save(state_dict, os.path.join(BASE_MODELS_DIR, subfolder_prefix, name))


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
                       focus="all", shift_to_gpu=True,
                       get_stats=False):
    vecs, stats = [], []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)

    print("Found %d models to read in %s" % (len(iterator), model_dir))

    dims, dims_fc, dims_conv = None, None, None
    for mpath in tqdm(iterator):
        # Folder for adv models (or others): skip
        if os.path.isdir(os.path.join(model_dir, mpath)):
            continue

        if get_stats:
            # Get test acc (and robustness) if requested
            numbers = mpath.replace(".pth", "").split("_")[1:]
            stat = [float(numbers[0])]
            if "adv" in mpath:
                # Extract two numbers
                adv_acc = float(numbers[2].replace("adv", ""))
                stat.append(adv_acc)
            stats.append(stat)

        model = get_model(os.path.join(model_dir, mpath), cpu=not shift_to_gpu)

        if focus in ["all", "combined"]:
            dims_conv, fvec_conv = get_weight_layers(
                model.features, first_n=first_n_conv, conv=True,
                start_n=start_n_conv,
                custom_layers=conv_custom,
                include_all=focus == "combined",)
            dims_fc, fvec_fc = get_weight_layers(
                model.classifier, first_n=first_n_fc,
                custom_layers=fc_custom,
                start_n=start_n_fc,)

            vecs.append(fvec_conv + fvec_fc)
        elif focus == "conv":
            dims, fvec = get_weight_layers(
                model.features, first_n=first_n_conv,
                custom_layers=conv_custom,
                start_n=start_n_conv, conv=True,)
            vecs.append(fvec)
        else:
            dims, fvec = get_weight_layers(
                model.classifier, first_n=first_n_fc,
                custom_layers=fc_custom,
                start_n=start_n_fc,)
            vecs.append(fvec)

        # Number of requested models read- break
        if len(vecs) == max_read:
            break

    if focus in ["all", "combined"]:
        if get_stats:
            return (dims_conv, dims_fc), vecs, stats
        return (dims_conv, dims_fc), vecs
    if get_stats:
        return dims, vecs, stats
    return dims, vecs


# Check with this model number exists
def check_if_exists(model_id, ratio, filter, split, is_adv, adv_folder_name, is_large=False):
    # Get folder of models to check
    if is_adv:
        subfolder_prefix = os.path.join(split, filter, str(ratio), adv_folder_name)
    elif is_large:
        subfolder_prefix = os.path.join(split, filter, str(ratio), "inception")
    else:
        subfolder_prefix = os.path.join(split, filter, str(ratio))
    model_check_path = os.path.join(BASE_MODELS_DIR, subfolder_prefix)
    for model_name in os.listdir(model_check_path):
        if model_name.startswith(model_id + "_"):
            return True
    return False