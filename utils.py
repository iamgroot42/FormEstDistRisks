"""
    Large collection of utility functions and classes that are
    shared across all datasets. In the long run, we would have a common outer
    structure for all datasets, with dataset-specific configuration files.
"""
import torch as ch
import numpy as np
from os import environ
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR
from robustness.tools import folder
from robustness.tools.misc import log_statement

from cleverhans.future.torch.attacks.projected_gradient_descent import projected_gradient_descent
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import pandas as pd
from typing import List
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def log(x):
    print(f"{bcolors.WARNING}%s{bcolors.ENDC}" % x)


class DataPaths:
    def __init__(self, name, data_path, stats_path):
        self.name = name
        self.data_path = data_path
        self.dataset = self.dataset_type(data_path)
        self.models = {'nat': None, 'l1': None,
                       'l2': None, 'temp': None, 'linf': None}
        self.model_prefix = {}
        self.stats_path = stats_path

    def get_dataset(self):
        return self.dataset

    def get_model(self, m_type, arch='resnet50', parallel=False):
        model_path = self.models.get(m_type, None)
        if not model_path:
            model_path = m_type
        else:
            model_path = self.model_prefix[arch] + self.models[m_type]
        model_kwargs = {
            'arch': arch,
            'dataset': self.dataset,
            'resume_path': model_path,
            'parallel': parallel
        }
        model, _ = make_and_restore_model(**model_kwargs)
        model.eval()
        return model

    def get_stats(self, m_type, arch='resnet50'):
        stats_path = os.path.join(self.stats_path, arch, m_type, "stats")
        return get_stats(stats_path)

    def get_deltas(self, m_type, arch='resnet50', numpy=False):
        ext = ".npy" if numpy else ".txt"
        deltas_path = os.path.join(
            self.stats_path, arch, m_type, "deltas" + ext)
        return get_sensitivities(deltas_path, numpy=numpy)


class BinaryCIFAR(DataPaths):
    def __init__(self, path):
        self.dataset_type = GenericBinary
        super(BinaryCIFAR, self).__init__('binary_cifar10', path, None)


class CIFAR10(DataPaths):
    def __init__(self, data_path=None):
        self.dataset_type = CIFAR
        datapath = "/p/adversarialml/as9rw/datasets/cifar10" if data_path is None else data_path
        super(CIFAR10, self).__init__('cifar10',
                                      datapath,
                                      "/p/adversarialml/as9rw/cifar10_stats/")
        self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_cifar10/"
        self.model_prefix['densenet169'] = "/p/adversarialml/as9rw/models_cifar10_densenet/"
        self.model_prefix['vgg19'] = "/p/adversarialml/as9rw/models_cifar10_vgg/"
        self.models['nat'] = "cifar_nat.pt"
        self.models['linf'] = "cifar_linf_8.pt"
        self.models['l2'] = "cifar_l2_0_5.pt"


class RobustCIFAR10(DataPaths):
    def __init__(self, datapath, stats_prefix):
        self.dataset_type = RobustCIFAR
        super(RobustCIFAR10, self).__init__('robustcifar10',
                                            datapath, stats_prefix)


class SVHN10(DataPaths):
    def __init__(self):
        self.dataset_type = SVHN
        super(SVHN10, self).__init__('svhn',
                                     "/p/adversarialml/as9rw/datasets/svhn",
                                     "/p/adversarialml/as9rw/svhn_stats/")
        self.model_prefix['vgg16'] = "/p/adversarialml/as9rw/models_svhn_vgg/"
        self.models['nat'] = "svhn_nat.pt"
        self.models['linf'] = "svhn_linf_4.pt"
        self.models['l2'] = "svhn_l2_0_5.pt"


class ImageNet1000(DataPaths):
    def __init__(self, data_path=None):
        self.dataset_type = ImageNet
        datapath = "/p/adversarialml/as9rw/datasets/imagenet/" if data_path is None else data_path
        super(ImageNet1000, self).__init__('imagenet1000',
                                           datapath,
                                           "/p/adversarialml/as9rw/imagenet_stats/")
        self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_imagenet/"
        self.models['nat'] = "imagenet_nat.pt"
        self.models['l2'] = "imagenet_l2_3_0.pt"
        self.models['linf'] = "imagenet_linf_4.pt"


def read_given_dataset(data_path):
    train_transform = transforms.Compose([])

    train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
    train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
    train_set = folder.TensorDataset(
        train_data, train_labels, transform=train_transform)

    X, Y = [], []
    for i in range(len(train_set)):
        X.append(train_set[i][0])
        Y.append(train_set[i][1].numpy())
    return (X, Y)


def scaled_values(val, mean, std, eps=1e-10):
    return (val - np.repeat(np.expand_dims(mean, 1), val.shape[1], axis=1)) / (np.expand_dims(std, 1) + eps)


def load_all_loader_data(data_loader):
    images, labels = [], []
    for (image, label) in data_loader:
        images.append(image)
        labels.append(label)
    images = ch.cat(images)
    labels = ch.cat(labels)
    return (images, labels)


def load_all_data(ds):
    batch_size = 512
    _, test_loader = ds.make_loaders(
        batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
    return load_all_loader_data(test_loader)


def get_sensitivities(path, numpy=False):
    log_statement("==> Loading Delta Values")
    # Directly load, if numpy array
    if numpy:
        return np.load(path)
    # Process, if text file
    features = []
    with open(path, 'r') as f:
        for line in tqdm(f):
            values = np.array([float(x) for x in line.rstrip('\n').split(',')])
            features.append(values)
    return np.array(features)


def best_target_image(mat, which=0):
    sum_m = []
    for i in range(mat.shape[1]):
        mat_interest = mat[mat[:, i] != np.inf, i]
        sum_m.append(np.average(np.abs(mat_interest)))
    best = np.argsort(sum_m)
    return best[which]


def get_statistics(diff):
    l1_norms = ch.sum(ch.abs(diff), dim=1)
    l2_norms = ch.norm(diff, dim=1)
    linf_norms = ch.max(ch.abs(diff), dim=1)[0]
    return (l1_norms, l2_norms, linf_norms)


def get_stats(base_path):
    mean = np.load(os.path.join(base_path, "feature_mean.npy"))
    std = np.load(os.path.join(base_path, "feature_std.npy"))
    return mean, std


def get_logits_layer_name(arch):
    if "vgg" in arch:
        return "module.model.classifier.weight"
    elif "resnet" in arch:
        return "module.model.fc.weight"
    elif "densenet" in arch:
        return "module.model.linear.weight"
    return None


class SpecificLayerModel(ch.nn.Module):
    def __init__(self, model, layer_index):
        super(SpecificLayerModel, self).__init__()
        self.model = model
        self.layer_index = layer_index

    def forward(self, x):
        logits, _ = self.model(x, this_layer_input=self.layer_index)
        return logits


class MadryToNormal:
    def __init__(self, model, fake_relu=False):
        self.model = model
        self.fake_relu = fake_relu
        self.model.eval()

    def __call__(self, x):
        logits, _ = self.model(x, fake_relu=self.fake_relu)
        return logits

    def eval(self):
        return self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()


def classwise_pixelwise_stats(loader, num_classes=10, classwise=False):
    images, labels = load_all_loader_data(loader)
    if not classwise:
        return ch.mean(images, 0), ch.std(images, 0)
    means, stds = [], []
    for i in range(num_classes):
        specific_images = images[labels == i]
        means.append(ch.mean(specific_images, 0))
        stds.append(ch.std(specific_images, 0))
    return means, stds


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Input size: [batch, n_features]
        # Output size: [batch, 3, 32, 32]
        # Expects 48, 4, 4
        self.dnn = nn.Sequential(
            nn.Linear(512, 768),
            nn.BatchNorm1d(768),
            nn.ReLU())
        self.decoder = nn.Sequential(
            # [batch, 24, 8, 8]
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            # [batch, 12, 16, 16]
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            # [batch, 3, 32, 32]
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_ = self.dnn(x)
        x_ = x_.view(x_.shape[0], 48, 4, 4)
        return self.decoder(x_)


class BasicDataset(ch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def compute_delta_values(logits, weights, actual_label=None):
    # Iterate through all possible classes, calculate flip probabilities
    actual_label = ch.argmax(logits)
    numerator = (logits[actual_label] - logits).unsqueeze(1)
    denominator = weights - weights[actual_label]
    numerator = numerator.repeat(1, denominator.shape[1])
    delta_values = ch.div(numerator, denominator)
    delta_values[actual_label] = np.inf
    return delta_values


def get_these_params(model, identifier):
    for name, param in model.state_dict().items():
        if name == identifier:
            return param
    return None


def flash_utils(args):
    log_statement("==> Arguments:")
    for arg in vars(args):
        print(arg, " : ", getattr(args, arg))


class MNISTFlatModel(nn.Module):
    def __init__(self):
        super(MNISTFlatModel, self).__init__()
        n_feat = 28 * 28
        self.dnn = nn.Sequential(
            nn.Linear(n_feat, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 10))

    def forward(self, x):
        x_ = x.view(x.shape[0], -1)
        return self.forward(x_)


def filter(df, condition, ratio, verbose=True):
    qualify = np.nonzero((condition(df)).to_numpy())[0]
    notqualify = np.nonzero(np.logical_not((condition(df)).to_numpy()))[0]
    current_ratio = len(qualify) / (len(qualify) + len(notqualify))
    # If current ratio less than desired ratio, subsample from non-ratio
    if verbose:
        print("Changing ratio from %.2f to %.2f" % (current_ratio, ratio))
    if current_ratio <= ratio:
        np.random.shuffle(notqualify)
        if ratio < 1:
            nqi = notqualify[:int(((1-ratio) * len(qualify))/ratio)]
            return pd.concat([df.iloc[qualify], df.iloc[nqi]])
        return df.iloc[qualify]
    else:
        np.random.shuffle(qualify)
        if ratio > 0:
            qi = qualify[:int((ratio * len(notqualify))/(1 - ratio))]
            return pd.concat([df.iloc[qi], df.iloc[notqualify]])
        return df.iloc[notqualify]


def get_cropped_faces(cropmodel, x):
    def renormalize(z): return (z * 0.5) + 0.5

    images = [Image.fromarray(
        (255 * np.transpose(renormalize(x_.numpy()), (1, 2, 0))).astype('uint8')) for x_ in x]
    crops = cropmodel(images)

    x_cropped = []
    indices = []
    for j, cr in enumerate(crops):
        if cr is not None:
            x_cropped.append(cr)
            indices.append(j)

    return ch.stack(x_cropped, 0), indices


# Function to extract model parameters
def get_weight_layers(m, normalize=False, transpose=True,
                      first_n=np.inf, start_n=0,
                      custom_layers=None,
                      conv=False, include_all=False,
                      prune_mask=[]):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Sort and store desired layers, if specified
    custom_layers = sorted(custom_layers) if custom_layers is not None else None

    track = 0
    for name, param in m.named_parameters():
        if "weight" in name:

            param_data = param.data.detach().cpu()

            # Apply pruning masks if provided
            if len(prune_mask) > 0:
                param_data = param_data * prune_mask[track]
                track += 1

            if transpose:
                param_data = param_data.T

            weights.append(param_data)
            if conv:
                dims.append(weights[-1].shape[2])
                dim_kernels.append(weights[-1].shape[0] * weights[-1].shape[1])
            else:
                dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

        # Assume each layer has weight & bias
        i += 1

        if custom_layers is None:
            # If requested, start looking from start_n layer
            if (i - 1) // 2 < start_n:
                dims, dim_kernels, weights, biases = [], [], [], []
                continue

            # If requested, look at only first_n layers
            if i // 2 > first_n - 1:
                break
        else:
            # If this layer was not asked for, omit corresponding weights & biases
            if i // 2 != custom_layers[j // 2]:
                dims = dims[:-1]
                dim_kernels = dim_kernels[:-1]
                weights = weights[:-1]
                biases = biases[:-1]
            else:
                # Specified layer was found, increase count
                j += 1

            # Break if all layers were processed
            if len(custom_layers) == j // 2:
                break

    if custom_layers is not None and len(custom_layers) != j // 2:
        raise ValueError("Custom layers requested do not match actual model")

    if include_all:
        if conv:
            middle_dim = weights[-1].shape[3]
        else:
            middle_dim = weights[-1].shape[1]

    if normalize:
        min_w = min([ch.min(x).item() for x in weights])
        max_w = max([ch.max(x).item() for x in weights])
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
        weights = [w / max_w for w in weights]

    cctd = []
    for w, b in zip(weights, biases):
        if conv:
            b_exp = b.unsqueeze(0).unsqueeze(0)
            b_exp = b_exp.expand(w.shape[0], w.shape[1], 1, -1)
            combined = ch.cat((w, b_exp), 2).transpose(2, 3)
            combined = combined.view(-1, combined.shape[2], combined.shape[3])
        else:
            combined = ch.cat((w, b), 0).T

        cctd.append(combined)

    if conv:
        if include_all:
            return (dims, dim_kernels, middle_dim), cctd
        return (dims, dim_kernels), cctd
    if include_all:
        return (dims, middle_dim), cctd
    return dims, cctd


class PermInvConvModel(nn.Module):
    def __init__(self, dim_channels, dim_kernels,
                 inside_dims=[64, 8], n_classes=2,
                 dropout=0.5, only_latent=False,
                 scale_invariance=False):
        super(PermInvConvModel, self).__init__()
        self.dim_channels = dim_channels
        self.dim_kernels = dim_kernels
        self.only_latent = only_latent
        self.scale_invariance = scale_invariance

        assert len(dim_channels) == len(
            dim_kernels), "Kernel size information missing!"

        self.dropout = dropout
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        # One network per kernel location
        def make_mini(y):
            layers = [
                nn.Dropout(self.dropout),
                nn.Linear(y, inside_dims[0]),
                nn.ReLU(),
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        # For each layer of kernels
        for i, dim in enumerate(self.dim_channels):
            # +1 for bias
            # prev_layer for previous layer
            if i > 0:
                prev_layer = inside_dims[-1] * dim

            # For each pixel in the kernel
            # Concatenated along pixels in kernel
            self.layers.append(
                make_mini(prev_layer + (1 + dim) * dim_kernels[i]))

        self.layers = nn.ModuleList(self.layers)

        # Experimental param: if scale invariance, also store overall scale multiplier
        dim_for_scale_invariance = 1 if self.scale_invariance else 0

        # Final network to combine them all
        # layer representations together
        if not self.only_latent:
            self.rho = nn.Linear(
                inside_dims[-1] * len(self.dim_channels) + dim_for_scale_invariance,
                n_classes)

    def forward(self, params):
        reps = []
        for_prev = None

        if self.scale_invariance:
            # Keep track of multiplier (with respect to smallest nonzero weight) across layers
            # For ease of computation, we will store in log scale
            scale_invariance_multiplier = ch.ones((params[0].shape[0]))
            # Shift to appropriate device
            scale_invariance_multiplier = scale_invariance_multiplier.to(params[0].device)

        for param, layer in zip(params, self.layers):
            # shape: (n_samples, n_pixels_in_kernel, channels_out, channels_in)
            prev_shape = param.shape

            # shape: (n_samples, channels_out, n_pixels_in_kernel, channels_in)
            param = param.transpose(1, 2)

            # shape: (n_samples, channels_out, n_pixels_in_kernel * channels_in)
            param = ch.flatten(param, 2)

            if self.scale_invariance:
                # TODO: Vectorize
                for i in range(param.shape[0]):
                    # Scaling mechanism- pick largest weight, scale weights
                    # such that largest weight becomes 1
                    scale_factor = ch.norm(param[i])
                    scale_invariance_multiplier[i] += ch.log(scale_factor)
                    # Scale parameter matrix (if it's not all zeros)
                    if scale_factor != 0:
                        param[i] /= scale_factor

            if for_prev is None:
                param_eff = param
            else:
                prev_rep = for_prev.repeat(1, param.shape[1], 1)
                param_eff = ch.cat((param, prev_rep), -1)

            # shape: (n_samples * channels_out, channels_in_eff)
            param_eff = param_eff.view(
                param_eff.shape[0] * param_eff.shape[1], -1)

            # shape: (n_samples * channels_out, inside_dims[-1])
            pp = layer(param_eff.reshape(-1, param_eff.shape[-1]))

            # shape: (n_samples, channels_out, inside_dims[-1])
            pp = pp.view(prev_shape[0], prev_shape[2], -1)

            # shape: (n_samples, inside_dims[-1])
            processed = ch.sum(pp, -2)

            # Store previous layer's representation
            # shape: (n_samples, channels_out * inside_dims[-1])
            for_prev = pp.view(pp.shape[0], -1)

            # shape: (n_samples, 1, channels_out * inside_dims[-1])
            for_prev = for_prev.unsqueeze(-2)

            # Store representation for this layer
            reps.append(processed)

        reps = ch.cat(reps, 1)

        # Add invariance multiplier
        if self.scale_invariance:
            scale_invariance_multiplier = ch.unsqueeze(scale_invariance_multiplier, 1)
            reps = ch.cat((reps, scale_invariance_multiplier), 1)

        if self.only_latent:
            return reps

        logits = self.rho(reps)
        return logits


class PermInvModel(nn.Module):
    def __init__(self, dims: List[int], inside_dims: List[int] = [64, 8],
                 n_classes: int = 2, dropout: float = 0.5,
                 only_latent: bool = False):
        super(PermInvModel, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.only_latent = only_latent
        self.final_act_size = inside_dims[-1] * len(dims)
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        def make_mini(y):
            layers = [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        for i, dim in enumerate(self.dims):
            # +1 for bias
            # prev_layer for previous layer
            # input dimension per neuron
            if i > 0:
                prev_layer = inside_dims[-1] * dim
            self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = nn.ModuleList(self.layers)

        if not self.only_latent:
            # Final network to combine them all together
            self.rho = nn.Linear(self.final_act_size, n_classes)

    def forward(self, params) -> ch.Tensor:
        reps = []
        prev_layer_reps = None
        is_batched = len(params[0].shape) > 2

        for param, layer in zip(params, self.layers):

            # Case where data is batched per layer
            if is_batched:
                if prev_layer_reps is None:
                    param_eff = param
                else:
                    prev_layer_reps = prev_layer_reps.repeat(
                        1, param.shape[1], 1)
                    param_eff = ch.cat((param, prev_layer_reps), -1)

                prev_shape = param_eff.shape
                processed = layer(param_eff.view(-1, param_eff.shape[-1]))
                processed = processed.view(
                    prev_shape[0], prev_shape[1], -1)

            else:
                if prev_layer_reps is None:
                    param_eff = param
                else:
                    prev_layer_reps = prev_layer_reps.repeat(param.shape[0], 1)
                    # Include previous layer representation
                    param_eff = ch.cat((param, prev_layer_reps), -1)
                processed = layer(param_eff)

            # Store this layer's representation
            reps.append(ch.sum(processed, -2))

            # Handle per-data/batched-data case together
            if is_batched:
                prev_layer_reps = processed.view(processed.shape[0], -1)
            else:
                prev_layer_reps = processed.view(-1)
            prev_layer_reps = ch.unsqueeze(prev_layer_reps, -2)

        if is_batched:
            reps_c = ch.cat(reps, 1)
        else:
            reps_c = ch.unsqueeze(ch.cat(reps), 0)

        if self.only_latent:
            return reps_c

        logits = self.rho(reps_c)
        return logits


class FullPermInvModel(nn.Module):
    def __init__(self, dims, middle_dim, dim_channels, dim_kernels,
                 inside_dims=[64, 8], n_classes=2, dropout=0.5):
        super(FullPermInvModel, self).__init__()
        self.dim_channels = dim_channels
        self.dim_kernels = dim_kernels
        self.middle_dim = middle_dim
        self.dims = dims
        self.total_layers = len(dim_channels) + len(dims)

        assert len(dim_channels) == len(
            dim_kernels), "Kernel size information missing!"

        self.dropout = dropout
        self.layers = []
        prev_layer = 0

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        # One network per kernel location
        def make_mini(y, add_drop=False):
            layers = []
            if add_drop:
                layers += [nn.Dropout(self.dropout)]
            layers += [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

            return nn.Sequential(*layers)

        # For each layer
        for i in range(self.total_layers):
            is_conv = i < len(self.dim_channels)

            if is_conv:
                dim = self.dim_channels[i]
            else:
                dim = self.dims[i - len(self.dim_channels)]

            # +1 for bias
            # prev_layer for previous layer
            if i > 0:
                prev_layer = inside_dims[-1] * dim

            if is_conv:
                # Concatenated along pixels in kernel
                self.layers.append(
                    make_mini(prev_layer + (1 + dim) * dim_kernels[i], add_drop=True))
            else:
                # FC layer
                if i == len(self.dim_channels):
                    prev_layer = inside_dims[-1] * middle_dim
                self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = nn.ModuleList(self.layers)

        # Final network to combine them all
        # layer representations together
        self.rho = nn.Linear(
            inside_dims[-1] * self.total_layers, n_classes)

    def forward(self, params: List[ch.Tensor]) -> ch.Tensor:
        reps = []
        for_prev = None
        i = 0

        for i, (param, layer) in enumerate(zip(params, self.layers)):
            is_conv = i < len(self.dim_channels)

            if is_conv:
                # Convolutional layer

                # shape: (n_samples, n_pixels_in_kernel, channels_out, channels_in)
                prev_shape = param.shape

                # shape: (n_samples, channels_out, n_pixels_in_kernel, channels_in)
                param = param.transpose(1, 2)

                # shape: (n_samples, channels_out, n_pixels_in_kernel * channels_in)
                param = ch.flatten(param, 2)

            # Concatenate previous layer representation, if available
            if for_prev is None:
                param_eff = param
            else:
                prev_rep = for_prev.repeat(1, param.shape[1], 1)
                param_eff = ch.cat((param, prev_rep), -1)

            if is_conv:
                # Convolutional layer

                # shape: (n_samples * channels_out, channels_in_eff)
                param_eff = param_eff.view(
                    param_eff.shape[0] * param_eff.shape[1], -1)

                # print(param_eff.reshape(-1, param_eff.shape[-1]).shape)

                # shape: (n_samples * channels_out, inside_dims[-1])
                pp = layer(param_eff.reshape(-1, param_eff.shape[-1]))

                # shape: (n_samples, channels_out, inside_dims[-1])
                pp = pp.view(prev_shape[0], prev_shape[2], -1)

            else:
                # FC layer
                prev_shape = param_eff.shape
                pp = layer(param_eff.view(-1, param_eff.shape[-1]))
                pp = pp.view(prev_shape[0], prev_shape[1], -1)

            processed = ch.sum(pp, -2)

            # Store previous layer's representation
            for_prev = pp.view(pp.shape[0], -1)
            for_prev = for_prev.unsqueeze(-2)

            # Store representation for this layer
            reps.append(processed)

        reps = ch.cat(reps, 1)
        logits = self.rho(reps)
        return logits


class CombinedPermInvModel(nn.Module):
    def __init__(self, dims, dim_channels, dim_kernels,
                 inside_dims=[64, 8], n_classes=2, dropout=0.5):
        super(CombinedPermInvModel, self).__init__()
        # Model for convolutional layers
        self.conv_perm = PermInvConvModel(
            dim_channels, dim_kernels, inside_dims,
            n_classes, dropout, only_latent=True)
        # Model for FC layers
        self.fc_perm = PermInvModel(
            dims, inside_dims, n_classes,
            dropout, only_latent=True)

        self.n_conv_layers = len(dim_channels)
        self.n_fc_layers = len(dims)

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        n_layers = self.n_conv_layers + self.n_fc_layers
        self.rho = nn.Linear(inside_dims[-1] * n_layers, n_classes)

    def forward(self, x):
        # First n_conv_layers are for CONV model
        conv_latent = self.conv_perm(x[:self.n_conv_layers])
        # Layers after that are for FC model
        fc_latent = self.fc_perm(x[-self.n_fc_layers:])

        # Concatenate feature representations
        latent = ch.cat((fc_latent, conv_latent), -1)
        logits = self.rho(latent)
        return logits


class CustomBertModel(nn.Module):
    def __init__(self, base_model):
        super(CustomBertModel, self).__init__()
        self.bert = base_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(768, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        outputs = self.bert(**x)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ensure_dir_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


@ch.no_grad()
def acc_fn(x, y):
    return ch.sum((y == (x >= 0)))


def get_outputs(model, X, no_grad=False):

    with ch.set_grad_enabled(not no_grad):
        outputs = model(X)

    return outputs[:, 0]


def prepare_batched_data(X, reduce=False, verbose=True):
    inputs = [[] for _ in range(len(X[0]))]
    iterator = X
    if verbose:
        iterator = tqdm(iterator, desc="Batching data")
    for x in iterator:
        for i, l in enumerate(x):
            inputs[i].append(l)

    inputs = np.array([ch.stack(x, 0) for x in inputs], dtype='object')
    if reduce:
        inputs = [x.view(-1, x.shape[-1]) for x in inputs]
    return inputs


def heuristic(df, condition, ratio,
              cwise_sample,
              class_imbalance=2.0,
              n_tries=1000,
              class_col="label",
              verbose=True):
    vals, pckds = [], []
    iterator = range(n_tries)
    if verbose:
        iterator = tqdm(iterator)
    for _ in iterator:
        pckd_df = filter(df, condition, ratio, verbose=False)

        # Class-balanced sampling
        zero_ids = np.nonzero(pckd_df[class_col].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df[class_col].to_numpy() == 1)[0]

        # Sub-sample data, if requested
        if cwise_sample is not None:
            if class_imbalance >= 1:
                zero_ids = np.random.permutation(
                    zero_ids)[:int(class_imbalance * cwise_sample)]
                one_ids = np.random.permutation(
                    one_ids)[:cwise_sample]
            else:
                zero_ids = np.random.permutation(
                    zero_ids)[:cwise_sample]
                one_ids = np.random.permutation(
                    one_ids)[:int(1 / class_imbalance * cwise_sample)]

        # Combine them together
        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
        pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
        if verbose:
            iterator.set_description(
                "%.4f" % (ratio + np.min([np.abs(zz-ratio) for zz in vals])))

    vals = np.abs(np.array(vals) - ratio)
    # Pick the one closest to desired ratio
    picked_df = pckds[np.argmin(vals)]
    return picked_df.reset_index(drop=True)


def train_epoch(train_loader, model, criterion, optimizer, epoch, verbose=True, adv_train=False, expect_extra=True):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    if verbose:
        iterator = tqdm(train_loader)
    for data in iterator:
        if expect_extra:
            images, labels, _ = data
        else:
            images, labels = data
        images, labels = images.cuda(), labels.cuda()
        N = images.size(0)

        if adv_train is False:
            # Clear accumulated gradients
            optimizer.zero_grad()
            outputs = model(images)[:, 0]
        else:
            # Adversarial inputs
            adv_x = projected_gradient_descent(
                model, images, eps=adv_train['eps'],
                eps_iter=adv_train['eps_iter'],
                nb_iter=adv_train['nb_iter'],
                norm=adv_train['norm'],
                clip_min=adv_train['clip_min'],
                clip_max=adv_train['clip_max'],
                random_restarts=adv_train['random_restarts'],
                binary_sigmoid=True)
            # Important to zero grad after above call, else model gradients
            # get accumulated over attack too
            optimizer.zero_grad()
            outputs = model(adv_x)[:, 0]

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        prediction = (outputs >= 0)
        train_acc.update(prediction.eq(
            labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        if verbose:
            iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (
                epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion, verbose=True, adv_train=False, expect_extra=True):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    adv_val_loss = AverageMeter()
    adv_val_acc = AverageMeter()

    with ch.set_grad_enabled(adv_train is not False):
        for data in val_loader:
            if expect_extra:
                images, labels, _ = data
            else:
                images, labels = data
            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)

            outputs = model(images)[:, 0]
            prediction = (outputs >= 0)

            if adv_train is not False:
                adv_x = projected_gradient_descent(
                    model, images, eps=adv_train['eps'],
                    eps_iter=adv_train['eps_iter'],
                    nb_iter=adv_train['nb_iter'],
                    norm=adv_train['norm'],
                    clip_min=adv_train['clip_min'],
                    clip_max=adv_train['clip_max'],
                    random_restarts=adv_train['random_restarts'],
                    binary_sigmoid=True)
                outputs_adv = model(adv_x)[:, 0]
                prediction_adv = (outputs_adv >= 0)

                adv_val_acc.update(prediction_adv.eq(
                    labels.view_as(prediction_adv)).sum().item()/N)

                adv_val_loss.update(
                    criterion(outputs_adv, labels.float()).item())

            val_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels.float()).item())

    if verbose:
        if adv_train is False:
            print('[Validation], Loss: %.5f, Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg))
        else:
            print('[Validation], Loss: %.5f, Accuracy: %.4f | Adv-Loss: %.5f, Adv-Accuracy: %.4f' %
                  (val_loss.avg, val_acc.avg,
                   adv_val_loss.avg, adv_val_acc.avg))
        print()

    if adv_train is False:
        return val_loss.avg, val_acc.avg
    return (val_loss.avg, adv_val_loss.avg), (val_acc.avg, adv_val_acc.avg)


def train(model, loaders, lr=1e-3, epoch_num=10,
          weight_decay=0, verbose=True, get_best=False,
          adv_train=False, expect_extra=True):
    # Get data loaders
    train_loader, val_loader = loaders

    # Define optimizer, loss function
    optimizer = ch.optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss().cuda()

    iterator = range(1, epoch_num+1)
    if not verbose:
        iterator = tqdm(iterator)

    best_model, best_loss = None, np.inf
    for epoch in iterator:
        _, tacc = train_epoch(train_loader, model,
                              criterion, optimizer, epoch,
                              verbose=verbose, adv_train=adv_train,
                              expect_extra=expect_extra)

        vloss, vacc = validate_epoch(
            val_loader, model, criterion, verbose=verbose,
            adv_train=adv_train,
            expect_extra=expect_extra)
        if not verbose:
            if adv_train is False:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))
            else:
                iterator.set_description(
                    "train_acc: %.2f | val_acc: %.2f | adv_val_acc: %.2f" % (tacc, vacc[0], vacc[1]))

        vloss_compare = vloss
        if adv_train is not False:
            vloss_compare = vloss[0]

        if get_best and vloss_compare < best_loss:
            best_loss = vloss_compare
            best_model = deepcopy(model)

    if get_best:
        return best_model, (vloss, vacc)
    return vloss, vacc


def compute_metrics(dataset_true, dataset_pred,
                    unprivileged_groups, privileged_groups):
    """ Compute the key metrics """
    from aif360.metrics import ClassificationMetric
    classified_metric_pred = ClassificationMetric(
        dataset_true,
        dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    metrics = OrderedDict()
    metrics["Balanced accuracy"] = 0.5 * \
        (classified_metric_pred.true_positive_rate() +
         classified_metric_pred.true_negative_rate())
    metrics["Statistical parity difference"] = \
        classified_metric_pred.statistical_parity_difference()
    metrics["Disparate impact"] = classified_metric_pred.disparate_impact()
    metrics["Average odds difference"] = \
        classified_metric_pred.average_odds_difference()
    metrics["Equal opportunity difference"] = \
        classified_metric_pred.equal_opportunity_difference()
    metrics["Theil index"] = classified_metric_pred.theil_index()
    metrics["False discovery rate difference"] = \
        classified_metric_pred.false_discovery_rate_difference()
    metrics["False discovery rate ratio"] = \
        classified_metric_pred.false_discovery_rate_ratio()
    metrics["False omission rate difference"] = \
        classified_metric_pred.false_omission_rate_difference()
    metrics["False omission rate ratio"] = \
        classified_metric_pred.false_omission_rate_ratio()
    metrics["False negative rate difference"] = \
        classified_metric_pred.false_negative_rate_difference()
    metrics["False negative rate ratio"] = \
        classified_metric_pred.false_negative_rate_ratio()
    metrics["False positive rate difference"] = \
        classified_metric_pred.false_positive_rate_difference()
    metrics["False positive rate ratio"] = \
        classified_metric_pred.false_positive_rate_ratio()

    return metrics


@ch.no_grad()
def test_meta(model, loss_fn, X, Y, batch_size, accuracy,
              binary: bool = True, regression=False, gpu: bool = False,
              combined: bool = False, X_acts=None,
              element_wise=False, get_preds: bool = False):
    model.eval()
    use_acts = (X_acts is not None)
    # Activations must be provided if not combined
    assert (not use_acts) or (
        combined), "Activations must be provided if not combined"

    # Batch data to fit on GPU
    num_samples, running_acc = 0, 0
    loss = [] if element_wise else 0
    all_outputs = []

    i = 0

    if combined:
        n_samples = len(X[0])
    else:
        n_samples = len(X)

    while i < n_samples:
        # Model features stored as list of objects
        outputs = []
        if not combined:
            for param in X[i:i+batch_size]:
                # Shift to GPU, if requested
                if gpu:
                    param = [a.cuda() for a in param]

                if binary or regression:
                    outputs.append(model(param)[:, 0])
                else:
                    outputs.append(model(param))
        # Model features stored as normal list
        else:
            param_batch = [x[i:i+batch_size] for x in X]
            if use_acts:
                acts_batch = X_acts[i:i+batch_size]
            if gpu:
                param_batch = [a.cuda() for a in param_batch]

            if binary or regression:
                if use_acts:
                    outputs.append(model(param_batch, acts_batch)[:, 0])
                else:
                    outputs.append(model(param_batch)[:, 0])
            else:
                if use_acts:
                    outputs.append(model(param_batch, acts_batch))
                else:
                    outputs.append(model(param_batch))

        outputs = ch.cat(outputs, 0)
        if get_preds:
            all_outputs.append(outputs.cpu().detach().numpy())

        num_samples += outputs.shape[0]
        if element_wise:
            loss.append(loss_fn(outputs, Y[i:i+batch_size]).detach().cpu())
        else:
            loss += loss_fn(outputs,
                            Y[i:i+batch_size]).item() * num_samples
        if not regression:
            running_acc += accuracy(outputs, Y[i:i+batch_size]).item()

        # Next batch
        i += batch_size

    if element_wise:
        loss = ch.cat(loss, 0)
    else:
        loss /= num_samples
    
    if get_preds:
        all_outputs = np.concatenate(all_outputs, axis=0)
        return 100 * running_acc / num_samples, loss, all_outputs

    return 100 * running_acc / num_samples, loss


# Function to train meta-classifier
def train_meta_model(model, train_data, test_data,
                     epochs, lr, eval_every=5,
                     binary=True, regression=False,
                     val_data=None, batch_size=1000,
                     gpu=False, combined=False,
                     shuffle=True, train_acts=None,
                     test_acts=None, val_acts=None):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    # Make sure both weights and activations available if val requested
    assert (val_data is not None or val_acts is None), "Weights or activations for validation data must be provided"

    use_acts = (train_acts is not None)
    # Activations must be provided if not combined
    assert (not use_acts) or (
        combined), "Activations must be provided if not combined"

    if regression:
        loss_fn = nn.MSELoss()
    else:
        if binary:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

    params, y = train_data
    params_test, y_test = test_data

    # Shift to GPU, if requested
    if gpu:
        y = y.cuda()
        y_test = y_test.cuda()

    # Reserve some data for validation, use this to pick best model
    if val_data is not None:
        params_val, y_val = val_data
        best_loss, best_model = np.inf, None
        if gpu:
            y_val = y_val.cuda()

    def acc_fn(x, y):
        if binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))

    iterator = tqdm(range(epochs))
    for e in iterator:
        # Training
        model.train()

        # Shuffle train data
        if shuffle:
            rp_tr = np.random.permutation(y.shape[0])
            if not combined:
                params, y = params[rp_tr], y[rp_tr]
            else:
                y = y[rp_tr]
                params = [x[rp_tr] for x in params]
            if use_acts:
                train_acts = train_acts[rp_tr]

        # Batch data to fit on GPU
        running_acc, loss, num_samples = 0, 0, 0
        i = 0

        if combined:
            n_samples = len(params[0])
        else:
            n_samples = len(params)

        while i < n_samples:

            # Model features stored as list of objects
            outputs = []
            if not combined:
                for param in params[i:i+batch_size]:
                    # Shift to GPU, if requested
                    if gpu:
                        param = [a.cuda() for a in param]

                    if binary or regression:
                        outputs.append(model(param)[:, 0])
                    else:
                        outputs.append(model(param))
            # Model features stored as normal list
            else:
                param_batch = [x[i:i+batch_size] for x in params]
                if use_acts:
                    acts_batch = train_acts[i:i+batch_size]
                if gpu:
                    param_batch = [a.cuda() for a in param_batch]

                if binary or regression:
                    if use_acts:
                        outputs.append(model(param_batch, acts_batch)[:, 0])
                    else:
                        outputs.append(model(param_batch)[:, 0])
                else:
                    if use_acts:
                        outputs.append(model(param_batch, acts_batch))
                    else:
                        outputs.append(model(param_batch))

            outputs = ch.cat(outputs, 0)

            # Clear accumulated gradients
            optimizer.zero_grad()

            # Compute loss
            loss = loss_fn(outputs, y[i:i+batch_size])

            # Compute gradients
            loss.backward()

            # Take gradient step
            optimizer.step()

            # Keep track of total loss, samples processed so far
            num_samples += outputs.shape[0]
            loss += loss.item() * outputs.shape[0]

            print_acc = ""
            if not regression:
                running_acc += acc_fn(outputs, y[i:i+batch_size])
                print_acc = ", Accuracy: %.2f" % (
                    100 * running_acc / num_samples)

            iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (
                e+1, loss / num_samples, print_acc))

            # Next batch
            i += batch_size

        # Evaluate on validation data, if present
        if val_data is not None:
            v_acc, val_loss = test_meta(model, loss_fn, params_val,
                                        y_val, batch_size, acc_fn,
                                        binary=binary, regression=regression,
                                        gpu=gpu, combined=combined,
                                        X_acts=val_acts)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(model)

        # Evaluate on test data now
        if (e+1) % eval_every == 0:
            if val_data is not None:
                print_acc = ""
                if not regression:
                    print_acc = ", Accuracy: %.2f" % (v_acc)

                log("[Validation] Loss: %.5f%s" % (val_loss, print_acc))

            # Also log test-data metrics
            t_acc, t_loss = test_meta(model, loss_fn, params_test,
                                      y_test, batch_size, acc_fn,
                                      binary=binary, regression=regression,
                                      gpu=gpu, combined=combined,
                                      X_acts=test_acts)
            print_acc = ""
            if not regression:
                print_acc = ", Accuracy: %.2f" % (t_acc)

            log("[Test] Loss: %.5f%s" % (t_loss, print_acc))
            print()

    # Pick best model (according to validation), if requested
    # And compute test accuracy on this model
    if val_data is not None:
        t_acc, t_loss = test_meta(best_model, loss_fn, params_test,
                                  y_test, batch_size, acc_fn,
                                  binary=binary, regression=regression,
                                  gpu=gpu, combined=combined,
                                  X_acts=test_acts)
        model = deepcopy(best_model)

    # Make sure model is in evaluation mode
    model.eval()

    if regression:
        return model, t_loss
    return model, t_acc


def get_z_value(metric_1, metric_2):
    assert len(metric_1) == len(metric_2), "Unequal sample sets!"
    n_samples = 2 * len(metric_1)
    m1, v1 = np.mean(metric_1), np.var(metric_1)
    m2, v2 = np.mean(metric_2), np.var(metric_2)

    mean_new = np.abs(m1 - m2)
    var_new = (v1 + v2) / n_samples

    Z = mean_new / np.sqrt(var_new)
    return Z


def get_threshold_acc(X, Y, threshold, rule=None):
    # Rule-1: everything above threshold is 1 class
    acc_1 = np.mean((X >= threshold) == Y)
    # Rule-2: everything below threshold is 1 class
    acc_2 = np.mean((X <= threshold) == Y)

    # If rule is specified, use that
    if rule == 1:
        return acc_1
    elif rule == 2:
        return acc_2

    # Otherwise, find and use the one that gives the best acc

    if acc_1 >= acc_2:
        return acc_1, 1
    return acc_2, 2


def find_threshold_acc(accs_1, accs_2, granularity=0.1):
    lower = min(np.min(accs_1), np.min(accs_2))
    upper = max(np.max(accs_1), np.max(accs_2))
    combined = np.concatenate((accs_1, accs_2))
    # Want to predict first set as 0s, second set as 1s
    classes = np.concatenate((np.zeros_like(accs_1), np.ones_like(accs_2)))
    best_acc = 0.0
    best_threshold = 0
    best_rule = None
    while lower <= upper:
        best_of_two, rule = get_threshold_acc(combined, classes, lower)
        if best_of_two > best_acc:
            best_threshold = lower
            best_acc = best_of_two
            best_rule = rule

        lower += granularity

    return best_acc, best_threshold, best_rule


def get_threshold_pred(X, Y, threshold, rule,
                       get_pred: bool = False,
                       confidence: bool = False):
    if X.shape[1] != Y.shape[0]:
        raise ValueError('Dimension mismatch between X and Y: %d and %d should match' % (X.shape[1], Y.shape[0]))
    if X.shape[0] != threshold.shape[0]:
        raise ValueError('Dimension mismatch between X and threshold: %d and %d should match' % (X.shape[0], threshold.shape[0]))
    res = []
    for i in range(X.shape[1]):
        prob = np.average((X[:, i] <= threshold) == rule)
        if confidence:
            res.append(prob)
        else:
            res.append(prob >= 0.5)
    res = np.array(res)
    if confidence:
        acc = np.mean((res >= 0.5) == Y)
    else:    
        acc = np.mean(res == Y)
    if get_pred:
        return res, acc
    return acc


def find_threshold_pred(pred_1, pred_2, granularity=0.005):
    if pred_1.shape[0] != pred_2.shape[0]:
        raise ValueError('dimension mismatch')
    thres, rules = [], []
    g = granularity
    for i in tqdm(range(pred_1.shape[0])):
        _, t, r = find_threshold_acc(pred_1[i], pred_2[i], g)
        while r is None:
            g = g/10
            _, t, r = find_threshold_acc(pred_1[i], pred_2[i], g)
        thres.append(t)
        rules.append(r-1)
    thres = np.array(thres)
    rules = np.array(rules)
    acc = get_threshold_pred(np.concatenate((pred_1, pred_2), axis=1), np.concatenate(
        (np.zeros(pred_1.shape[1]), np.ones(pred_2.shape[1]))), thres, rules)
    return acc, thres, rules


# Fix for repeated random augmentation issue
# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# identity function
class basic(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # return input.clamp(min=0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# fake relu function
class fakerelu(ch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # return input.clamp(min=0)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


# Fake-relu module wrapper
class FakeReluWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(FakeReluWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return fakerelu.apply(input)


# identity function module wrapper
class BasicWrapper(nn.Module):
    def __init__(self, inplace: bool = False):
        super(BasicWrapper, self).__init__()
        self.inplace = inplace

    def forward(self, input: ch.Tensor):
        return fakerelu.apply(input)


def get_n_effective(acc, r0, r1):
    if max(r0, r1) == 0:
        return np.inf

    if r0 == r1:
        return 0

    if acc == 1 or np.abs(r0 - r1) == 1:
        return np.inf

    num = np.log(1 - ((2 * acc - 1) ** 2))
    ratio_0 = min(r0, r1) / max(r0, r1)
    ratio_1 = (1 - max(r0, r1)) / (1 - min(r0, r1))
    den = np.log(max(ratio_0, ratio_1))
    return num / den


def bound(x, y, n):
    if max(x, y) == 0:
        return 0.5

    def bound_1():
        # Handle 0/0 form gracefully
        # if x == 0 and y == 0:
        #     return 0
        ratio = min(x, y) / max(x, y)
        return np.sqrt(1 - (ratio ** n))

    def bound_2():
        ratio = (1 - max(x, y)) / (1 - min(x, y))
        return np.sqrt(1 - (ratio ** n))

    l1 = bound_1()
    l2 = bound_2()
    pick = min(l1, l2) / 2
    return 0.5 + pick


def extract_adv_params(
        eps, eps_iter, nb_iter, norm,
        random_restarts, clip_min, clip_max):
    adv_params = {}
    adv_params["eps"] = eps
    adv_params["eps_iter"] = eps_iter
    adv_params["nb_iter"] = nb_iter
    adv_params["norm"] = norm
    adv_params["clip_min"] = clip_min
    adv_params["clip_max"] = clip_max
    adv_params["random_restarts"] = random_restarts

    return adv_params


class ActivationMetaClassifier(nn.Module):
    def __init__(self, n_samples, dims, reduction_dims,
                 inside_dims=[64, 16],
                 n_classes=2, dropout=0.2):
        super(ActivationMetaClassifier, self).__init__()
        self.n_samples = n_samples
        self.dims = dims
        self.reduction_dims = reduction_dims
        self.dropout = dropout
        self.layers = []

        assert len(dims) == len(reduction_dims), "dims and reduction_dims must be same length"

        # If binary, need only one output
        if n_classes == 2:
            n_classes = 1

        def make_mini(y, last_dim):
            layers = [
                nn.Linear(y, inside_dims[0]),
                nn.ReLU()
            ]
            for i in range(1, len(inside_dims)):
                layers.append(nn.Linear(inside_dims[i-1], inside_dims[i]))
                layers.append(nn.ReLU())
            layers.append(
                nn.Linear(inside_dims[len(inside_dims)-1], last_dim))
            layers.append(nn.ReLU())

            return nn.Sequential(*layers)

        # Reducing each activation into smaller representations
        for ld, dim in zip(self.reduction_dims, self.dims):
            self.layers.append(make_mini(dim, ld))

        self.layers = nn.ModuleList(self.layers)

        # Final layer to concatenate all representations across examples
        self.rho = nn.Sequential(
            nn.Linear(sum(self.reduction_dims) * self.n_samples, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, n_classes),
        )

    def forward(self, params):
        reps = []
        for param, layer in zip(params, self.layers):
            processed = layer(param.view(-1, param.shape[2]))
            # Store this layer's representation
            reps.append(processed)

        reps_c = ch.cat(reps, 1)
        reps_c = reps_c.view(-1, self.n_samples * sum(self.reduction_dims))

        logits = self.rho(reps_c)
        return logits


def op_solution(x, y):
    """
        Return the optimal rotation to apply to x so that it aligns with y.
    """
    u, s, vh = np.linalg.svd(x.T @ y)
    optimal_x_to_y = u @ vh
    return optimal_x_to_y


def align_all_features(reference_point, features):
    """
        Perform layer-wise alignment of given features, using
        reference point. Return aligned features.
    """
    aligned_features = []
    for feature in tqdm(features, desc="Aligning features"):
        inside_feature = []
        for (ref_i, x_i) in zip(reference_point, feature):
            aligned_feature = x_i @ op_solution(x_i, ref_i)
            inside_feature.append(aligned_feature)
        aligned_features.append(inside_feature)
    return np.array(aligned_features, dtype=object)


def wrap_data_for_act_meta_clf(models_neg, models_pos,
                               data, get_activation_fn,
                               detach: bool = True):
    """
        Given models from two different distributions, get their
        activations on given data and activation-extraction function, and
        combine them into data-label format for a meta-classifier.
    """
    neg_w, neg_labels, _ = get_activation_fn(
        models_pos, data, 1, detach, verbose=False)
    pos_w, pos_labels, _ = get_activation_fn(
        models_neg, data, 0, detach, verbose=False)
    pp_x = prepare_batched_data(pos_w, verbose=False)
    np_x = prepare_batched_data(neg_w, verbose=False)
    X = [ch.cat((x, y), 0) for x, y in zip(pp_x, np_x)]
    Y = ch.cat((pos_labels, neg_labels))
    return X, Y.cuda().float()


def coordinate_descent(models_train, models_val,
                       models_test, dims, reduction_dims,
                       get_activation_fn,
                       n_samples, meta_train_args,
                       gen_optimal_fn, seed_data,
                       n_times: int = 10,
                       restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        dims: Dimensions of feature activations.
        reduction_dims: Dimensions for meta-classifier internal models.
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = ActivationMetaClassifier(
                    n_samples, dims,
                    reduction_dims=reduction_dims)
    metamodel = metamodel.cuda()

    best_clf, best_tacc = None, 0
    val_data = None
    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        X_tr, Y_tr = wrap_data_for_act_meta_clf(
            models_train[0], models_train[1], seed_data, get_activation_fn)
        X_te, Y_te = wrap_data_for_act_meta_clf(
            models_test[0], models_test[1], seed_data, get_activation_fn)
        if models_val is not None:
            val_data = wrap_data_for_act_meta_clf(
                models_val[0], models_val[1], seed_data, get_activation_fn)

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = ActivationMetaClassifier(
                n_samples, dims,
                reduction_dims=reduction_dims)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        metamodel.train()
        clf, tacc = train_meta_model(
                    metamodel,
                    (X_tr, Y_tr), (X_te, Y_te),
                    epochs=meta_train_args['epochs'],
                    binary=True, lr=1e-3,
                    regression=False,
                    batch_size=meta_train_args['batch_size'],
                    val_data=val_data, combined=True,
                    eval_every=10, gpu=True)
        all_accs.append(tacc)

        # Keep track of best model and latest model
        if tacc > best_tacc:
            best_tacc = tacc
            best_clf = clf

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data, get_activation_fn)

    # Return best and latest models
    return (best_tacc, best_clf), (tacc, clf), all_accs


def check_if_inside_cluster():
    """
        Check if current code is being run inside a cluster.
    """
    if environ.get('ISRIVANNA') == "1":
        return True
    return False


class AffinityMetaClassifier(nn.Module):
    def __init__(self, num_dim: int, numlayers: int,
                 num_final: int = 16, only_latent: bool = False):
        super(AffinityMetaClassifier, self).__init__()
        self.num_dim = num_dim
        self.numlayers = numlayers
        self.only_latent = only_latent
        self.num_final = num_final
        self.final_act_size = num_final * self.numlayers
        self.models = []

        def make_small_model():
            return nn.Sequential(
                nn.Linear(self.num_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_final),
            )
        for _ in range(numlayers):
            self.models.append(make_small_model())
        self.models = nn.ModuleList(self.models)
        if not self.only_latent:
            self.final_layer = nn.Linear(self.num_final * self.numlayers, 1)

    def forward(self, x) -> ch.Tensor:
        # Get intermediate activations for each layer
        # Aggreage them to get a single feature vector
        all_acts = []
        for i, model in enumerate(self.models):
            all_acts.append(model(x[:, i]))
        all_accs = ch.cat(all_acts, 1)
        # Return pre-logit activations if requested
        if self.only_latent:
            return all_accs
        return self.final_layer(all_accs)


def make_affinity_feature(model, data, use_logit=False, detach=True, verbose=True):
    """
         Construct affinity matrix per layer based on affinity scores
         for a given model. Model them in a way that does not
         require graph-based models.
    """
    # Build affinity graph for given model and data
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Start with getting layer-wise model features
    model_features = model(data, get_all=True, detach_before_return=detach)
    layerwise_features = []
    for i, feature in enumerate(model_features):
        # Old (before 2/4)
        # Skip logits if asked not to use (default)
        # if not use_logit and i == (len(model_features) - 1):
            # break
        scores = []
        # Pair-wise iteration of all data
        for i in range(len(data)-1):
            others = feature[i+1:]
            scores += cos(ch.unsqueeze(feature[i], 0), others)
        layerwise_features.append(ch.stack(scores, 0))

    # New (2/4)
    # If asked to use logits, convert them to probability scores
    # And then consider them as-it-is (instead of pair-wise comparison)
    if use_logit:
        logits = model_features[-1]
        probs = ch.sigmoid(logits)
        layerwise_features.append(probs)

    concatenated_features = ch.stack(layerwise_features, 0)
    return concatenated_features


def make_affinity_features(models, data, use_logit=False, detach=True, verbose=True):
    all_features = []
    iterator = models
    if verbose:
        iterator = tqdm(iterator, desc="Building affinity matrix")
    for model in iterator:
        all_features.append(
            make_affinity_feature(
                model, data, use_logit=use_logit, detach=detach, verbose=verbose)
        )
    return ch.stack(all_features, 0)


def coordinate_descent_new(models_train, models_val,
                           num_features, num_layers,
                           get_features,
                           meta_train_args,
                           gen_optimal_fn, seed_data,
                           n_times: int = 10,
                           restart_meta: bool = False):
    """
    Coordinate descent- optimize to find good data points, followed by
    training of meta-classifier model.
    Parameters:
        models_train: Tuple of (pos, neg) models to train.
        models_test: Tuple of (pos, neg) models to test.
        num_layers: Number of layers of models used for activations
        get_features: Function that takes (models, data) as input and returns features
        gen_optimal_fn: Function that generates optimal data points.
        seed_data: Seed data to get activations for.
        n_times: Number of times to run gradient descent.
        meta_train_args: Argument dict for meta-classifier training
    """

    # Define meta-classifier model
    metamodel = AffinityMetaClassifier(num_features, num_layers)
    metamodel = metamodel.cuda()

    all_accs = []
    for _ in range(n_times):
        # Get activations for data
        train_loader = get_features(
            models_train[0], models_train[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])
        val_loader = get_features(
            models_val[0], models_val[1],
            seed_data, meta_train_args['batch_size'],
            use_logit=meta_train_args['use_logit'])

        # Re-init meta-classifier if requested
        if restart_meta:
            metamodel = AffinityMetaClassifier(num_features, num_layers)
            metamodel = metamodel.cuda()

        # Train meta-classifier for a few epochs
        # Make sure meta-classifier is in train mode
        _, val_acc = train(metamodel, (train_loader, val_loader),
                           epoch_num=meta_train_args['epochs'],
                           expect_extra=False,
                           verbose=False)
        all_accs.append(val_acc)

        # Generate new data starting from previous data
        seed_data = gen_optimal_fn(metamodel,
                                   models_train[0], models_train[1],
                                   seed_data)

    # Return all accuracies
    return all_accs


class WeightAndActMeta(nn.Module):
    """
        Combined meta-classifier that uses model weights as well as activation
        trends for property prediction.
    """
    def __init__(self, dims: List[int], num_dims: int, num_layers: int):
        super(WeightAndActMeta, self).__init__()
        self.dims = dims
        self.num_dims = num_dims
        self.num_layers = num_layers
        self.act_clf = AffinityMetaClassifier(
            num_dims, num_layers, only_latent=True)
        self.weights_clf = PermInvModel(dims, only_latent=True)
        self.final_act_size = self.act_clf.final_act_size + \
            self.weights_clf.final_act_size
        self.combination_layer = nn.Linear(self.final_act_size, 1)

    def forward(self, w, x) -> ch.Tensor:
        # Output for weights
        weights = self.weights_clf(w)
        # Output for activations
        act = self.act_clf(x)
        # Combine them
        all_acts = ch.cat([act, weights], 1)
        return self.combination_layer(all_acts)


def get_meta_preds(model, X, batch_size, on_gpu=True):
    """
        Get predictions for meta-classifier.
        Parameters:
            model: Model to get predictions for.
            X: Data to get predictions for.
            batch_size: Batch size to use.
            on_gpu: Whether to use GPU.
    """
    # Get predictions for model
    preds = []
    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        if on_gpu:
            x_batch = [x.cuda() for x in x_batch]
        batch_preds = model(x_batch)
        preds.append(batch_preds.detach())
    return ch.cat(preds, 0)


def order_points(p1s, p2s):
    """
        Estimate utility of individual points, done by taking
        absolute difference in their predictions.
    """
    abs_dif = np.absolute(np.sum(p1s, axis=0) - np.sum(p2s, axis=0))
    inds = np.argsort(abs_dif)
    return inds


def _perpoint_threshold_on_ratio(preds_1, preds_2, classes, threshold, rule):
    """
        Run perpoint threshold test (confidence)
        for a given "quartile" ratio
    """
    # Combine predictions into one vector
    combined = np.concatenate((preds_1, preds_2), axis=1)

    # Compute accuracy for given predictions, thresholds, and rules
    preds, acc = get_threshold_pred(
        combined, classes, threshold, rule, get_pred=True,
        confidence=True)

    return 100 * acc, preds


def perpoint_threshold_test_per_dist(preds_adv: List, preds_victim: List,
                                     ratios: List = [1.],
                                     granularity: float = 0.005):
    """
        Compute thresholds (based on probabilities) for each given datapoint,
        search for thresholds using given adv model's predictions.
        Compute accuracy and predictions using given data and predictions
        on victim model's predictions.
        Try this out with different values of "quartiles", where points
        are ranked according to some utility estimate.
    """
    # Predictions by adversary's models
    p1, p2 = preds_adv
    # Predictions by victim's models
    pv1, pv2 = preds_victim

    # Optimal order of point
    order = order_points(p1, p2)

    # Order points according to computed utility
    p1 = np.transpose(p1)[order][::-1]
    p2 = np.transpose(p2)[order][::-1]
    pv1 = np.transpose(pv1)[order][::-1]
    pv2 = np.transpose(pv2)[order][::-1]

    # Get thresholds for all points
    _, thres, rs = find_threshold_pred(p1, p2, granularity=granularity)

    # Ground truth
    classes_adv = np.concatenate(
        (np.zeros(p1.shape[1]), np.ones(p2.shape[1])))
    classes_victim = np.concatenate(
        (np.zeros(pv1.shape[1]), np.ones(pv2.shape[1])))

    adv_accs, victim_accs, victim_preds, adv_preds = [], [], [], []
    for ratio in ratios:
        # Get first <ratio> percentile of points
        leng = int(ratio * p1.shape[0])
        p1_use, p2_use = p1[:leng], p2[:leng]
        pv1_use, pv2_use = pv1[:leng], pv2[:leng]
        thres_use, rs_use = thres[:leng], rs[:leng]

        # Compute accuracy for given data size on adversary's models
        adv_acc, adv_pred = _perpoint_threshold_on_ratio(
            p1_use, p2_use, classes_adv, thres_use, rs_use)
        adv_accs.append(adv_acc)
        # Compute accuracy for given data size on victim's models
        victim_acc, victim_pred = _perpoint_threshold_on_ratio(
            pv1_use, pv2_use, classes_victim, thres_use, rs_use)
        victim_accs.append(victim_acc)
        # Keep track of predictions on victim's models
        victim_preds.append(victim_pred)
        adv_preds.append(adv_pred)

    adv_accs = np.array(adv_accs)
    victim_accs = np.array(victim_accs)
    victim_preds = np.array(victim_preds, dtype=object)
    adv_preds = np.array(adv_preds, dtype=object)
    return adv_accs, adv_preds, victim_accs, victim_preds


def perpoint_threshold_test(preds_adv: List, preds_victim: List,
                            ratios: List = [1.],
                            granularity: float = 0.005):
    """
        Take predictions from both distributions and run attacks.
        Pick the one that works best on adversary's models
    """
    # Get data for first distribution
    adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1 = perpoint_threshold_test_per_dist(
        preds_adv[0], preds_victim[0], ratios, granularity)
    # Get data for second distribution
    adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2 = perpoint_threshold_test_per_dist(
        preds_adv[1], preds_victim[1], ratios, granularity)

    # Get best adv accuracies for both distributions and compare
    which_dist = 0
    if np.max(adv_accs_1) > np.max(adv_accs_2):
        adv_accs_use, adv_preds_use, victim_accs_use, victim_preds_use = adv_accs_1, adv_preds_1, victim_accs_1, victim_preds_1
    else:
        adv_accs_use, adv_preds_use, victim_accs_use, victim_preds_use = adv_accs_2, adv_preds_2, victim_accs_2, victim_preds_2
        which_dist = 1

    # Out of the best distribution, pick best ratio according to accuracy on adversary's models
    ind = np.argmax(adv_accs_use)
    victim_acc_use = victim_accs_use[ind]
    victim_pred_use = victim_preds_use[ind]
    adv_acc_use = adv_accs_use[ind]
    adv_pred_use = adv_preds_use[ind]

    return (victim_acc_use, victim_pred_use), (adv_acc_use, adv_pred_use), (which_dist, ind)


def get_ratio_info_for_reg_meta(metamodel, X, Y, num_per_dist, batch_size, combined: bool = True):
    """
        Get MSE and actual predictions for each
        ratio given in Y, using a trained metamodel.
        Returnse MSE per ratio, actual predictions per ratio, and
        predictions for each ratio a v/s be using regression
        meta-classifier for binary classification.
    """
    # Evaluate
    metamodel = metamodel.cuda()
    loss_fn = ch.nn.MSELoss(reduction='none')
    _, losses, preds = test_meta(
        metamodel, loss_fn, X, Y.cuda(),
        batch_size, None,
        binary=True, regression=True, gpu=True,
        combined=combined, element_wise=True,
        get_preds=True)
    y_np = Y.numpy()
    losses = losses.numpy()
    # Get all unique ratios (sorted) in GT, and their average losses from model
    ratios = np.unique(y_np)
    losses_dict = {}
    ratio_wise_preds = {}
    for ratio in ratios:
        losses_dict[ratio] = np.mean(losses[y_np == ratio])
        ratio_wise_preds[ratio] = preds[y_np == ratio]
    # Conctruct a matrix where every (i, j) entry is the accuracy
    # for ratio[i] v/s ratio [j], where whichever ratio is closer to the
    # ratios is considered the "correct" one
    # Assume equal number of models per ratio, stored in order of
    # ratios
    acc_mat = np.zeros((len(ratios), len(ratios)))
    for i in range(acc_mat.shape[0]):
        for j in range(i + 1, acc_mat.shape[0]):
            # Get relevant GT for ratios[i] (0) v/s ratios[j] (1)
            gt_z = (y_np[num_per_dist * i:num_per_dist * (i + 1)]
                    == float(ratios[j]))
            gt_o = (y_np[num_per_dist * j:num_per_dist * (j + 1)]
                    == float(ratios[j]))
            # Get relevant preds
            pred_z = preds[num_per_dist * i:num_per_dist * (i + 1)]
            pred_o = preds[num_per_dist * j:num_per_dist * (j + 1)]
            pred_z = (pred_z >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            pred_o = (pred_o >= (0.5 * (float(ratios[i]) + float(ratios[j]))))
            # Compute accuracies and store
            acc = np.concatenate((gt_z, gt_o), 0) == np.concatenate(
                (pred_z, pred_o), 0)
            acc_mat[i, j] = np.mean(acc)

    return losses_dict, acc_mat, ratio_wise_preds
