from html.entities import name2codepoint
import torch as ch
import numpy as np
from collections import OrderedDict
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR
from robustness.tools import folder
from robustness.tools.misc import log_statement
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
        # print(datapath, "wtf?!")
        # exit(0)
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
                      conv=False, include_all=False):
    dims, dim_kernels, weights, biases = [], [], [], []
    i, j = 0, 0

    # Sort and store desired layers, if specified
    custom_layers = sorted(custom_layers) if custom_layers is not None else None

    for name, param in m.named_parameters():
        if "weight" in name:
            param_data = param.data.detach().cpu()
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
    def __init__(self, dim_channels, dim_kernels, inside_dims=[64, 8], n_classes=2, dropout=0.5, only_latent=False):
        super(PermInvConvModel, self).__init__()
        self.dim_channels = dim_channels
        self.dim_kernels = dim_kernels
        self.only_latent = only_latent

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

        # Final network to combine them all
        # layer representations together
        if not self.only_latent:
            self.rho = nn.Linear(
                inside_dims[-1] * len(self.dim_channels), n_classes)

    def forward(self, params):
        reps = []
        for_prev = None

        for param, layer in zip(params, self.layers):
            # shape: (n_samples, n_pixels_in_kernel, channels_out, channels_in)
            prev_shape = param.shape

            # shape: (n_samples, channels_out, n_pixels_in_kernel, channels_in)
            param = param.transpose(1, 2)

            # shape: (n_samples, channels_out, n_pixels_in_kernel * channels_in)
            param = ch.flatten(param, 2)

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
        if self.only_latent:
            return reps

        logits = self.rho(reps)
        return logits


class PermInvModel(nn.Module):
    def __init__(self, dims, inside_dims=[64, 8], n_classes=2, dropout=0.5, only_latent=False):
        super(PermInvModel, self).__init__()
        self.dims = dims
        self.dropout = dropout
        self.only_latent = only_latent
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
            self.rho = nn.Linear(inside_dims[-1] * len(dims), n_classes)

    def forward(self, params):
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


def prepare_batched_data(X):
    inputs = [[] for _ in range(len(X[0]))]
    for x in X:
        for i, l in enumerate(x):
            inputs[i].append(l)

    inputs = np.array([ch.stack(x, 0) for x in inputs], dtype='object')
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


def train_epoch(train_loader, model, criterion, optimizer, epoch, verbose=True):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    if verbose:
        iterator = tqdm(train_loader)
    for data in iterator:
        images, labels, _ = data
        images, labels = images.cuda(), labels.cuda()
        N = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)[:, 0]

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


def validate_epoch(val_loader, model, criterion, verbose=True):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with ch.no_grad():
        for data in val_loader:
            images, labels, _ = data
            images, labels = images.cuda(), labels.cuda()
            N = images.size(0)

            outputs = model(images)[:, 0]
            prediction = (outputs >= 0)

            val_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels.float()).item())

    if verbose:
        print('[Validation], Loss: %.5f, Accuracy: %.4f' %
              (val_loss.avg, val_acc.avg))
        print()
    return val_loss.avg, val_acc.avg


def train(model, loaders, lr=1e-3, epoch_num=10,
          weight_decay=0, verbose=True, get_best=False):
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
                              criterion, optimizer, epoch, verbose)
        vloss, vacc = validate_epoch(val_loader, model, criterion, verbose)
        if not verbose:
            iterator.set_description(
                "train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))

        if get_best and vloss < best_loss:
            best_loss = vloss
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
              binary=True, regression=False, gpu=False,
              combined=False):
    model.eval()

    # Batch data to fit on GPU
    loss, num_samples, running_acc = 0, 0, 0
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
            if gpu:
                param_batch = [a.cuda() for a in param_batch]

            if binary or regression:
                outputs.append(model(param_batch)[:, 0])
            else:
                outputs.append(model(param_batch))

        outputs = ch.cat(outputs, 0)

        num_samples += outputs.shape[0]
        loss += loss_fn(outputs,
                        Y[i:i+batch_size]).item() * num_samples
        if not regression:
            running_acc += accuracy(outputs, Y[i:i+batch_size]).item()

        # Next batch
        i += batch_size

    return 100 * running_acc / num_samples, loss / num_samples


# Function to train meta-classifier
def train_meta_model(model, train_data, test_data,
                     epochs, lr, eval_every=5,
                     binary=True, regression=False,
                     val_data=None, batch_size=1000,
                     gpu=False, combined=False):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

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
        rp_tr = np.random.permutation(y.shape[0])
        if not combined:
            params, y = params[rp_tr], y[rp_tr]
        else:
            y = y[rp_tr]
            params = [x[rp_tr] for x in params]

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
                if gpu:
                    param_batch = [a.cuda() for a in param_batch]

                if binary or regression:
                    outputs.append(model(param_batch)[:, 0])
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
                                        gpu=gpu, combined=combined)
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
                                      gpu=gpu, combined=combined)
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
                                  gpu=gpu, combined=combined)
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
    classes = np.concatenate((np.zeros_like(accs_1), np.ones_like(accs_2)))
    best_acc = 0.0
    best_threshold = 0
    while lower < upper:
        best_of_two, rule = get_threshold_acc(combined, classes, lower)
        if best_of_two > best_acc:
            best_threshold = lower
            best_acc = best_of_two
            best_rule = rule

        lower += granularity

    return best_acc, best_threshold, best_rule


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
