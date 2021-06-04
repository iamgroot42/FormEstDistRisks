import torch as ch
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from utils import ensure_dir_exists


BASE_MODELS_DIR = "/u/as9rw/work/fnb/celeba/celeba_models/70_30/"


class VGG11BN(nn.Module):
    def __init__(self, num_classes: int = 1):
        super(VGG11BN, self).__init__()
        self.model = models.vgg11_bn(num_classes=num_classes)
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: ch.Tensor, latent=None):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = ch.flatten(x, 1)
        x = self.model.classifier(x)
        return x


# Classifier on top of face features
class FaceModel(nn.Module):
    def __init__(self, n_feat, weight_init='vggface2',
                 train_feat=False, hidden=[64, 16]):
        super(FaceModel, self).__init__()
        self.train_feat = train_feat
        if weight_init == "none":
            weight_init = None
        self.feature_model = InceptionResnetV1(
            pretrained=weight_init)
        if not self.train_feat:
            self.feature_model.eval()
            for param in self.feature_model.parameters():
                param.requires_grad = False

        layers = []
        # Input features -> hidden layer
        layers.append(nn.Linear(n_feat, hidden[0]))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout())
        for i in range(len(hidden)-1):
            layers.append(nn.Linear(hidden[i], hidden[i+1]))
            layers.append(nn.ReLU())

        # Last hidden -> binary classification layer
        layers.append(nn.Linear(hidden[-1], 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x, only_latent=False,
                deep_latent=None, within_block=None,
                flatmode=False):
        with ch.set_grad_enabled(not self.train_feat):
            x_ = self.feature_model(
                x, with_latent=deep_latent,
                within_block=within_block,
                flatmode=flatmode)

        # Check if Tuple
        if type(x_) is tuple and x_[1] is not None:
            return x_[1]
        if only_latent:
            return x_
        return self.dnn(x_)


class FlatFaceModel(nn.Module):
    def __init__(self, n_feat):
        super(FlatFaceModel, self).__init__()
        self.fc1 = nn.Linear(n_feat, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)

        # Weight init
        ch.nn.init.xavier_uniform(self.fc1.weight)
        ch.nn.init.xavier_uniform(self.fc2.weight)
        ch.nn.init.xavier_uniform(self.fc3.weight)

    def forward(self, x):
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(F.relu(self.fc2(x)), 0.5)
        x = self.fc3(x)
        return x


def create_model(dim, hidden, weight_init, train_feat=True, parallel=True):
    model = FaceModel(dim, weight_init=weight_init,
                      train_feat=train_feat, hidden=hidden).cuda()
    if parallel:
        model = nn.DataParallel(model)
    return model


def get_model(path, dim=512,
              hidden=[64, 16],
              weight_init=None,
              use_prefix=True,
              train_feat=True,
              parallel=True):
    if use_prefix:
        path = os.path.join(BASE_MODELS_DIR, path)

    model = create_model(dim, hidden, weight_init,
                         train_feat=train_feat, parallel=parallel)
    model.load_state_dict(ch.load(path), strict=False)

    if parallel:
        model = nn.DataParallel(model)

    model.eval()
    return model


def save_model(model, split, property, ratio, name, dataparallel=True):
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


def get_features_for_model(dataloader, MODELPATH, weight_init,
                           method_type, layers=[64, 16],):
    # Load model
    model = get_model(MODELPATH, dim=512, hidden=layers,
                      weight_init=weight_init)

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
