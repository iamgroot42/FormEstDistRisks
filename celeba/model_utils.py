import torch as ch
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import normalize


BASE_MODELS_DIR = "/u/as9rw/work/fnb/celeba/celeba_models_split/70_30/"


# Classifier on top of face features
class FaceModel(nn.Module):
    def __init__(self, n_feat, weight_init='vggface2',
                 train_feat=False, hidden=[64, 16]):
        super(FaceModel, self).__init__()
        self.train_feat = train_feat
        if weight_init == "none":
            weight_init = None
        self.feature_model = InceptionResnetV1(
            pretrained=weight_init)  # .eval()
        if not self.train_feat:
            self.feature_model.eval()
        # for param in self.feature_model.parameters(): param.requires_grad = False

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
        if self.train_feat:
            x_ = self.feature_model(
                # x, with_latent=deep_latent)
                x, with_latent=deep_latent,
                within_block=within_block,
                flatmode=flatmode)
        else:
            with ch.no_grad():
                x_ = self.feature_model(
                    # x, with_latent=deep_latent)
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


def get_model(path, dim=512,
              hidden=[64, 16],
              weight_init=None,
              use_prefix=True):
    if use_prefix:
        path = os.path.join(BASE_MODELS_DIR, path)

    model = FaceModel(dim,
                      train_feat=True,
                      weight_init=weight_init,
                      hidden=hidden).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(ch.load(path), strict=False)
    model.eval()
    return model


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
