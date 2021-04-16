from re import L
import torch as ch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR, ImageNet, SVHN, RobustCIFAR, CelebA
from robustness.tools import folder
from robustness.tools.misc import log_statement
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import Dataset
from sklearn import preprocessing
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm
import requests
import pandas as pd
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


class Celeb(DataPaths):
    def __init__(self, data_path=None):
        self.dataset_type = CelebA
        datapath = "/p/adversarialml/as9rw/datasets/celeba/" if data_path is None else data_path
        super(Celeb, self).__init__('celeb',
                                    datapath,
                                    "/p/adversarialml/as9rw/celeba_stats/")
        self.attr_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                           'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                           'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                           'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                           'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                           'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                           'Wearing_Necklace', 'Wearing_Necktie', 'Young']
        # self.model_prefix['resnet50'] = "/p/adversarialml/as9rw/models_celeba/"


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


# US Income dataset
class CensusIncome:
    def __init__(self, path):
        self.urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
                     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]
        self.columns = ["age", "workClass", "fnlwgt", "education", "education-num",
                        "marital-status", "occupation", "relationship",
                        "race", "sex", "capital-gain", "capital-loss",
                        "hours-per-week", "native-country", "income"]
        self.path = path
        self.download_dataset()

    def download_dataset(self):
        if not os.path.exists(self.path):
            log_statement("==> Downloading US Census Income dataset")
            os.mkdir(self.path)
            log_statement(
                "==> Please modify test file to remove stray dot characters")

            for url in self.urls:
                data = requests.get(url).content
                filename = os.path.join(self.path, os.path.basename(url))
                with open(filename, "wb") as file:
                    file.write(data)

    def process_df(self, df):
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        def oneHotCatVars(x, colname):
            df_1 = df.drop(columns=colname, axis=1)
            df_2 = pd.get_dummies(df[colname], prefix=colname, prefix_sep=':')
            return (pd.concat([df_1, df_2], axis=1, join='inner'))

        # colnames = ['workClass', 'education', 'occupation', 'race', 'sex', 'marital-status', 'relationship', 'native-country']
        colnames = ['workClass', 'occupation', 'race', 'sex',
                    'marital-status', 'relationship', 'native-country']
        # Drop education
        df = df.drop(columns='education', axis=1)
        for colname in colnames:
            df = oneHotCatVars(df, colname)
        return df

    def load_data(self,
                  train_filter=None,
                  test_ratio=0.5,
                  random_state=42,
                  first=None,
                  sample_ratio=1.):
        train_data = pd.read_csv(os.path.join(self.path, 'adult.data'), names=self.columns,
                                 sep=' *, *', na_values='?', engine='python')
        test_data = pd.read_csv(os.path.join(self.path, 'adult.test'), names=self.columns,
                                sep=' *, *', skiprows=1, na_values='?', engine='python')

        # Add field to identify train/test, process together, split back
        train_data['is_train'] = 1
        test_data['is_train'] = 0
        df = pd.concat([train_data, test_data], axis=0)
        df = self.process_df(df)

        train_df, test_df = df[df['is_train'] == 1], df[df['is_train'] == 0]

        def s_split(this_df, rs=random_state):
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=test_ratio,
                                         random_state=rs)
            splitter = sss.split(
                this_df, this_df[["sex:Female", "race:White", "income"]])
            split_1, split_2 = next(splitter)
            return this_df.iloc[split_1], this_df.iloc[split_2]

        train_df_first, train_df_second = s_split(train_df)
        test_df_first, test_df_second = s_split(test_df)

        # Further sample data, if requested
        if sample_ratio < 1:
            # Don't fixed random seed (random sampling wanted)
            # In this mode, train and test data are combined
            train_df_first = pd.concat([train_df_first, test_df_first])
            train_df_second = pd.concat([train_df_second, test_df_second])
            _, train_df_first = s_split(train_df_first, None)
            _, train_df_second = s_split(train_df_second, None)

        def get_x_y(P):
            # Scale X values
            Y = P['income'].to_numpy()
            X = P.drop(columns='income', axis=1)
            cols = X.columns
            X = X.to_numpy()
            return (X.astype(float), np.expand_dims(Y, 1), cols)

        def prepare_one_set(TRAIN_DF, TEST_DF):
            # Apply filter to train data
            # Efectively using different distribution to train it
            if train_filter is not None:
                TRAIN_DF = train_filter(TRAIN_DF)

            (x_tr, y_tr, cols), (x_te, y_te, cols) = get_x_y(
                TRAIN_DF), get_x_y(TEST_DF)
            # Preprocess data (scale)
            X = np.concatenate((x_tr, x_te), 0)
            X = preprocessing.scale(X)
            x_tr = X[:x_tr.shape[0]]
            x_te = X[x_tr.shape[0]:]

            return (x_tr, y_tr), (x_te, y_te), cols

        if first is None:
            return prepare_one_set(train_df, test_df)
        if first is True:
            return prepare_one_set(train_df_first, test_df_first)
        return prepare_one_set(train_df_second, test_df_second)

    def split_on_col(self, x, y, cols,
                     second_ratio, seed=42, debug=False):
        # Temporarily combine cols and y for stratification
        y_ = np.concatenate((y, x[:, cols]), 1)
        x_1, x_2, y_1, y_2 = train_test_split(x,
                                              y_,
                                              test_size=second_ratio,
                                              random_state=seed)
        # If debugging enabled, print ratios of col_index
        # Before and after splits
        if debug:
            print("Before:", np.mean(x[:, cols], 0), np.mean(y)) 
            print("After:", np.mean(x_1[:, cols], 0), np.mean(y_1[:, 0]))
            print("After:", np.mean(x_2[:, cols], 0), np.mean(y_2[:, 0]))
        # Get rid of extra column in y
        y_1, y_2 = y_1[:, 0], y_2[:, 0]
        # Return split data
        return (x_1, y_1), (x_2, y_2)


# Classifier on top of face features
class FaceModel(nn.Module):
    def __init__(self, n_feat, weight_init='vggface2', train_feat=False, hidden=[64, 16]):
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


class MNISTFlatModel(nn.Module):
    def __init__(self):
        super(MNISTFlatModel, self).__init__()
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


class CelebACustomBinary(Dataset):
    def __init__(self, root_dir, shuffle=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get filenames
        path_0, path_1 = os.path.join(
            self.root_dir, "0"), os.path.join(self.root_dir, "1")
        filenames_0 = [os.path.join(path_0, x) for x in os.listdir(path_0)]
        filenames_1 = [os.path.join(path_1, x) for x in os.listdir(path_1)]
        self.filenames = filenames_0 + filenames_1
        if shuffle:
            np.random.shuffle(self.filenames)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        x = Image.open(filename)
        y = os.path.basename(os.path.normpath(filename)).split("_")[0]
        y = np.array([int(c) for c in y])
        if self.transform:
            x = self.transform(x)
        return x, y


def get_weight_layers(m, normalize=False):
    dims, weights, biases = [], [], []
    for name, param in m.named_parameters():
        if "weight" in name:
            weights.append(param.data.detach().cpu().T)
            dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

    if normalize:
        min_w = min([ch.min(x).item() for x in weights])
        max_w = max([ch.max(x).item() for x in weights])
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
        weights = [w / max_w for w in weights]

    cctd = []
    for w, b in zip(weights, biases):
        cctd.append(ch.cat((w, b), 0).T)

    return dims, cctd


# Currently works with a batch size of 1
# Shouldn't be that big a deal, since here's only a few
# Couple hundred models

class PermInvModel(nn.Module):
    def __init__(self, dims, inside_dims=[64, 8], n_classes=2):
        super(PermInvModel, self).__init__()
        self.dims = dims
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
            layers.append(nn.Dropout())

            return nn.Sequential(*layers)

        for i, dim in enumerate(self.dims):
            # +1 for bias
            # prev_layer for previous layer
            # input dimension per neuron
            if i > 0:
                prev_layer = inside_dims[-1] * dim
            self.layers.append(make_mini(prev_layer + 1 + dim))

        self.layers = nn.ModuleList(self.layers)
        # Final network to combine them all together
        self.rho = nn.Linear(inside_dims[-1] * len(dims), n_classes)

    def forward(self, params):
        reps = []
        prev_layer_reps = None
        is_batched = len(params[0].shape) > 2

        for param, layer in zip(params, self.layers):
            # Process nodes in this layer
            if prev_layer_reps is None:
                if is_batched:
                    prev_shape = param.shape
                    processed = layer(param.view(-1, param.shape[-1]))
                    processed = processed.view(
                        prev_shape[0], prev_shape[1], -1)
                else:
                    processed = layer(param)
            else:
                # Handle per-data/batched-data case together
                if is_batched:
                    prev_layer_reps = prev_layer_reps.repeat(
                        1, param.shape[1], 1)
                else:
                    prev_layer_reps = prev_layer_reps.repeat(param.shape[0], 1)

                # Include previous layer representation
                param_eff = ch.cat((param, prev_layer_reps), -1)
                if is_batched:
                    prev_shape = param_eff.shape
                    processed = layer(param_eff.view(-1, param_eff.shape[-1]))
                    processed = processed.view(
                        prev_shape[0], prev_shape[1], -1)
                else:
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

        logits = self.rho(reps_c)
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


def acc_fn(x, y):
    with ch.no_grad():
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

    inputs = [ch.stack(x, 0) for x in inputs]
    return inputs


def heuristic(df, condition, ratio,
              cwise_sample,
              class_imbalance=2.0,
              n_tries=1000):
    vals, pckds = [], []
    iterator = tqdm(range(n_tries))
    for _ in iterator:
        pckd_df = filter(df, condition, ratio, verbose=False)
        # Class-balanced sampling
        zero_ids = np.nonzero(pckd_df["label"].to_numpy() == 0)[0]
        one_ids = np.nonzero(pckd_df["label"].to_numpy() == 1)[0]

        zero_ids = np.random.permutation(
            zero_ids)[:int(class_imbalance * cwise_sample)]
        one_ids = np.random.permutation(
            one_ids)[:cwise_sample]
        # Combine them together
        pckd = np.sort(np.concatenate((zero_ids, one_ids), 0))
        pckd_df = pckd_df.iloc[pckd]

        vals.append(condition(pckd_df).mean())
        pckds.append(pckd_df)

        # Print best ratio so far in descripton
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


def train(model, loaders, lr=1e-3, epoch_num=10, weight_decay=0, verbose=True):
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
    for epoch in iterator:
        tloss, tacc = train_epoch(train_loader, model, criterion, optimizer, epoch, verbose)
        vloss, vacc = validate_epoch(val_loader, model, criterion, verbose)
        if not verbose:
            iterator.set_description(
                "train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))

    return vloss, vacc
