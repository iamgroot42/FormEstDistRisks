import numpy as np
import os
import torch as ch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from glob import glob
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import utils
from sklearn.model_selection import train_test_split
import pandas as pd


def stratified_df_split(df, second_ratio):
    stratify = df['stratify']
    df_1, df_2 = train_test_split(
        df, test_size=second_ratio, stratify=stratify)
    return df_1.reset_index(), df_2.reset_index()


def process_data(path, split_second_ratio=0.5):
    df = pd.read_csv(os.path.join(path, "HAM10000_metadata.csv"))
    lesion_type_dict = {
        'nv': 0,
        'mel': 1,
        'bkl': 2,
        'bcc': 3,
        'akiec': 4,
        'vasc': 5,
        'df': 6
    }

    binary_task_map = {
        0: 0, 1: 1, 2: 1,
        3: 1, 4: 1, 5: 1, 6: 1
    }

    # Binarize gender entries
    df['sex'] = df['sex'].map(lambda x: 1 * (x == 'female'))
    # Binarize into age <=50 and > 50 (roughly-balanced split)
    df['age'] = df['age'].map(lambda x: 1 * (x > 50))

    # Get rid of ambiguous entries
    wanted = np.logical_not(np.logical_or(
        df.sex == 'unknown', np.isnan(df.age)))
    df = df[wanted]

    all_image_paths = glob(os.path.join(path, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[
        0]: x for x in all_image_paths}

    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df['label'] = df['dx'].map(lesion_type_dict.get)
    # Binarize label
    df['label'] = df['label'].map(binary_task_map.get)

    # Get new column for stratification purposes
    df['stratify'] = df.apply(lambda row: str(
        row.sex) + str(row.age) + str(row.label), axis=1)

    stratify = df['stratify']
    df_1, df_2 = train_test_split(
        df, test_size=split_second_ratio,
        stratify=stratify)

    return df_1.reset_index(), df_2.reset_index()


class HamDataset(Dataset):
    # Borrow basic pre-processing steps from this notebook:
    # https://www.kaggle.com/xinruizhuang/skin-lesion-classification-acc-90-pytorch
    def __init__(self, df, argument=None, processed=False):
        if processed:
            self.features = ch.tensor(argument)
        else:
            self.transform = argument
        self.processed = processed
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.processed:
            X = self.features[index]
        else:
            X = Image.open(self.df['path'][index])
            if self.transform:
                X = self.transform(X)

        y = ch.tensor(int(self.df['label'][index]))
        age = ch.tensor(int(self.df['age'][index]))
        sex = ch.tensor(int(self.df['sex'][index]))

        return X, y, (age, sex)


class HamWrapper:
    def __init__(self, df_train, df_val, dfilter=None):
        self.df_train = df_train
        self.df_val = df_val
        self.do = load_dataset("amazon_reviews_multi", 'en')
        self.dfilter = dfilter

        # Apply data filter, if filter provided
        if dfilter:
            wanted_tr = dfilter(self.df_train)
            wanted_te = dfilter(self.df_val)
            self.df_train = self.df_train[wanted_tr]
            self.df_val = self.df_val[wanted_te]
            self.df_train.reset_index()
            self.df_val.reset_index()


def property_splits(data, ratio=0.5,  verbose=False):
    # Construct temporary column for staratification
    # Based on all columns except the one to be modified
    pass


def set_parameter_requires_grad(model, fe):
    if fe:
        for param in model.parameters():
            param.requires_grad = False


class HamModel(nn.Module):
    def __init__(self, n_inp):
        super(HamModel, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_inp, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Linear(8, 1))

    def forward(self, x):
        return self.layers(x)


def make_feature_extractor():
    model = models.densenet121(pretrained=True)
    set_parameter_requires_grad(model, False)
    # model.classifier = nn.Identity()
    model.classifier = HamModel(1024)
    # Shift to GPU, add parallel-GPU support
    model = model.cuda()
    model = nn.DataParallel(model)
    return model


def get_features(model, loader):
    with ch.no_grad():
        all_features = []
        for (x, y, _) in tqdm(loader):
            features = model(x.cuda()).detach().cpu()
            all_features.append(features)
        all_features = ch.cat(all_features)
    return all_features.numpy()


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
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
        iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f]' % (
            epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
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

    print('[Validation], Loss: %.5f, Accuracy: %.4f' %
          (val_loss.avg, val_acc.avg))
    print()
    return val_loss.avg, val_acc.avg


def train(model, loaders, lr=1e-3, epoch_num=10):
    # Get data loaders
    train_loader, val_loader = loaders
    # Define optimizer, loss function
    optimizer = ch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss().cuda()

    for epoch in range(1, epoch_num+1):
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        validate_epoch(val_loader, model, criterion, optimizer, epoch)


if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/"
    ratio = 0.5
    input_size = 224
    # batch_size = 128 * 4
    batch_size = 100 * 4

    # Define augmentations
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(
            brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    model = make_feature_extractor()
    # Process datasets and get features
    df_1 = pd.read_csv("./data/split_1/original.csv")
    df_2 = pd.read_csv("./data/split_2/original.csv")
    ds_1 = HamDataset(df_1, val_transform)
    ds_2 = HamDataset(df_2, val_transform)
    train_loader = DataLoader(
        ds_1, batch_size=batch_size, shuffle=False, num_workers=8)
    val_loader = DataLoader(ds_2, batch_size=batch_size,
                            shuffle=False, num_workers=8)
    train(model, (train_loader, val_loader), lr=1e-3, epoch_num=20)
    exit(0)

    csv_available = False
    if csv_available:
        # Load DF data
        df_1 = pd.read_csv("./data/split_1/original.csv")
        df_2 = pd.read_csv("./data/split_2/original.csv")
        # Load pre-computed features
        features_1 = np.load("./data/split_1/features.npy")
        features_2 = np.load("./data/split_2/features.npy")
    else:
        # Get data splits (model trainer and adversary)
        df_1, df_2 = process_data(base)

        # Get feature extractor
        fe = make_feature_extractor()

        # Process datasets and get features
        ds_1 = HamDataset(df_1, val_transform)
        ds_2 = HamDataset(df_2, val_transform)
        loader_1 = DataLoader(ds_1, batch_size=batch_size,
                              shuffle=False, num_workers=8)
        loader_2 = DataLoader(ds_2, batch_size=batch_size,
                              shuffle=False, num_workers=8)
        # Get features for this data
        features_1 = get_features(fe, loader_1)
        features_2 = get_features(fe, loader_2)

        # Save DF file splits for later use
        df_1.to_csv("./data/split_1/original.csv")
        df_2.to_csv("./data/split_2/original.csv")
        # Save features in npy files
        np.save("./data/split_1/features", features_1)
        np.save("./data/split_2/features", features_2)

    # Stratified split for adversary/local data training
    train_df, val_df = stratified_df_split(df_1, 0.2)

    # hd_train = HamDataset(train_df, transform=train_transform)
    hd_train = HamDataset(train_df, features_1, processed=True)
    hd_val = HamDataset(val_df, features_2, processed=True)

    # Define loaders
    train_loader = DataLoader(hd_train, batch_size=batch_size,
                              shuffle=True, num_workers=8)
    val_loader = DataLoader(hd_val, batch_size=batch_size,
                            shuffle=False, num_workers=8)

    # Make model
    model = HamModel(1024)
    model = model.cuda()
    model = nn.DataParallel(model)

    # Train model
    train(model, (train_loader, val_loader), lr=1e-3, epoch_num=20)

    # ad = load_dataset("amazon_reviews_multi", 'en')
    # pick, save = ['train', 'validation', 'test'], ['train', 'val', 'test']
    # for p, s in zip(pick, save):
    #     features = get_features(ad[p], model, 32)
    #     np.save(os.path.join(base, s, "features"), features)

    # for p, s in zip(pick, save):
    #     first, second = prop_length_preserving_split(
    #         ad[p], ratio, verbose=False)
    #     np.save(os.path.join("./data/splits", "first", s), first)
    #     np.save(os.path.join("./data/splits", "second", s), second)
