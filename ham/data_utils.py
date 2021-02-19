import numpy as np
import os
import torch as ch
from tqdm import tqdm
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from glob import glob
from torch.utils.data import Dataset, DataLoader
import utils
from sklearn.model_selection import train_test_split
import pandas as pd


def stratified_df_split(df, second_ratio):
    # Get new column for stratification purposes
    def fn(row): return str(row.sex) + str(row.age) + str(row.label)
    col = df.apply(fn, axis=1)
    df = df.assign(stratify=col.values)

    stratify = df['stratify']
    df_1, df_2 = train_test_split(
        df, test_size=second_ratio,
        stratify=stratify)

    # Delete remporary stratification column
    df.drop(columns=['stratify'], inplace=True)
    df_1.drop(columns=['stratify'], inplace=True)
    df_2.drop(columns=['stratify'], inplace=True)
    return df_1.reset_index(inplace=True), df_2.reset_index(inplace=True)


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

    # Return stratified split
    return stratified_df_split(df, split_second_ratio)


class HamDataset(Dataset):
    # Borrow basic pre-processing steps from this notebook:
    # https://www.kaggle.com/xinruizhuang/skin-lesion-classification-acc-90-pytorch
    def __init__(self, df, argument=None, processed=False):
        if processed:
            self.features = argument
        else:
            self.transform = argument
        self.processed = processed
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.processed:
            X = self.features[self.df['path'].iloc[index]]
        else:
            X = Image.open(self.df['path'][index])
            if self.transform:
                X = self.transform(X)

        y = ch.tensor(int(self.df['label'][index]))
        age = ch.tensor(int(self.df['age'][index]))
        sex = ch.tensor(int(self.df['sex'][index]))

        return X, y, (age, sex)


class HamWrapper:
    def __init__(self, df_train, df_val, features=None):
        self.df_train = df_train
        self.df_val = df_val
        self.input_size = 224
        data_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        if features is None:
            self.ds_train = HamDataset(self.df_train, data_transform)
            self.ds_val = HamDataset(self.df_val, data_transform)
        else:
            self.ds_train = HamDataset(
                self.df_train, features["train"], processed=True)
            self.ds_val = HamDataset(
                self.df_val, features["val"], processed=True)

    def get_loaders(self, batch_size):
        train_loader = DataLoader(
            self.ds_train, batch_size=batch_size,
            shuffle=False, num_workers=2)
        # If train mode can handle BS (weight + gradient)
        # No-grad mode can surely hadle 2 * BS?
        val_loader = DataLoader(
            self.ds_val, batch_size=batch_size * 2,
            shuffle=False, num_workers=2)

        return train_loader, val_loader


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


def make_feature_extractor(fe=False):
    model = models.densenet121(pretrained=True)
    set_parameter_requires_grad(model, fe=fe)
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


def train_epoch(train_loader, model, criterion, optimizer, epoch, verbose=True):
    model.train()
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
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
            iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f]' % (
                epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion, verbose=True):
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
        train_epoch(train_loader, model, criterion, optimizer, epoch, verbose)
        vloss, vacc = validate_epoch(val_loader, model, criterion, verbose)

    return vloss, vacc


def useful_stats(df):
    print("%d | %.2f | %.2f | %.2f" % (
        len(df),
        df["label"].mean(),
        df["age"].mean(),
        df["sex"].mean()))


if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/"

    # First split of data, only visible to victim
    # Second split of data, only visible to adversary
    # For the sake of varying property ratios, both use
    # Samples from their respective splits
    # Same goes for train-val splits
    df_victim, df_adv = process_data(base)

    # Save these splits for victim
    for i, df in enumerate([df_victim, df_adv]):

        df_processed = utils.filter(df, None, None, verbose=False)

        # Print ratios for different properties and labels
        # As a sanity check
        utils.log_statement("original")
        useful_stats(df_processed)
        print()

        # Get train-val splits
        train_df, val_df = stratified_df_split(df_processed, 0.2)

        # Ensure directory exists
        dir_prefix = "./data/split_%d/%s/" % (i+1, "original")
        utils.ensure_dir_exists(dir_prefix)

        # Save train-test splits
        train_df.to_csv(os.path.join(dir_prefix, "train.csv"))
        val_df.to_csv(os.path.join(dir_prefix, "val.csv"))
