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


def process_data(path):
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

    # this will tell us how many images are associated with each lesion_id
    df_undup = df.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df['duplicates'] = df['lesion_id']
    # apply the function to this new column
    df['duplicates'] = df['duplicates'].apply(get_duplicates)

    # now we create a val set using df because we are sure that
    # none of these images have augmented duplicates in the train set
    df_undup = df[df['duplicates'] == 'unduplicated']

    stratify = df_undup['stratify']
    _, df_val = train_test_split(
        df_undup, test_size=0.2,
        stratify=stratify)

    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df['train_or_val'] = df['image_id']
    df['train_or_val'] = df['train_or_val'].apply(get_val_rows)
    df_train = df[df['train_or_val'] == 'train']

    return df_train.reset_index(), df_val.reset_index()


class HamDataset(Dataset):
    # Borrow basic pre-processing steps from this notebook:
    # https://www.kaggle.com/xinruizhuang/skin-lesion-classification-acc-90-pytorch
    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        X = Image.open(self.df['path'][index])
        if self.transform:
            X = self.transform(X)

        y = ch.tensor(int(self.df['label'][index]))
        age = ch.tensor(int(self.df['age'][index]))
        sex = ch.tensor(int(self.df['sex'][index]))

        return X, y, (age, sex)


class AmazonDataset(Dataset):
    def __init__(self, data, path, indices=None, dfilter=None):
        self.X = np.load(path)
        Y, P = data['stars'], data['product_category']
        self.Y = np.array(Y)
        self.P = np.array(P)

        if indices is not None:
            self.X = self.X[indices]
            self.Y = self.Y[indices]
            self.P = self.P[indices]

        # Get rid of neutral data
        wanted = (self.Y != 3)
        self.X = self.X[wanted]
        self.Y = self.Y[wanted]
        self.P = self.P[wanted]

        # Binarize data into positive-negative
        self.Y = 1 * (self.Y > 3)

        # Run filter to get rid of specific data
        if dfilter:
            wanted = dfilter(self.P)
            self.X = self.X[wanted]
            self.Y = self.Y[wanted]
            self.P = self.P[wanted]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        return (self.X[index], self.Y[index], self.P[index])


class AmazonWrapper:
    def __init__(self, path, indices_path=None, dfilter=None,
                 secondary_indices_path=None, merge_ratio=0.5):
        self.path = path
        self.do = load_dataset("amazon_reviews_multi", 'en')
        self.dfilter = dfilter
        self.properties = [
            'apparel', 'automotive', 'baby_product', 'beauty', 'book',
            'camera', 'digital_ebook_purchase', 'digital_video_download',
            'drugstore', 'electronics', 'furniture', 'grocery', 'home',
            'home_improvement', 'industrial_supplies', 'jewelry', 'kitchen',
            'lawn_and_garden', 'luggage', 'musical_instruments',
            'office_product', 'other', 'pc', 'personal_care_appliances',
            'pet_products', 'shoes', 'sports', 'toy', 'video_games', 'watch',
            'wireless']
        # If supplied with path to pick subset of indices, use them
        self.indices_path = indices_path
        self.tr, self.va, self.te = None, None, None
        if indices_path is not None:
            va = np.load(os.path.join(indices_path, "val.npy"))
            te = np.load(os.path.join(indices_path, "test.npy"))
            tr = np.load(os.path.join(indices_path, "train.npy"))

        # If second source of data provided
        # Sample data from both sources according to given ratio
        # Of overlap between two sources
        if secondary_indices_path is not None:
            va_2 = np.load(os.path.join(secondary_indices_path, "val.npy"))
            te_2 = np.load(os.path.join(secondary_indices_path, "test.npy"))
            tr_2 = np.load(os.path.join(secondary_indices_path, "train.npy"))

            # Sample 'merge_ratio' data from second source
            # And remaining data from original source
            def merge(b, a):
                numa = int(merge_ratio * len(a))
                numb = b.shape[0] - numa
                return np.concatenate((np.random.choice(a, numa, replace=False),
                                       np.random.choice(b, numb, replace=False)))

            self.va, self.te, self.tr = merge(
                va, va_2), merge(te, te_2), merge(tr, tr_2)

        else:
            self.va, self.te, self.tr = va, te, tr

    def load_all_data(self):
        self.testdata = AmazonDataset(self.do['test'],
                                      os.path.join(self.path,
                                                   "test", "features.npy"),
                                      indices=self.te,
                                      dfilter=self.dfilter)

        self.traindata = AmazonDataset(self.do['train'],
                                       os.path.join(self.path,
                                                    "train", "features.npy"),
                                       indices=self.tr,
                                       dfilter=self.dfilter)

    def get_train_loader(self, batch_size):
        return DataLoader(self.traindata,
                          batch_size=batch_size,
                          shuffle=True)

    def get_test_loader(self, batch_size):
        return DataLoader(self.testdata,
                          batch_size=batch_size,
                          shuffle=True)


def prop_length_preserving_split(data, ratio=0.5,
                                 batch_size=5000, verbose=False):
    props, lens, stars = [], [], []
    indices = np.arange(len(data))
    for i in tqdm(range(0, len(data), batch_size)):
        texts = data[i:i+batch_size]['review_body']
        lens.append(list(map(lambda x: len(x.split(' ')), texts)))
        props.append(data[i:i+batch_size]['product_category'])
        stars.append(data[i:i+batch_size]['stars'])

    # Do not consider 3-rating reviews
    stars = np.concatenate(stars)
    wanted = (stars != 3)
    indices = indices[wanted]
    props = np.concatenate(props)
    lens = np.concatenate(lens)

    # Convert ratings to binary
    stars = 1 * (stars > 3)
    parts = int(np.ceil(1 / ratio))

    # Maintain same distribution across properties
    uprops = np.unique(props[wanted])
    first, second = [], []
    for up in tqdm(uprops):
        # And stars
        for i in range(2):
            # And lengths
            current = np.logical_and(stars[wanted] == i, props[wanted] == up)
            curr_lens = lens[wanted][current]
            curr_indices = indices[current]

            # Sort chosen lenghts
            sorted_order = np.argsort(curr_lens)
            curr_indices = curr_indices[sorted_order]

            # Pick every 'parts' for first split
            first_picked = curr_indices[::parts]
            second_picked = list(set(curr_indices) - set(first_picked))

            first.append(first_picked)
            if len(second_picked) != 0:
                second.append(second_picked)

    first = np.concatenate(first)
    second = np.concatenate(second)

    if verbose:
        # Print sizes of splits
        print("Desired ratio: %.2f, wanted ratio: %.2f" % (
            first.shape[0]/(first.shape[0] + second.shape[0]), ratio
        ))
        # Print property distribution (difference)
        print("Distributionof items per product-category")
        first_distr = np.unique(props[first], return_counts=True)[1]
        second_distr = np.unique(props[second], return_counts=True)[1]
        print(first_distr / len(props[first]))
        print(second_distr / len(props[second]))
        # Print length distribution statistics
        print("Mean, median, min/max range for first split:")
        print(np.mean(lens[first]), np.median(lens[first]),
              np.min(lens[first]), np.max(lens[first]))
        print("Mean, median, min/max range for second split:")
        print(np.mean(lens[second]), np.median(lens[second]),
              np.min(lens[second]), np.max(lens[second]))

    return first, second


def make_model():
    model_ft = models.densenet121(pretrained=True)
    for param in model_ft.parameters():
        param.requires_grad = False

    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.cuda()
    model_ft = nn.DataParallel(model_ft)

    return model_ft


def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    iterator = tqdm(train_loader)
    for data in iterator:
        images, labels, _ = data
        images, labels = images.cuda(), labels.cuda() 
        N = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(
            labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())
        curr_iter += 1
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

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(
                labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels).item())

    print('[Validation], Loss: %.5f, Accuracy: %.4f' %
          (val_loss.avg, val_acc.avg))
    print()
    return val_loss.avg, val_acc.avg


def train(model, loaders, lr=1e-3, epoch_num=10):
    # Get data loaders
    train_loader, val_loader = loaders
    # Define optimizer, loss function
    optimizer = ch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(1, epoch_num+1):
        train_epoch(train_loader, model, criterion, optimizer, epoch)
        validate_epoch(val_loader, model, criterion, optimizer, epoch)


if __name__ == "__main__":
    base = "/p/adversarialml/as9rw/datasets/ham10000/"
    ratio = 0.5

    # Process data
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    train_df, val_df = process_data(base)

    # Stratified split for adversary/local data training
    train_df, _ = stratified_df_split(train_df, 0.5)
    # val_df, _ = stratified_df_split(val_df, 0.5)

    hd_train = HamDataset(train_df, transform=transform)
    hd_val = HamDataset(val_df, transform=transform)

    # Define loaders
    batch_size = 128
    train_loader = DataLoader(hd_train, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(hd_val, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # Make model
    model = make_model()

    # Train model
    train(model, (train_loader, val_loader), lr=1e-2)

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
