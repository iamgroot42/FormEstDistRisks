import numpy as np
import os
import torch as ch
from tqdm import tqdm
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AlbertTokenizer, AlbertModel


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
    def __init__(self, path, indices_path=None, dfilter=None):
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
            self.va = np.load(os.path.join(indices_path, "val.npy"))
            self.te = np.load(os.path.join(indices_path, "test.npy"))
            self.tr = np.load(os.path.join(indices_path, "train.npy"))

    def load_all_data(self):
        self.valdata = AmazonDataset(self.do['validation'],
                                     os.path.join(self.path,
                                                  "val", "features.npy"),
                                     indices=self.va,
                                     dfilter=self.dfilter)

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

    def get_val_loader(self, batch_size):
        return DataLoader(self.valdata,
                          batch_size=batch_size,
                          shuffle=True)

    def get_test_loader(self, batch_size):
        return DataLoader(self.testdata,
                          batch_size=batch_size,
                          shuffle=True)


class RatingModel(nn.Module):
    def __init__(self, n_inp, binary=False):
        super(RatingModel, self).__init__()
        last_num = 5
        if binary:
            last_num = 1

        self.layers = nn.Sequential(
            nn.Linear(n_inp, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, last_num)
        )

    def forward(self, x):
        return self.layers(x)


def get_features(data, model, batch_size=32):
    all_outputs = []
    for i in tqdm(range(0, len(data), batch_size)):
        texts = data[i:i+batch_size]['review_body']

        tokenized = tokenizer(texts,
                              return_tensors="pt",
                              truncation=True,
                              max_length=512,
                              padding=True)

        for k, v in tokenized.items():
            tokenized[k] = v.cuda()

        with ch.no_grad():
            _, output = model(**tokenized)
            output = output.detach().cpu().numpy()
            all_outputs.append(output)

    all_outputs = np.concatenate(all_outputs, 0)
    return all_outputs


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


if __name__ == "__main__":
    model = AlbertModel.from_pretrained('albert-base-v2')
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
    model.eval()
    model.cuda()

    base = "./data/albert/"
    ratio = 0.5

    ad = load_dataset("amazon_reviews_multi", 'en')
    pick, save = ['train', 'validation', 'test'], ['train', 'val', 'test']
    for p, s in zip(pick, save):
        features = get_features(ad[p], model, 32)
        np.save(os.path.join(base, s, "features"), features)

    for p, s in zip(pick, save):
        first, second = prop_length_preserving_split(
            ad[p], ratio, verbose=False)
        np.save(os.path.join("./data/splits", "first", s), first)
        np.save(os.path.join("./data/splits", "second", s), second)
