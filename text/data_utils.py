import numpy as np
import os
import torch as ch
from tqdm import tqdm
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, RobertaTokenizer, \
    RobertaModel, AlbertTokenizer, AlbertModel


class AmazonDataset(Dataset):
    def __init__(self, data, path, dfilter=None):
        self.X = np.load(path)
        Y, P = data['stars'], data['product_category']
        self.Y = np.array(Y)
        self.P = np.array(P)

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
        # Shift down to [0-4] range for 0-indexing compatibility
        # return (self.X[index], self.Y[index]-1, self.P[index])
        return (self.X[index], self.Y[index], self.P[index])


class AmazonWrapper:
    def __init__(self, path, dfilter=None):
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

    def load_all_data(self):

        self.valdata = AmazonDataset(self.do['validation'],
                                     os.path.join(self.path,
                                                  "val", "features.npy"),
                                     self.dfilter)

        self.testdata = AmazonDataset(self.do['test'],
                                      os.path.join(self.path,
                                                   "test", "features.npy"),
                                      self.dfilter)

        self.traindata = AmazonDataset(self.do['train'],
                                       os.path.join(self.path,
                                                    "train", "features.npy"),
                                       self.dfilter)

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

        # self.layers = nn.Sequential(
        #     nn.Linear(n_inp, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, last_num)
        # )
        # 84, 86

        # self.layers = nn.Sequential(
        #     nn.Dropout(0.1),
        #     nn.Linear(n_inp, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, last_num)
        # )
        # 82, 85

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

        # print(tokenized)
        for k, v in tokenized.items():
            tokenized[k] = v.cuda()
        # input_ids = tokenized['input_ids'].cuda()
        # token_type_ids = tokenized['token_type_ids'].cuda()
        # attention_mask = tokenized['attention_mask'].cuda()

        with ch.no_grad():
            _, output = model(**tokenized)
            output = output.detach().cpu().numpy()
            all_outputs.append(output)

    all_outputs = np.concatenate(all_outputs, 0)
    return all_outputs


if __name__ == "__main__":
    # model = BertModel.from_pretrained("bert-base-uncased")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # model = RobertaModel.from_pretrained('roberta-base')
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    model = AlbertModel.from_pretrained('albert-base-v2')
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    model.eval()
    model.cuda()

    ad = load_dataset("amazon_reviews_multi", 'en')
    features = get_features(ad['train'], model, 32)
    np.save("./data/albert/train/features", features)
