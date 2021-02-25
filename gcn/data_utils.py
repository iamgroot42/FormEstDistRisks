from dgl.data import AmazonCoBuyComputerDataset, PubmedGraphDataset, AmazonCoBuyPhotoDataset
import torch as ch
from dgl.transform import add_self_loop
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from dgl.nn.pytorch import GraphConv
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class GraphData:
    def __init__(self, dclass, shuffle=False):
        self.dataset = dclass()

        self.data = self.dataset[0]
        self.data = add_self_loop(self.data)
        self.data = self.data.int().to('cuda')
        # if shuffle:
        #     self.dataset.shuffle()
        self.num_features = self.data.ndata['feat'].shape[1]
        self.num_classes = self.dataset.num_classes
        self.num_nodes = self.data.number_of_nodes()

    def make_train_test_masks(self, test_ratio=0.2):
        # train_mask = self.data.ndata['train_mask']
        # test_mask = self.data.ndata['val_mask']
        num_test = int(test_ratio * self.num_nodes)
        train_mask = ch.zeros(self.num_nodes, dtype=ch.bool)
        train_mask[num_test:] = 1
        test_mask = ch.zeros(
            self.num_nodes, dtype=ch.bool)
        test_mask[:num_test] = 1
        return train_mask, test_mask

    def get_data(self):
        return self.data

    def get_features(self):
        return self.data.ndata['feat']

    def get_labels(self):
        return self.data.ndata['label']


class AmazonComputersData(GraphData):
    def __init__(self, shuffle=False):
        super(AmazonComputersData, self).__init__(
            AmazonCoBuyComputerDataset, shuffle=shuffle)


class AmazonPhotoData(GraphData):
    def __init__(self, shuffle=False):
        super(AmazonPhotoData, self).__init__(
            AmazonCoBuyPhotoDataset, shuffle=shuffle)


class PubMedData(GraphData):
    def __init__(self, shuffle=False):
        super(PubMedData, self).__init__(
            PubmedGraphDataset, shuffle=shuffle)


class GCN(nn.Module):
    def __init__(self, ds,
                 n_hidden, n_layers,
                 dropout):
        super(GCN, self).__init__()
        self.g = ds.data
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(ds.num_features, n_hidden, activation=F.relu))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=F.relu))
        # output layer
        self.layers.append(GraphConv(n_hidden, ds.num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h


def get_model(ds):
    model = GCN(ds, n_hidden=16, n_layers=1, dropout=0.1)
    model.cuda()
    return model


def train_model(ds, test_ratio, model, lr=1e-3, epochs=2000):
    optimizer = ch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    loss_fn = ch.nn.CrossEntropyLoss().cuda()
    model.train()

    train_mask, test_mask = ds.make_train_test_masks(test_ratio)
    data = ds.get_features()
    labels = ds.get_labels()

    iterator = tqdm(range(epochs))
    for _ in iterator:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        iterator.set_description("Loss: %.3f" % loss.item())

    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(labels[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())
    return acc


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig("./clustered_tsne.png")
