from dgl.nn.pytorch import GraphConv
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import argparse
import data_utils
from utils import get_weight_layers


BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_arxiv_new/"


class GCN(nn.Module):
    def __init__(self, ds,
                 n_hidden, n_layers,
                 dropout):
        super(GCN, self).__init__()
        # Get actual graph
        self.g = ds.g

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

    def forward(self, features, latent=None):
        if latent is not None:
            if latent < 0 or latent > len(self.layers):
                raise ValueError("Invald interal layer requested")

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i == latent:
                return h
        return h


def get_model(ds, args):
    model = GCN(ds, args.hidden_channels,
                args.num_layers, args.dropout)
    if args.gpu:
        model = model.cuda()
    return model


def save_model(model, split, prop_and_name, prefix=None):
    if prefix is None:
        prefix = BASE_MODELS_DIR
    savepath = os.path.join(split, prop_and_name)
    ch.save(model.state_dict(), os.path.join(prefix, savepath))


def train(model, ds, train_idx, optimizer, loss_fn):
    model.train()

    X = ds.get_features()
    Y = ds.get_labels()

    optimizer.zero_grad()
    out = model(X)[train_idx]
    loss = loss_fn(out, Y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@ch.no_grad()
def test(model, ds, train_idx, test_idx, evaluator):
    model.eval()

    X = ds.get_features()
    Y = ds.get_labels()

    out = model(X)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': Y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': Y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, test_acc


def train_model(ds, model, evaluator, args):
    run_accs = {
        "train": [],
        "test": []
    }

    train_idx, test_idx = ds.get_idx_split()

    optimizer = ch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=5e-4)
    loss_fn = ch.nn.CrossEntropyLoss().cuda()
    iterator = tqdm(range(1, 1 + args.epochs))

    for epoch in iterator:
        loss = train(model, ds, train_idx, optimizer, loss_fn)
        train_acc, test_acc = test(
            model, ds, train_idx, test_idx, evaluator)

        iterator.set_description(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%')

        # Keep track of train/test accuracies across runs
        run_accs["train"].append(train_acc)
        run_accs["test"].append(test_acc)

    return run_accs


def get_model_features(model_dir, ds, args, max_read=None):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        # Define model
        model = get_model(ds, args)

        # Load weights into model
        if args.gpu:
            model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        else:
            model.load_state_dict(ch.load(os.path.join(
                model_dir, mpath), map_location=ch.device('cpu')))
        model.eval()

        # Extract model weights
        dims, fvec = get_weight_layers(
            model, transpose=False, first_n=args.first_n,
            start_n=args.start_n)

        vecs.append(fvec)

    return dims, vecs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--split', choices=['victim', 'adv'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument("--load_path", help="path to save trained model")
    args = parser.parse_args()
    print(args)

    # Get dataset ready
    ds = data_utils.ArxivNodeDataset(args.split)

    # Load weights into model
    model = get_model(ds, args)
    model.load_state_dict(ch.load(args.load_path))

    # Extract model weights
    get_weight_layers(model)
