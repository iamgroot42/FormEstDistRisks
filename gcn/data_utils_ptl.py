from torch_geometric.nn import GCNConv, SAGEConv
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class GraphNodeDataset:
    def __init__(self, name, device, make_symm=False):
        self.dataset = PygNodePropPredDataset(name=name, transform=T.ToSparseTensor())
        self.device = device
        
        self.data = self.dataset[0]
        # make undirected
        if make_symm:
            self.data.adj_t = self.data.adj_t.to_symmetric()
        
        self.data = self.data.to(self.device)

        self.num_features = self.data.num_features
        self.num_classes = self.dataset.num_classes
        self.num_nodes = self.data.num_nodes


class ArxivNodeDataset(GraphNodeDataset):
    def __init__(self, device, split):
        super(ArxivNodeDataset, self).__init__('ogbn-arxiv', device, make_symm=True)
        # 59:41 victim:adv data split (all original data, including train/val/test)
        # Original data had 54:46 train-nontrain split
        # Get similar splits
        split_year = 2016
        if split == 'adv':
            # 77:23 train:test split
            test_year = 2015
            self.train_idx = self.data.node_year < test_year
            self.test_idx = ch.logical_and(
                self.data.node_year >= test_year, self.data.node_year < split_year)
        elif split == 'victim':
            # 66:34 train:test split
            test_year = 2019
            self.train_idx = ch.logical_and(
                self.data.node_year != test_year, self.data.node_year >= split_year)
            self.test_idx = (self.data.node_year == test_year)
        else:
            raise ValueError("Invalid split requested!")
            
        self.train_idx = ch.nonzero(self.train_idx, as_tuple=True)[0]
        self.test_idx = ch.nonzero(self.test_idx, as_tuple=True)[0]
    
    def get_idx_split(self):
        return self.train_idx, self.test_idx


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)

def get_model(ds, device, args):
    model = GCN(ds.num_features, args.hidden_channels,
                ds.num_classes, args.num_layers,
                args.dropout).to(device)
    return model


def train(model, data, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@ch.no_grad()
def test(model, data, train_idx, test_idx, evaluator):
    model.eval()

    out = model(data.x, data.adj_t)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[train_idx],
        'y_pred': y_pred[train_idx],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[test_idx],
        'y_pred': y_pred[test_idx],
    })['acc']

    return train_acc, test_acc


def train_model(ds, model, evaluator, args):
    run_accs = {
        "train": [],
        "test": []
    }

    train_idx, test_idx = ds.get_idx_split()

    model.reset_parameters()
    optimizer = ch.optim.Adam(model.parameters(), lr=args.lr)
    iterator = tqdm(range(1, 1 + args.epochs))
        
    for epoch in iterator:
        loss = train(model, ds.data, train_idx, optimizer)
        train_acc, test_acc = test(
            model, ds.data, train_idx, test_idx, evaluator)

        iterator.set_description(f'Epoch: {epoch:02d}, '
                    f'Loss: {loss:.4f}, '
                    f'Train: {100 * train_acc:.2f}%, '
                    f'Test: {100 * test_acc:.2f}%')
        
        # Keep track of train/test accuracies across runs
        run_accs["train"].append(train_acc)
        run_accs["test"].append(test_acc)

    return run_accs


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig("./clustered_tsne.png")
