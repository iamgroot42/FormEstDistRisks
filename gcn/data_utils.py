import dgl
from dgl.nn.pytorch import GraphConv
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
import torch as ch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


class GraphData:
    def __init__(self, name, normalize=True):
        self.data = DglNodePropPredDataset(name=name)

        self.g, self.labels = self.data[0]

        # Extract node features
        self.features = self.g.ndata['feat']

        # Degrees of nodes
        self.degs = []

        # Normalize features
        if normalize:
            m, std = ch.mean(self.features, 0), ch.std(self.features, 0)
            self.features = (self.features - m) / std

        self.num_features = self.features.shape[1]
        self.num_classes = self.data.num_classes
        self.num_nodes = self.g.number_of_nodes()

        # Extract any extra data
        self.before_init()

        # Process graph before shifting to GPU
        self.pre_process()

        # Shift data to GPU
        self.shift_to_gpu()

    def before_init(self):
        pass

    def pre_process(self):
        # Add self loops
        self.g = dgl.remove_self_loop(self.g)
        self.g = dgl.add_self_loop(self.g)

        # Make bidirectional
        self.g = dgl.to_bidirected(self.g)

    def shift_to_gpu(self):
        # Shift graph, labels to cuda
        self.g = self.g.to('cuda')
        self.labels = self.labels.cuda()
        self.features = self.features.cuda()

    def get_idx_split(self, test_ratio=0.2):
        # train_mask = self.data.ndata['train_mask']
        # test_mask = self.data.ndata['val_mask']
        num_test = int(test_ratio * self.num_nodes)
        train_mask = ch.zeros(self.num_nodes, dtype=ch.bool)
        train_mask[num_test:] = 1
        test_mask = ch.zeros(
            self.num_nodes, dtype=ch.bool)
        test_mask[:num_test] = 1
        return train_mask, test_mask

    def get_features(self):
        return self.features

    def get_labels(self):
        return self.labels

    def precompute_degrees(self):
        X, _ = self.g.edges()
        degs = []
        for i in tqdm(range(self.g.number_of_nodes())):
            degs.append(ch.sum(X == i).item())
        self.degs = np.array(degs)


class ArxivNodeDataset(GraphData):
    def __init__(self, split, normalize=True):
        super(ArxivNodeDataset, self).__init__(
            'ogbn-arxiv', normalize=normalize)

        # 59:41 victim:adv data split (all original data, including train/val/test)
        # Original data had 54:46 train-nontrain split
        # Get similar splits
        split_year = 2016
        if split == 'adv':
            # 77:23 train:test split
            test_year = 2015
            self.train_idx = self.years < test_year
            self.test_idx = ch.logical_and(
                self.years >= test_year, self.years < split_year)
        elif split == 'victim':
            # 66:34 train:test split
            test_year = 2019
            self.train_idx = ch.logical_and(
                self.years != test_year, self.years >= split_year)
            self.test_idx = (self.years == test_year)
        else:
            raise ValueError("Invalid split requested!")

        self.train_idx = ch.nonzero(self.train_idx, as_tuple=True)[0]
        self.test_idx = ch.nonzero(self.test_idx, as_tuple=True)[0]

        # Sort them now, for easier access later
        self.train_idx = ch.sort(self.train_idx)[0]
        self.test_idx = ch.sort(self.test_idx)[0]

    def before_init(self):
        # Extract years
        self.years = ch.squeeze(self.g.ndata['year'], 1)
        self.years = self.years.cuda()

    def get_idx_split(self):
        return self.train_idx, self.test_idx

    def random_split_pick(self, ratio):
        # Randomly sample 'ratio' worth of specified train data
        # Set train mask to those ones
        n_elems = len(self.train_idx)
        perm = ch.randperm(n_elems)[:int(ratio * n_elems)]
        self.train_idx = self.train_idx[perm]

    def label_ratio_preserving_pick(self, total_tr, total_te):
        # While maintaining relative label ratios for classes
        # Sample a set of size total
        labels = ch.cat(
            (self.labels[self.train_idx, 0], self.labels[self.test_idx, 0]), 0)
        elems, counts = ch.unique(labels, return_counts=True)
        # Get ratios for these elems
        counts = counts.float() / len(labels)
        # Sample according to ratio from existing train, test sets
        train_new, test_new = [], []
        for e, c in zip(elems, counts):
            # for train
            qualify = ch.nonzero(
                self.labels[self.train_idx, 0] == e, as_tuple=True)[0]
            ids = ch.randperm(len(qualify))[:int(c * total_tr)]
            train_new.append(qualify[ids])
            # for test
            qualify = ch.nonzero(
                self.labels[self.test_idx, 0] == e, as_tuple=True)[0]
            ids = ch.randperm(len(qualify))[:int(c * total_te)]
            test_new.append(qualify[ids])

        self.train_idx = ch.cat(train_new, 0)
        self.test_idx = ch.cat(test_new, 0)

    def change_mean_degree(self, wanted_degree):
        # If no change requested, perform no change
        if wanted_degree is None:
            return

        # Compute degrees
        self.precompute_degrees()

        # Prune graph, get rid of nodes
        self.g, pruned_nodes = achieve_mean_degree(
            self.g, self.degs, wanted_degree)

        # Make mapping between old and new IDs
        not_pruned = ch.ones(self.num_nodes).byte()
        not_pruned[pruned_nodes] = False
        not_pruned = ch.nonzero(not_pruned, as_tuple=True)[0]
        mapping = {x.item(): i for i, x in enumerate(not_pruned)}

        # Function to modify current masks to reflect pruning
        def process(ids):
            # Convert indices to mask
            keep = ch.zeros(self.num_nodes).byte()
            keep[ids] = True

            # Get rid of pruned nodes from this, get back to IDs
            not_pruned = ch.ones(self.num_nodes).byte()
            not_pruned[pruned_nodes] = False
            not_pruned = ch.nonzero(not_pruned, as_tuple=True)[0]
            keep[pruned_nodes] = False
            keep = ch.nonzero(keep, as_tuple=True)[0]

            # Update current mask to point to correct IDs
            for i, x in enumerate(keep):
                keep[i] = mapping[x.item()]

            # Shift mask back to GPU
            keep = keep.cuda()
            return keep

        # Update masks to account for pruned nodes,  re-indexing
        self.train_idx = process(self.train_idx)
        self.test_idx = process(self.test_idx)

        # Update features, labels, year-information
        self.years = self.years[not_pruned]
        self.features = self.features[not_pruned]
        self.labels = self.labels[not_pruned]

        # Update number of nodes in graph
        self.num_nodes -= len(pruned_nodes)


def find_to_prune(g, degs, wanted_deg):
    prune = []
    # Take note of mean degree right now
    cur_deg = np.mean(degs)

    # If desired degree is more than current, prune low-degree nodes
    if wanted_deg > cur_deg:

        inf_val = np.max(degs) + 1
        while cur_deg < wanted_deg:
            # Find minimum right now
            pick = np.argmin(degs)

            # Reduce degree of all nodes that were connected to pruned node
            # And are up for consideration currently
            L, R = g.edges()
            neighbors = R[L == pick].cpu().numpy()
            neighbors_mask = np.zeros_like(degs, dtype=bool)
            neighbors_mask[neighbors] = True
            neighbors_mask[degs == inf_val] = False
            degs[neighbors_mask] -= 1

            # Removed this node, mark as INF
            degs[pick] = inf_val

            cur_deg = np.mean(degs[degs != inf_val])
            prune.append(pick)

    # Else, prune high-degree nodes
    else:

        inf_val = -1
        while cur_deg > wanted_deg:
            # Find minimum right now
            pick = np.argmax(degs)

            # Reduce degree of all nodes that were connected to pruned node
            # And are up for consideration currently
            L, R = g.edges()
            neighbors = R[L == pick].cpu().numpy()
            neighbors_mask = np.zeros_like(degs, dtype=bool)
            neighbors_mask[neighbors] = True
            neighbors_mask[degs == inf_val] = False
            degs[neighbors_mask] -= 1

            # Removed this node, mark as -1
            degs[pick] = inf_val

            cur_deg = np.mean(degs[degs != inf_val])
            prune.append(pick)
    # Return list of nodes that should be removed
    return prune


def achieve_mean_degree(g, degs, wanted_deg):
    # Find which nodes should be pruned
    to_prune = find_to_prune(g, degs, wanted_deg)

    # Get rid of these nodes from graph
    g = dgl.remove_nodes(g, to_prune)

    return g, to_prune


def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.savefig("./clustered_tsne.png")
