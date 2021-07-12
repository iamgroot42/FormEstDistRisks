from torch.utils.data import Dataset
from botdet.data.dataloader import GraphDataLoader
import deepdish as dd
from tqdm import tqdm
import os
import torch as ch
import random
import numpy as np
import dgl

LOCAL_DATA_DIR = "/localtmp/as9rw/datasets/botnet_temp"
# LOCAL_DATA_DIR = "/p/adversarialml/as9rw/datasets/raw_botnet"
METADATA_FILE_PATH = "coeffs.csv"


class BotNetWrapper:
    def __init__(self, split, prop_val, feat_len=1,
                 botnet_size=10000, interval=2, n_copies=2):

        self.n_copies = n_copies
        self.victim_adv_ratio = 0.67
        self.train_test_ratio = 0.8
        self.path = LOCAL_DATA_DIR

        # Use local storage (much faster)
        if not os.path.exists(LOCAL_DATA_DIR):
            raise FileNotFoundError("Local data not found")

        if split == "victim":
            num_train_graphs = 16
        else:
            num_train_graphs = 8

        # Get train/test splits
        wanted_ids = self.get_ids(split, prop_val)

        # Sample according to number of graphs
        # print("Got %d graphs before sampling" % (len(wanted_ids)))
        # wanted_ids = wanted_ids[:num_graphs]

        # Train/test split right here
        tr_te_split = int(self.train_test_ratio * len(wanted_ids))

        # Load graph according to specific IDs
        graphs = self.load_graphs(wanted_ids)

        # Split into train/test
        graphs_train = graphs[:tr_te_split]
        graphs_test = graphs[tr_te_split:]

        self.train_data = BotnetDataset(
            graphs_train, num_node=botnet_size,
            num_edge=botnet_size, interval=interval,
            num_features=feat_len, n_copies=n_copies)

        self.test_data = BotnetDataset(
            graphs_test, num_node=botnet_size,
            num_edge=botnet_size, interval=interval,
            num_features=feat_len, n_copies=n_copies)

    def get_ids(self, split, val):
        # Load metadata file
        names, values = [], []
        with open(METADATA_FILE_PATH, 'r') as f:
            for line in f:
                n, v = line.rstrip("\n").split(',')
                names.append("/" + n)
                values.append(float(v))
        names = np.array(names)
        values = np.array(values)

        # Split according to months
        nov = np.array(["20181115" in x for x in names])
        dec = np.array(["20181220" in x for x in names])

        # val_0 = (values >= 0.006)
        # val_1 = (values <= 0.0055)
        val_0 = (values >= 0.0057)
        val_1 = (values <= 0.0053)

        nov = np.nonzero(np.logical_and(nov, val_0))[0]
        dec = np.nonzero(np.logical_and(dec, val_1))[0]

        # Create victim/adversary splits
        # Use a 2:1 split
        split_0 = int(self.victim_adv_ratio * len(nov))
        split_1 = int(self.victim_adv_ratio * len(dec))

        # Property-based split right here
        if split == "victim":
            zero = nov[:split_0]
            one = dec[:split_1]
        else:
            zero = nov[split_0:]
            one = dec[split_1:]

        if val == "nov":
            return names[zero]
        else:
            return names[one]

    def load_graphs(self, wanted_ids):
        # Load graphs
        graphs = dd.io.load(os.path.join(
            self.path, "all_graphs.hdf5"), wanted_ids)

        return graphs

    def get_loaders(self, batch_size, shuffle=False, num_workers=0):
        # Create loaders
        self.train_loader = GraphDataLoader(
            self.train_data, batch_size, shuffle, num_workers)
        self.test_loader = GraphDataLoader(
            self.test_data, batch_size, False, num_workers)

        return self.train_loader, self.test_loader


class BotnetDataset(Dataset):
    def __init__(self, graphs, num_node, num_edge, interval,
                 num_features=1, n_copies=2):
        super(BotnetDataset, self).__init__()
        self.graph_format = "dgl"
        self.num_node = num_node
        self.num_edge = num_edge
        self.interval = interval
        self.num_features = num_features
        self.graphs = []
        self.n_copies = n_copies

        for g in tqdm(graphs):

            for _ in range(self.n_copies):
                # Pre-process graph
                x = self.pre_process_graph(g)

                # Add to graphs
                self.graphs.append(x)

    def generate_botnet_graph(self, graph):
        # Generate botnet edges
        botnet_edges = chord(self.num_node, self.num_edge, self.interval)

        # OVerlay botnets on graph
        evil_edges, evil = overlay_botnet_on_graph(graph, botnet_edges)
        graph.add_edges_from(evil_edges.T)

        # Create node labels
        y = np.zeros(graph.number_of_nodes())
        y[evil] = 1

        return (graph, y)

    def pre_process_graph(self, graph):

        # Overlay botnet
        graph, label = self.generate_botnet_graph(graph)

        # Convert to DGL graph, assign features and labels
        dg_graph = dgl.from_networkx(graph)

        # Convert to undirected
        dg_graph = dgl.to_bidirected(dg_graph)

        # Add self loops
        dg_graph = dgl.transform.add_self_loop(dg_graph)

        # Add node features
        features = ch.ones(dg_graph.num_nodes(), self.num_features,)
        dg_graph.ndata['x'] = features

        # Add labels
        dg_graph.ndata['y'] = ch.from_numpy(label)

        return dg_graph

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def chord(num_node, num_edge, interval):
    edge = [[i % num_node, (i+1) % num_node]for i in range(num_node)]

    fingers = [x for x in range(0, num_node, interval)]
    for (i, finger) in enumerate(fingers):
        for j in range(i+1, len(fingers)):
            edge.append([finger, fingers[j]])

    edge_select = random.sample(edge, num_edge)
    return np.array(edge_select)


def overlay_botnet_on_graph(graph, botnet_edges):
    # select evil node randomly
    evil_edges = np.array(botnet_edges).T
    evil_original = list(
        set(evil_edges[0, :].tolist()+evil_edges[1, :].tolist()))
    num_evil = len(evil_original)
    evil = random.sample(range(graph.number_of_nodes()), num_evil)

    evil_dict = {evil_original[i]: evil[i] for i in range(num_evil)}
    for row in range(evil_edges.shape[0]):
        for col in range(evil_edges.shape[1]):
            evil_edges[row, col] = evil_dict[evil_edges[row, col]]

    return evil_edges, evil


if __name__ == "__main__":
    ds = BotNetWrapper("adv", "nov", botnet_size=1000)
    print()
    ds = BotNetWrapper("adv", "dec", botnet_size=1000)
    print()
    ds = BotNetWrapper("victim", "nov", botnet_size=1000)
    print()
    ds = BotNetWrapper("victim", "dec", botnet_size=1000)
    print()
