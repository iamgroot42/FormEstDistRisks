from torch.utils.data import Dataset
from botdet.data.dataloader import GraphDataLoader
import os
import random
import numpy as np
import dgl

LOCAL_DATA_DIR = "/localtmp/as9rw/datasets/botnet_temp"
# VICTIM_COEFF_PATH = "./victim_info.txt"
# ADV_COEFF_PATH = "./adv_info.txt"
VICTIM_COEFF_PATH = "./victim_info_new.txt"
ADV_COEFF_PATH = "./adv_info_new.txt"


class BotNetWrapper:
    def __init__(self, split, prop_val, sample=True):
        self.train_test_ratio = 0.9
        self.path = LOCAL_DATA_DIR

        # if split == "adv":
        #     num_graphs_train_sample = 58
        # else:
        #     num_graphs_train_sample = 90

        num_graphs_train_sample = 75

        # Use local storage (much faster)
        if not os.path.exists(LOCAL_DATA_DIR):
            raise FileNotFoundError("Local data not found")

        # Load graph according to specific IDs
        graphs = self.load_graphs(split, prop_val)

        # Train/test split right here
        tr_te_split = int(self.train_test_ratio * len(graphs))

        # Split into train/test
        graphs_train = graphs[:tr_te_split]
        graphs_test = graphs[tr_te_split:]

        # Use specified number of graphs to train
        # Sample from train
        if sample:
            random_chosen_ids = np.random.choice(
                len(graphs_train), num_graphs_train_sample, replace=False)
            graphs_train = [graphs_train[i] for i in random_chosen_ids]

        self.train_data = BotnetDataset(graphs_train)
        self.test_data = BotnetDataset(graphs_test)

    def get_ids(self, split, val):
        # Load metadata file
        values = []
        if split == "adv":
            filename = ADV_COEFF_PATH
        else:
            filename = VICTIM_COEFF_PATH

        with open(filename, 'r') as f:
            for line in f:
                _, v = line.rstrip("\n").split(',')
                values.append(float(v))
        values = np.array(values)

        # Old:
        # val=0 -> coeff <= 0.0066
        # val=1 -> coeff > 0.0071

        if val == 0:
            ids = np.where(values < 0.0061)[0]
        else:
            ids = np.where(values > 0.0074)[0]

        return ids

    def load_graphs(self, split, val):
        # if split == "adv":
        #     graph_name = "dgl_adv.hdf5"
        # else:
        #     graph_name = "dgl_victim.hdf5"
        if split == "adv":
            graph_name = "dgl_adv_new.hdf5"
        else:
            graph_name = "dgl_victim_new.hdf5"

        # Load graphs
        graphs, _ = dgl.load_graphs(os.path.join(
            self.path, graph_name))

        wanted_ids = self.get_ids(split, val)
        graphs = [graphs[i] for i in wanted_ids]

        return graphs

    def get_loaders(self, batch_size, shuffle=False, num_workers=0):
        # Create loaders
        self.train_loader = GraphDataLoader(
            self.train_data, batch_size, shuffle, num_workers)
        self.test_loader = GraphDataLoader(
            self.test_data, batch_size, False, num_workers)

        return self.train_loader, self.test_loader


class BotnetDataset(Dataset):
    def __init__(self, graphs):
        super(BotnetDataset, self).__init__()
        self.graph_format = "dgl"
        self.graphs = graphs

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
    ds = BotNetWrapper("adv", 0)
    ds = BotNetWrapper("adv", 0)
    ds = BotNetWrapper("victim", 1)
    ds = BotNetWrapper("victim", 1)
