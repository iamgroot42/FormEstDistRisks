from torch.utils.data import Dataset
from botdet.data.dataloader import GraphDataLoader
from networkx.algorithms.components import connected_components
from networkx.algorithms.cluster import triangles
import deepdish as dd
from networkx.algorithms import approximation
from tqdm import tqdm
import os
import torch as ch
import random
import numpy as np
import dgl


LOCAL_DATA_DIR = "/localtmp/as9rw/datasets/botnet"
SPLIT_INFO_PATH = "/p/adversarialml/as9rw/datasets/botnet"


class BotNetWrapper:
    def __init__(self, split, prop_val, dataset_name="chord",
                 feat_len=1, botnet_size=1000, interval=2,
                 property="mcc"):
        if split == "victim":
            num_train_graphs = 200
            num_test_graphs = 50
        else:
            num_train_graphs = 100
            num_test_graphs = 25

        self.train_data = BotnetDataset(
            dataset_name, split, num_node=botnet_size,
            num_edge=botnet_size, interval=interval,
            property=property, value=prop_val,
            num_features=feat_len,
            num_graphs=num_train_graphs, is_train=True)

        self.test_data = BotnetDataset(
            dataset_name, split, num_node=botnet_size,
            num_edge=botnet_size, interval=interval,
            property=property, value=prop_val,
            num_features=feat_len,
            num_graphs=num_test_graphs, is_train=False)

    def get_loaders(self, batch_size, shuffle=False, num_workers=0):
        # Create loaders
        self.train_loader = GraphDataLoader(
            self.train_data, batch_size, shuffle, num_workers)
        self.test_loader = GraphDataLoader(
            self.test_data, batch_size, False, num_workers)

        return self.train_loader, self.test_loader


class BotnetDataset(Dataset):
    def __init__(self, dataset_name, split, num_node, num_edge, interval,
                 num_graphs, property, value, is_train, num_features=1):
        super(BotnetDataset, self).__init__()
        self.graph_format = "dgl"
        self.num_node = num_node
        self.num_edge = num_edge
        self.interval = interval
        self.value = value
        self.split = split
        self.num_features = num_features
        self.num_graphs = num_graphs
        self.is_train = is_train
        self.dataset_name = dataset_name

        # Use local storage (much faster)
        if not os.path.exists(LOCAL_DATA_DIR):
            raise FileNotFoundError("Local data not found")
        self.path = os.path.join(
            LOCAL_DATA_DIR, self.dataset_name, "graph.hdf5")

        if property == "mcc":
            self.property_fn = modify_clustering_coefficient
        else:
            raise ValueError("%s property not implemented" % property)

        graphs = self.load_graphs()

        self.graphs = []
        for g in tqdm(graphs):
            x, y = self.pre_process_graph(g)

            # Convert to DGL graph, assign features and labels
            dg_graph = dgl.from_networkx(x)
            features = ch.ones(dg_graph.num_nodes(), self.num_features,)
            dg_graph.ndata['feat'] = features
            dg_graph.ndata['y'] = ch.from_numpy(y)

            # Add to graphs
            self.graphs.append(dg_graph)

    def get_splits(self):
        splitname = "split_70_30.npz"
        splitfile = np.load(os.path.join(
            SPLIT_INFO_PATH, self.dataset_name, splitname),
            allow_pickle=True)
        return splitfile[self.split]

    def load_graphs(self):
        # TODO: Fetch indices according to split
        train_split, test_split = self.get_splits()
        if self.is_train:
            indices = train_split
        else:
            indices = test_split

        # Pick random sample of graph IDs from these indices
        wanted_ids = np.random.choice(indices, self.num_graphs, replace=False)
        wanted_ids = [str(x) for x in wanted_ids]

        # Load graphs from memory
        graphs = dd.io.load(self.path, group=wanted_ids)

        return graphs

    def generate_botnet_graph(self, graph):
        # Generate botnet edges
        botnet_edges = chord(self.num_node, self.num_edge, self.interval)

        # OVerlay botnets on graph
        evil_edges, evil = overlay_botnet_on_graph(graph, botnet_edges)
        graph.add_edges_from(evil_edges)

        # Create node labels
        y = np.zeros(graph.number_of_nodes())
        y[evil] = 1

        return (graph, evil)

    def pre_process_graph(self, graph):
        # Change property according to specification
        graph = self.property_fn(graph, self.value)

        # OVerlay botnet
        graph, label = self.generate_botnet_graph(graph)
        return graph, label

    def __len__(self):
        return self.num_graphs

    def __item__(self, index):
        return self.graphs(index)


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


def modify_clustering_coefficient(graph, coeff,
                                  n_trials=20000, together=10,
                                  m_trials=10, p_check=0.9,
                                  random_permute=0.1,
                                  at_most=5000, verbose=False):

    # Get counts for all triangles
    trians = np.array([triangles(graph, x) for x in graph.nodes])
    sorted_ids = np.argsort(-trians)

    # Add slight perturbation to sorted order
    # To increase variation in generated graphs
    # Randomly pick X% of data and permute order of indices
    random_ids = np.random.choice(len(trians), size=int(
        random_permute * len(trians)), replace=False)
    sorted_ids[random_ids] = sorted_ids[np.random.permutation(random_ids)]

    # Iterate through nodes
    iterator = tqdm(range(0, at_most, together))
    for i in iterator:

        # Remove 'together' nodes together to process faster
        for j in range(i, i + together):
            graph.remove_node(sorted_ids[j])

        # Get approximate clustering coefficient
        ca = approximation.clustering_coefficient.average_clustering(
                graph, trials=n_trials)

        if verbose:
            iterator.set_description(
                "Clustering-coefficient: %.3f | %d" %
                (ca, graph.number_of_nodes()))

        # If coefficient drops to desired value, run 'm_trials' more trials
        # if all of them satisfy desired coefficient, return graph
        if ca <= coeff:

            satisfy = 0
            for j in range(m_trials):
                ca_ = approximation.clustering_coefficient.average_clustering(
                    graph, trials=n_trials)
                satisfy += (ca_ <= coeff)

            # If 'p_check' of them satisfy required coefficient requirement
            # return current graph
            if (satisfy / m_trials) >= p_check:

                # Consider only largest component of pruned graph
                largest_cc = max(connected_components(graph), key=len)
                graph_ = graph.subgraph(largest_cc).copy()
                if verbose:
                    print("Numer of nodes decreased from %d to %d" %
                    (graph.number_of_nodes(), graph_.number_of_nodes()))

                return graph_

    # If nothing happened until this point, desired coefficient
    # was nto achieved - raise error
    raise ValueError("Desired coefficient not achieved!")


if __name__ == "__main__":
    dataset_name = "chord"

    ds = BotNetWrapper(dataset_name)

    # Create victim/adv splits
    victim_ratio = 0.7
    train_perm = np.random.permutation(len(ds.train_data))
    val_perm = np.random.permutation(len(ds.val_data))
    test_perm = np.random.permutation(len(ds.test_data))

    victim_train = train_perm[:int(victim_ratio * len(ds.train_data))]
    adv_train = train_perm[int(victim_ratio * len(ds.train_data)):]

    victim_val = val_perm[:int(victim_ratio * len(ds.val_data))]
    adv_val = val_perm[int(victim_ratio * len(ds.val_data)):]

    victim_test = test_perm[:int(victim_ratio * len(ds.test_data))]
    adv_test = test_perm[int(victim_ratio * len(ds.test_data)):]

    splits = {
        "victim": np.array(
            [victim_train, victim_val, victim_test], dtype=object),
        "adv": np.array(
            [adv_train, adv_val, adv_test], dtype=object)
    }
    print(len(victim_train), len(victim_val), len(victim_test))
    print(len(adv_train), len(adv_val), len(adv_test))

    # Save split information
    np.savez(os.path.join(SPLIT_INFO_PATH, dataset_name, "split_70_30"),
             adv=splits["adv"], victim=splits["victim"])
    print("Saved split information!")

    # Check time taken to load data
    import time
    start = time.time()
    ds = BotNetWrapper(split="adv")
    end = time.time()
    print("Time taken to load adv", end - start)

    start = time.time()
    ds = BotNetWrapper(split="victim")
    end = time.time()
    print("Time taken to load victim", end - start)
