from botdet.data.dataset_botnet import BotnetDataset
from botdet.data.dataloader import GraphDataLoader
import os
import numpy as np


LOCAL_DATA_DIR = "/localtmp/as9rw/datasets/botnet"
SPLIT_INFO_PATH = "/p/adversarialml/as9rw/datasets/botnet"


class BotNetWrapper:
    def __init__(self, dataset_name="chord", split=None, feat_len=40):
        self.dataset_name = dataset_name
        self.split = split

        # Use local storage (much faster)
        if not os.path.exists(LOCAL_DATA_DIR):
            raise FileNotFoundError("Local data not found")
        data_dir = os.path.join(LOCAL_DATA_DIR, dataset_name)

        # Use adv/victim split
        split_info = [None] * 3
        if split is not None:
            splits = self.get_splits(split)
            for i, spl in enumerate(splits):
                split_info[i] = [str(x) for x in spl]

        self.train_data = BotnetDataset(
            name=dataset_name, root=data_dir,
            add_features_dgl=feat_len, in_memory=True,
            split='train', graph_format='dgl',
            partial_load=split_info[0])

        self.val_data = BotnetDataset(
            name=dataset_name, root=data_dir,
            add_features_dgl=feat_len, in_memory=True,
            split='val', graph_format='dgl',
            partial_load=split_info[1])

        self.test_data = BotnetDataset(
            name=dataset_name, root=data_dir,
            add_features_dgl=feat_len, in_memory=True,
            split='test', graph_format='dgl',
            partial_load=split_info[2])

    def get_splits(self, split):
        splitname = "split_70_30.npz"
        splitfile = np.load(os.path.join(
            SPLIT_INFO_PATH, self.dataset_name, splitname),
            allow_pickle=True)
        return splitfile[split]

    def get_loaders(self, batch_size, shuffle=False, num_workers=0):
        # Create loaders
        self.train_loader = GraphDataLoader(
            self.train_data, batch_size, shuffle, num_workers)
        self.test_loader = GraphDataLoader(
            self.test_data, batch_size, False, num_workers)

        return self.train_loader, self.test_loader


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
