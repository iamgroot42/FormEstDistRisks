# Base file: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/gnn.py

import argparse
from matplotlib.pyplot import disconnect

import torch as ch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200

import data_utils

from ogb.nodeproppred import Evaluator



def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--split', choices=['victim', 'adv'])
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if ch.cuda.is_available() else 'cpu'
    device = ch.device(device)

    ds = data_utils.ArxivNodeDataset(device, split='victim')
    
    model = data_utils.get_model(ds, device, args)
    evaluator = Evaluator(name='ogbn-arxiv')

    run_accs = data_utils.train_model(ds, model, evaluator, args)
    mean_tr, std_tr = ch.mean(run_accs["train"]), ch.std(run_accs["train"])
    mean_te, std_te = ch.mean(run_accs["test"]), ch.std(run_accs["test"])

    print("Train accuracy: %.2f +- %.2f" % (mean_tr, std_tr))
    print("Test accuracy: %.2f +- %.2f" % (mean_te, std_te))


if __name__ == "__main__":
    main()
