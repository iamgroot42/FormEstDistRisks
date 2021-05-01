from data_utils import ArxivNodeDataset
import torch as ch
import argparse
import numpy as np
from tqdm import tqdm
import os
from model_utils import get_model, extract_model_weights
from utils import PermInvModel
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def get_model_features(model_dir, ds, args, max_read=None):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        # Define model
        model = get_model(ds, args)

        # Extract model weights
        dims, w = extract_model_weights(model)

        # Load weights into model
        model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        model.eval()

        dims, fvec = extract_model_weights(model)

        vecs.append(fvec)

    return dims, vecs


# Function to test meta-classifier
@ch.no_grad()
def test_model(model, params_test):
    outputs = []
    for param in params_test:
        outputs.append(model(param)[:, 0])

    outputs = ch.cat(outputs, 0)

    return outputs


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    print(args)

    # Set dark background
    plt.style.use('dark_background')

    # Get dataset ready (only need meta-data from this object)
    ds = ArxivNodeDataset('adv')

    # Directories where saved models are stored
    degrees = ["12.5", "13.5"]
    # degrees = ["9", "10", "11", "12", "13", "14", "15", "16", "17"]
    test_dirs = ["models/victim/deg" + x for x in degrees]

    # Load models, convert to features
    test_vecs = []
    for ted in test_dirs:
        dims, vecs_test = get_model_features(
            ted, ds, args, max_read=1000)

        test_vecs.append(vecs_test)

    model = PermInvModel(dims)
    model.load_state_dict(ch.load("./metamodel_0.37.pth"))
    model.eval()

    for i, vte in enumerate(test_vecs):

        output = test_model(model, vte)
        so, _ = ch.sort(output)
        so = so.numpy()

        # Predictions
        plt.plot(np.arange(len(so)), so, label=degrees[i])
        # Reference line
        plt.axhline(y=float(degrees[i]), linewidth=1.0, linestyle='--', color='C%d' % i)

    plt.xlabel("Models (1000), sorted by meta-classifier's predicted degree")
    plt.ylabel("Degree of graph predicted by meta-classifier")
    plt.savefig("./regression_plot_nolegend.png")
    plt.legend()
    plt.savefig("./regression_plot.png")


if __name__ == "__main__":
    main()
