from functools import reduce
from data_utils import ArxivNodeDataset
import torch as ch
import argparse
import numpy as np
from tqdm import tqdm
import os
from model_utils import get_model, BASE_MODELS_DIR
from utils import PermInvModel
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200


def extract_model_weights(m, normalize=False):
    dims, weights, biases = [], [], []
    for name, param in m.named_parameters():
        if "weight" in name:
            weights.append(param.data.detach().cpu())
            dims.append(weights[-1].shape[0])
        if "bias" in name:
            biases.append(ch.unsqueeze(param.data.detach().cpu(), 0))

    if normalize:
        min_w = min([ch.min(x).item() for x in weights])
        max_w = max([ch.max(x).item() for x in weights])
        weights = [(w - min_w) / (max_w - min_w) for w in weights]
        weights = [w / max_w for w in weights]

    cctd = []
    for w, b in zip(weights, biases):
        cctd.append(ch.cat((w, b), 0).T)

    return dims, cctd


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
    parser.add_argument('--darkplot', action="store_true",
                        help='Use dark background for plotting results')
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--legend', action="store_true",
                        help='Add legend to plots')
    parser.add_argument(
        '--degrees', default="9,10,11,12,12.5,13,13.5,14,15,16,17")
    args = parser.parse_args()
    print(args)

    # Set font size
    plt.rcParams.update({'font.size': 14})
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    plt.rc('axes', labelsize=14)

    if args.darkplot:
        # Set dark background
        plt.style.use('dark_background')

    # Get dataset ready (only need meta-data from this object)
    ds = ArxivNodeDataset('adv')

    # Directories where saved models are stored
    degrees = args.degrees.split(",")
    test_dirs = [os.path.join(BASE_MODELS_DIR, "victim", "deg" + x)
                 for x in degrees]

    # Load models, convert to features
    test_vecs = []
    for ted in test_dirs:
        dims, vecs_test = get_model_features(
            # ted, ds, args, max_read=10)
            ted, ds, args, max_read=1000)

        test_vecs.append(vecs_test)

    model = PermInvModel(dims)
    # model.load_state_dict(ch.load("./metamodel_old_0.53.pth"))
    model.load_state_dict(ch.load("./metamodel_old_400.pth"))
    model.eval()

    mse_loss, n_elems = 0, 0
    for i, vte in enumerate(test_vecs):

        # Get model predictions
        output = test_model(model, vte)

        # Compute MSE loss
        loss_fn = ch.nn.MSELoss(reduction='sum')
        gt = (ch.ones_like(output) * float(degrees[i]))
        loss_rn = loss_fn(output, gt).item()
        mse_loss += loss_rn
        n_elems += output.shape[0]

        so, _ = ch.sort(output)
        so = so.numpy()

        # Print individual losses
        print("MSE loss:", loss_rn / output.shape[0])

        # Predictions
        plt.plot(np.arange(len(so)), so, label=degrees[i])
        # Reference line
        plt.axhline(y=float(degrees[i]), linewidth=1.0,
                    linestyle='--', color='C%d' % i)

    print("Mean MSE loss: %.4f" % (mse_loss / n_elems))

    plt.xlabel("Models, sorted by predicted degree")
    plt.ylabel(r"Predicted mean-degree of training data ($\alpha$)")
    if args.legend:
        plt.legend(prop={'size': 13})
    plt.savefig("./regression_plot.png")


if __name__ == "__main__":
    main()
