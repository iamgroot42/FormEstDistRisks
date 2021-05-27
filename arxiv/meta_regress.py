from data_utils import ArxivNodeDataset
import torch as ch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model_utils import BASE_MODELS_DIR, get_model_features
from utils import PermInvModel


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

# Function to train meta-classifier
def train_model(model, train_data, test_data,
                eval_every=5, epochs=200, lr=0.001,
                binary=True, regression=False,
                save_every=100):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    if regression:
        loss_fn = nn.MSELoss()
    else:
        if binary:
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            loss_fn = nn.CrossEntropyLoss()

    params, y = train_data
    params_test, y_test = test_data

    def acc_fn(x, y):
        if binary:
            return ch.sum((y == (x >= 0)))
        return ch.sum(y == ch.argmax(x, 1))

    iterator = tqdm(range(epochs))
    for e in iterator:
        # Training
        model.train()

        outputs = []
        for param in params:
            if binary or regression:
                outputs.append(model(param)[:, 0])
            else:
                outputs.append(model(param))

        outputs = ch.cat(outputs, 0)
        optimizer.zero_grad()

        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

        num_samples = outputs.shape[0]
        loss = loss.item() * num_samples

        print_acc = ""
        if not regression:
            running_acc = acc_fn(outputs, y)
            print_acc = ", Accuacy: %.2f" % (100 * running_acc / num_samples)

        iterator.set_description("Epoch %d : [Train] Loss: %.5f%s" % (
                                     e, loss / num_samples, print_acc))

        if (e+1) % eval_every == 0:
            # Validation
            model.eval()
            outputs = []
            for param in params_test:
                if binary or regression:
                    outputs.append(model(param)[:, 0])
                else:
                    outputs.append(model(param))
            outputs = ch.cat(outputs, 0)
            with ch.no_grad():
                num_samples = outputs.shape[0]
                loss = loss_fn(outputs, y_test).item() * num_samples

                print_acc = ""
                if not regression:
                    running_acc = acc_fn(outputs, y_test)
                    print_acc = ", Accuacy: %.2f" % (100 * running_acc / num_samples)

                print("[Test] Loss: %.5f%s" % (loss / num_samples, print_acc))

        # Save if requested
        if (e + 1) % save_every == 0:
            # Save meta-model
            model.eval()
            ch.save(model.state_dict(), "./metamodel_old_%d.pth" % (e+1))

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--iters', type=int, default=300)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--degrees', default="9,10,11,12,13,14,15,16,17")
    parser.add_argument('--regression', action="store_true")
    parser.add_argument('--gpu', action="store_true")
    parser.add_argument('--first_n', type=int, default=np.inf,
                        help="Only consider first N layers")
    args = parser.parse_args()
    print(args)

    # Get dataset ready (only need meta-data from this object)
    ds = ArxivNodeDataset('adv', on_gpu=args.gpu)

    degrees = args.degrees.split(",")
    binary = len(degrees) == 2

    # Directories where saved models are stored
    train_dirs = [os.path.join(BASE_MODELS_DIR, "adv", "deg" + x)
                  for x in degrees]
    test_dirs = [os.path.join(BASE_MODELS_DIR, "victim", "deg" + x)
                 for x in degrees]

    # Load models, convert to features
    train_vecs, test_vecs = [], []
    for trd, ted in zip(train_dirs, test_dirs):
        dims, vecs_train = get_model_features(
            trd, ds, args, max_read=700)
        _, vecs_test = get_model_features(
            ted, ds, args, max_read=1000)

        train_vecs.append(vecs_train)
        test_vecs.append(vecs_test)

    # Ready train, test data
    Y_train, Y_test = [], []
    X_train, X_test = [], []
    for i, (vtr, vte) in enumerate(zip(train_vecs, test_vecs)):
        i_ = i
        if args.regression:
            i_ = float(degrees[i_])
        elif binary:
            i_ = float(i_)
        Y_train.append([i_] * len(vtr))
        Y_test.append([i_] * len(vte))

        X_train += vtr
        X_test += vte

    Y_train = ch.from_numpy(np.concatenate(Y_train))
    Y_test = ch.from_numpy(np.concatenate(Y_test))

    if binary or args.regression:
        Y_train = Y_train.float()
        Y_test = Y_test.float()

    # First experiment: shuffle labels and use those to train
    # np.random.shuffle(Y_train)

    # Second experiment- run as a n-class classification problem
    # Cells added/modified above

    # Train meta-classifier model
    if binary or args.regression:
        metamodel = PermInvModel(dims)
    else:
        metamodel = PermInvModel(dims, n_classes=len(degrees))

    metamodel = train_model(metamodel,
                            (X_train, Y_train),
                            (X_test, Y_test),
                            epochs=args.iters,
                            binary=binary,
                            regression=args.regression,
                            lr=0.001, eval_every=5,
                            save_every=100)


if __name__ == "__main__":
    main()
