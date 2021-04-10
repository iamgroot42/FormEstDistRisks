from data_utils import ArxivNodeDataset
import torch as ch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from model_utils import get_model, extract_model_weights
from utils import PermInvModel


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


# Function to train meta-classifier
def train_model(model, train_data, test_data,
                eval_every=5, epochs=200, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    params, y = train_data
    params_test, y_test = test_data

    def acc_fn(x, y):
        return ch.sum((y == (x >= 0)))

    iterator = tqdm(range(epochs))
    for e in iterator:
        # Training
        model.train()

        outputs = []
        for param in params:
            outputs.append(model(param)[:, 0])

        outputs = ch.cat(outputs, 0)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y.float())

        loss.backward()
        optimizer.step()

        num_samples = outputs.shape[0]
        loss = loss.item() * num_samples
        running_acc = acc_fn(outputs, y)

        iterator.set_description("Epoch %d : [Train] Loss: %.5f "
                                 "Accuacy: %.2f" % (
                                     e, loss / num_samples,
                                     100 * running_acc / num_samples))

        if (e+1) % eval_every == 0:
            # Validation
            model.eval()
            outputs = []
            for param in params_test:
                outputs.append(model(param)[:, 0])
            outputs = ch.cat(outputs, 0)
            with ch.no_grad():
                num_samples = outputs.shape[0]
                loss = loss_fn(outputs, y_test.float()).item() * num_samples
                running_acc = acc_fn(outputs, y_test)
                print("[Test] Loss: %.5f, Accuracy: %.2f" % (
                    loss / num_samples, 100 * running_acc / num_samples
                ))

    # return best_model, best_vacc


def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GNN)')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument("--load_path", help="path to save trained model")
    args = parser.parse_args()
    print(args)

    # Get dataset ready (only need meta-data from this object)
    ds = ArxivNodeDataset('adv')

    # Directories where saved models are stored
    train_dir_1 = "models/adv/deg13"
    train_dir_2 = "models/adv/deg11"
    test_dir_1 = "models/victim/deg13"
    test_dir_2 = "models/victim/deg11"

    # Load models, convert to features
    dims, vecs_train_1 = get_model_features(train_dir_1, ds, args, max_read=700)
    _, vecs_train_2 = get_model_features(train_dir_2, ds, args, max_read=700)

    _, vecs_test_1 = get_model_features(test_dir_1, ds, args)
    _, vecs_test_2 = get_model_features(test_dir_2, ds, args)

    # Ready train, test data
    Y_train = [0.] * len(vecs_train_1) + [1.] * len(vecs_train_2)
    Y_train = ch.from_numpy(np.array(Y_train))
    X_train = vecs_train_1 + vecs_train_2

    Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
    Y_test = ch.from_numpy(np.array(Y_test))
    X_test = vecs_test_1 + vecs_test_2

    # Train meta-classifier model
    metamodel = PermInvModel(dims)

    train_model(metamodel,
                (X_train, Y_train),
                (X_test, Y_test),
                epochs=40,
                eval_every=5)


if __name__ == "__main__":
    main()
