from utils import PermInvModel, get_weight_layers
from data_utils import RatingModel
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch as ch
import numpy as np


def load_models(path):
    params = []
    for p in tqdm(os.listdir(path)):
        fp = os.path.join(path, p)
        model = RatingModel(768, binary=True).cuda()
        model.load_state_dict(ch.load(fp))
        model.eval()
        dims, param = get_weight_layers(model, normalize=True)
        params.append(param)
    return params, dims


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


if __name__ == "__main__":
    def get_data(pos_path, neg_path):
        # Load models
        pos_params, dims = load_models(pos_path)
        neg_params, _ = load_models(neg_path)
        data = pos_params + neg_params
        labels = [1.] * len(pos_params) + [0.] * len(neg_params)
        labels = ch.from_numpy(np.array(labels))
        return data, labels, dims

    # data, labels, dims = get_data("./data/models/Pyes", "./data/models/Pno")
    # data, labels, dims = get_data(
    #     "./data/models_split/second/yes", "./data/models_split/second/no")
    # X_train, X_test, y_train, y_test = train_test_split(data, labels,
    #                                                     test_size=0.35)

    X_train, y_train, dims = get_data(
        "./data/models_split/second/yes", "./data/models_split/second/no")
    X_test, y_test, _ = get_data(
        "./data/models_split/first/yes", "./data/models_split/first/no")

    # Train metamodel
    metamodel = PermInvModel(dims)
    train_model(metamodel,
                (X_train, y_train),
                (X_test, y_test),
                epochs=150,
                eval_every=10)
