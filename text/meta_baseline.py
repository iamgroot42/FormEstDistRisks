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
        dims, param = get_weight_layers(model)
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

    # best_model, best_vacc = None, 0.0

    iterator = tqdm(range(epochs))
    for e in iterator:
        # Train
        model.train()

        outputs = []
        for param in params:
            outputs.append(model(param)[:, 0])

        outputs = ch.cat(outputs, 0)
        optimizer.zero_grad()
        loss = loss_fn(outputs, y.float())

        loss.backward()
        optimizer.step()

        loss = loss.item()
        running_acc = acc_fn(outputs, y)
        num_samples = outputs.shape[0]

        iterator.set_description("Epoch %d : [Train] Loss: %.5f Accuacy: %.2f" % (
            e, loss / num_samples, 100 * running_acc / num_samples))

        if e % eval_every == 0:
            # Validation
            model.eval()
            outputs = []
            for param in params_test:
                outputs.append(model(param)[:, 0])
            outputs = ch.cat(outputs, 0)
            with ch.no_grad():
                loss = loss_fn(outputs, y_test.float()).item()
                running_acc = acc_fn(outputs, y_test)
                num_samples = outputs.shape[0]
                print("[Test] Loss: %.5f, Accuracy: %.2f" % (
                    loss / num_samples, 100 * running_acc / num_samples
                ))

    # return best_model, best_vacc


if __name__ == "__main__":
    positive_path = "./data/models/Pyes"
    negative_path = "./data/models/Pno"
    # Load models
    pos_params, dims = load_models(positive_path)
    neg_params, _ = load_models(negative_path)
    data = pos_params + neg_params
    labels = [1.] * len(pos_params) + [0.] * len(neg_params)
    labels = ch.from_numpy(np.array(labels))

    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.35)

    # Train metamodel
    metamodel = PermInvModel(dims)
    train_model(metamodel,
                (X_train, y_train),
                (X_test, y_test),
                eval_every=10)
