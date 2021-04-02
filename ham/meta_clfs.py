import data_utils
import torch as ch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils
import os


# Function to extract model weights for all models in given directory
def get_model_features(model_dir, max_read=None):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        model = data_utils.HamModel(1024)
        model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        model.eval()

        dims, fvec = utils.get_weight_layers(model)

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


if __name__ == "__main__":
    import sys
    first_cat = sys.argv[1]
    second_cat = sys.argv[2]

    train_dir_1 = "/p/adversarialml/as9rw/models_ham_justin/split_1/%s/" % first_cat
    train_dir_2 = "/p/adversarialml/as9rw/models_ham_justin/split_1/%s/" % second_cat
    test_dir_1 = "/p/adversarialml/as9rw/models_ham_justin/split_2/%s/" % first_cat
    test_dir_2 = "/p/adversarialml/as9rw/models_ham_justin/split_2/%s/" % second_cat
    features_avail = False

    # Load models, convert to features
    dims, vecs_train_1 = get_model_features(train_dir_1)
    _, vecs_train_2 = get_model_features(train_dir_2)

    _, vecs_test_1 = get_model_features(test_dir_1)
    _, vecs_test_2 = get_model_features(test_dir_2)

    # Ready train, test data
    Y_train = [0.] * len(vecs_train_1) + [1.] * len(vecs_train_2)
    Y_train = ch.from_numpy(np.array(Y_train))
    X_train = vecs_train_1 + vecs_train_2

    Y_test = [0.] * len(vecs_test_1) + [1.] * len(vecs_test_2)
    Y_test = ch.from_numpy(np.array(Y_test))
    X_test = vecs_test_1 + vecs_test_2

    # Batch data
    print("Batching data: hold on")

    # Train meta-classifier model
    metamodel = utils.PermInvModel(dims, inside_dims=[128, 64, 32])

    train_model(metamodel,
                (X_train, Y_train),
                (X_test, Y_test),
                epochs=150,
                eval_every=10)
