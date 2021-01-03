import torch as ch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
import os
import copy
import torch.nn as nn
from data_utils import AmazonWrapper, RatingModel


def train_model(model, t_loader, v_loader, epochs=50, lr=0.01, verbose=True):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')

    def acc_fn(x, y):
        return ch.sum((y == (x >= 0)))

    best_model, best_vacc = None, 0.0
    outiterator = range(epochs)
    if not verbose:
        outiterator = tqdm(outiterator)
    for e in outiterator:
        # Train
        running_loss, running_acc = 0.0, 0.0
        num_samples = 0
        model.train()
        iterator = t_loader
        if verbose:
            iterator = tqdm(iterator)
        for (x, y, _) in iterator:
            x, y = x.cuda(), y.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)[:, 0]
            loss = loss_fn(outputs, y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += acc_fn(outputs, y)
            num_samples += x.shape[0]

            if verbose:
                iterator.set_description("Epoch %d : [Train] Loss: %.5f Accuacy: %.2f" % (
                    e, running_loss / num_samples, 100 * running_acc / num_samples))

        # Validation
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        num_samples = 0
        for (x, y, _) in v_loader:
            x, y = x.cuda(), y.cuda()

            with ch.no_grad():
                outputs = model(x)[:, 0]
                loss = loss_fn(outputs, y.float())
                running_loss += loss.item()
                running_acc += acc_fn(outputs, y)
                num_samples += x.shape[0]

        if verbose:
            print("[Val] Loss: %.5f Accuacy: %.2f\n" %
                  (running_loss / num_samples, 100 * running_acc / num_samples))

        if running_acc / num_samples > best_vacc:
            best_model = copy.deepcopy(model)
            best_vacc = running_acc / num_samples

    return best_model, best_vacc


if __name__ == "__main__":
    import sys
    base_path = sys.argv[1]
    data_path = sys.argv[2]
    model_num = int(sys.argv[3])
    want_prop = int(sys.argv[4]) != 0

    # Property filter
    if want_prop:
        def dfilter(x):
            return np.logical_and(x != 'home', x != 'home_improvement')
        prefix = "yes"
    else:
        dfilter = None
        prefix = "no"

    # Load dataset
    do = AmazonWrapper("./data/roberta-base",
                       indices_path=data_path,
                       dfilter=dfilter)
    do.load_all_data()
    batch_size = 256
    # Get loaders ready
    train_loader = do.get_train_loader(batch_size)
    val_loader = do.get_val_loader(5000)
    test_loader = do.get_test_loader(5000)

    # Create model
    model = RatingModel(768, binary=True).cuda()
    # Train model
    best_model, best_vacc = train_model(model, train_loader, val_loader,
                                        lr=0.001, epochs=30, verbose=False)

    # Save model with best performance on validation data
    ch.save(best_model.state_dict(),
            os.path.join(base_path, prefix, "%d_%.3f.pth" % (model_num,
                                                             best_vacc)))
