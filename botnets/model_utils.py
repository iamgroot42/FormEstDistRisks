import torch.nn as nn
import torch.optim as optim
import torch as ch
from tqdm import tqdm
from dgl.nn.pytorch import GraphConv


# TODO: Find out why right norm method fails
class GCN(nn.Module):
    def __init__(self, n_inp, n_hidden, n_layers,
                 dropout=0.5, n_classes=2, residual=False,
                 norm='both'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.residual = residual

        # input layer
        self.layers.append(
            GraphConv(n_inp, n_hidden, norm=norm))

        # hidden layers
        for i in range(n_layers-1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, norm=norm))

        # output layer
        self.final = GraphConv(n_hidden, n_classes, norm=norm)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

    def forward(self, g, latent=None):

        if latent is not None:
            if latent < 0 or latent > len(self.layers):
                raise ValueError("Invald interal layer requested")

        x = g.ndata['feat']
        for i, layer in enumerate(self.layers):
            xo = self.activation(layer(g, x))
            xo = self.dropout(xo)

            # Add prev layer directly, if requested
            if self.residual and i != 0:
                xo = self.activation(xo + x)

            x = xo

            # Return representation, if requested
            if i == latent:
                return x

        return self.final(g, x)


def true_positive(pred, target):
    return (target[pred == 1] == 1).sum().item()


def get_metrics(y, y_pred, threshold=0.5):
    y_ = 1 * (y_pred > threshold)
    tp = true_positive(y_, y)
    precision = tp / ch.sum(y_ == 1)
    recall = tp / ch.sum(y == 1)
    f1 = (2 * precision * recall) / (precision + recall)

    precision = precision.item()
    recall = recall.item()
    f1 = f1.item()

    # Check for NaNs
    if precision != precision:
        precision = 0
    if recall != recall:
        recall = 0
    if f1 != f1:
        f1 = 0

    return (precision, recall, f1)


def epoch(model, loader, gpu, optimizer=None, verbose=False):
    loss_func = nn.CrossEntropyLoss()
    is_train = True
    if optimizer is None:
        is_train = False

    tot_loss, precision, recall, f1 = 0, 0, 0, 0
    iterator = enumerate(loader)
    if verbose:
        iterator = tqdm(iterator, total=len(loader))

    # with ch.set_grad_enabled(is_train):
    with ch.set_grad_enabled(True):
        for e, batch in iterator:

            if gpu:
                # Shift graph to GPU
                batch = batch.to('cuda')

            # Get model predictions and get loss
            labels = batch.ndata['y'].long()
            logits = model(batch)
            loss = loss_func(logits, labels)

            with ch.no_grad():
                probs = ch.softmax(logits, dim=1)[:, 1]

            # Backprop gradients if training
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Get metrics
            m = get_metrics(labels, probs)
            precision += m[0]
            recall += m[1]
            f1 += m[2]

            tot_loss += loss.detach().item()
            if verbose:
                iterator.set_description(
                    "Loss: %.5f | Precision: %.3f | Recall: %.3f | F-1: %.3f" %
                    (tot_loss / (e+1), precision / (e+1), recall / (e+1), f1 / (e+1)))
    return tot_loss / (e+1)


def train_model(net, ds, args):
    train_loader, test_loader = ds.get_loaders(args.batch_size, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.25, patience=1)

    for e in range(args.epochs):
        # Train
        print("[Train]")
        net.train()
        tr_loss = epoch(net, train_loader, args.gpu,
                        optimizer, verbose=args.verbose)

        # Test
        print("[Eval]")
        net.eval()
        epoch(net, test_loader, args.gpu, None, verbose=args.verbose)

        # Scheduler step
        scheduler.step(tr_loss)

        print()
