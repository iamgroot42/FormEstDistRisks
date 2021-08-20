import torch.nn as nn
import torch.optim as optim
import torch as ch
from tqdm import tqdm
import os
import copy
import dgl.function as fn
from dgl.utils import expand_as_pair
from torch.nn import init
import numpy as np
from utils import get_weight_layers


# BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_botnet/"
BASE_MODELS_DIR = "/p/adversarialml/as9rw/models_botnet_new/"


class RightNormGraphConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(RightNormGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.weight = nn.Parameter(ch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(ch.Tensor(out_feats))

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        init.zeros_(self.bias)

    def forward(self, graph, feat):
        with graph.local_scope():
            if (graph.in_degrees() == 0).any():
                raise ValueError("Graph has 0-in-degree nodes")
            aggregate_fn = fn.copy_src('h', 'm')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # normalize node representations before parsing
            degs = graph.out_degrees().float().clamp(min=1)
            norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat_dst.dim() - 1)
            norm = ch.reshape(norm, shp)

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                feat_src = ch.matmul(feat_src, self.weight)

                # normalize with out-degree
                feat_src *= norm

                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # normalize with out-degree
                feat_src *= norm

                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                rst = ch.matmul(rst, self.weight)

            # Add bias term
            rst = rst + self.bias

            return rst


class GCN(nn.Module):
    def __init__(self, n_hidden, n_layers,
                 n_inp=1, n_classes=2, residual=True, dropout=0.5):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.residual = residual
        self.drop_p = dropout

        # input layer
        self.layers.append(
            RightNormGraphConv(n_inp, n_hidden))

        # hidden layers
        for i in range(n_layers-1):
            self.layers.append(
                RightNormGraphConv(n_hidden, n_hidden))

        # output layer
        self.final = RightNormGraphConv(n_hidden, n_classes)
        self.activation = nn.ReLU()
        if self.drop_p > 0:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, latent=None):

        if latent is not None:
            if latent < 0 or latent > len(self.layers):
                raise ValueError("Invald interal layer requested")

        x = g.ndata['x']
        for i, layer in enumerate(self.layers):
            xo = layer(g, x)

            if self.drop_p > 0:
                xo = self.dropout(xo)

            # Add prev layer directly, if requested
            if self.residual and i != 0:
                xo += x

            x = self.activation(xo)

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

    n_samples, tot_loss, precision, recall, f1 = 0, 0, 0, 0, 0
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
            samples_now = batch.batch_size
            m = get_metrics(labels, probs)
            precision += m[0] * samples_now
            recall += m[1] * samples_now
            f1 += m[2] * samples_now
            n_samples += samples_now

            tot_loss += loss.detach().item() * samples_now
            if verbose:
                iterator.set_description(
                    "Loss: %.5f | Precision: %.3f | Recall: %.3f | F-1: %.3f" %
                    (tot_loss / n_samples, precision / n_samples, recall / n_samples, f1 / n_samples))
    return tot_loss / n_samples, f1 / n_samples


def train_model(net, ds, args):
    train_loader, test_loader = ds.get_loaders(args.batch_size, shuffle=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.25, patience=1)

    best_model, best_f1, best_f1_val = None, 0, 0
    iterator = range(args.epochs)
    if not args.verbose:
        iterator = tqdm(iterator)
    for e in iterator:

        if args.verbose:
            print("Epoch %d/%d" % (e + 1, args.epochs))
            print("[Train]")
        # Train
        net.train()
        tr_loss, tr_f1 = epoch(net, train_loader, args.gpu,
                               optimizer, verbose=args.verbose)

        # Test
        if args.verbose:
            print("[Eval]")
        net.eval()
        te_loss, te_f1 = epoch(net, test_loader, args.gpu,
                               None, verbose=args.verbose)

        # Scheduler step
        scheduler.step(tr_loss)

        if args.verbose:
            print()
        else:
            iterator.set_description(
                "[Train] Loss: %.3f, F-1: %.3f | [Test] Loss: %.3f, F-1: %.3f" %2
                (tr_loss, tr_f1, te_loss, te_f1))

        # Keep track of best performing model
        if args.best_model and tr_f1 > best_f1:
            best_model = copy.deepcopy(net)
            best_f1 = tr_f1
            best_f1_val = te_f1

    if args.best_model:
        return best_model, (best_f1, best_f1_val)

    return net, (tr_f1, te_f1)


def save_model(model, split, prop_val, name, prefix=None):
    if prefix is None:
        prefix = BASE_MODELS_DIR
    savepath = os.path.join(split, str(prop_val), name)
    ch.save(model.state_dict(), os.path.join(prefix, savepath))


def get_model(args):
    model = GCN(n_inp=args.n_feat, n_hidden=args.hidden_channels,
                n_layers=args.num_layers, dropout=args.dropout,
                residual=True)
    if args.gpu:
        model = model.cuda()
    return model


def get_model_features(model_dir, args, max_read=None,
                       residual_modification=False):
    vecs = []
    iterator = os.listdir(model_dir)
    if max_read is not None:
        np.random.shuffle(iterator)
        iterator = iterator[:max_read]

    for mpath in tqdm(iterator):
        # Define model
        model = get_model(args)

        # Load weights into model
        if args.gpu:
            model.load_state_dict(ch.load(os.path.join(model_dir, mpath)))
        else:
            model.load_state_dict(ch.load(os.path.join(
                model_dir, mpath), map_location=ch.device('cpu')))
        model.eval()

        # Extract model weights
        dims, fvec = get_weight_layers(
            model, transpose=False, first_n=args.first_n,
            start_n=args.start_n)

        if residual_modification:
            extras = []
            for i, fv in enumerate(fvec):
                if i == 0:
                    extras.append(None)
                    continue
                pseudo_w = 4 * ch.from_numpy(np.linalg.pinv(fv[:, :-1].numpy()))
                pseuso_fv = ch.ones_like(fv)
                pseuso_fv[:, :-1] = pseudo_w
                extras.append(pseuso_fv)

            # Try to handle residual connection
            # Skip first layer
            FVEC, DIMS = [], []
            for i, (fv, fv_) in enumerate(zip(fvec, extras)):
                if i > 0:
                    FVEC.append(fv_)
                FVEC.append(fv)

            for fv in FVEC:
                DIMS.append(fv.shape[1] - 1)

            dims, fvec = DIMS, FVEC

        vecs.append(fvec)

    return dims, vecs
