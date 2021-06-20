from dgl.nn.pytorch import GraphConv
import torch.nn.functional as F
import torch.nn as nn
import torch as ch
import tqdm as tqdm


# Start off with standard GCN -> worry about Random Walk later
class GCN(nn.Module):
    def __init__(self, ds,
                 n_hidden, n_layers,
                 dropout):
        super(GCN, self).__init__()
        # Get actual graph
        self.g = ds.g

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(ds.num_features, n_hidden, activation=F.relu))

        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=F.relu))

        # output layer
        self.layers.append(GraphConv(n_hidden, ds.num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, latent=None):
        if latent is not None:
            if latent < 0 or latent > len(self.layers):
                raise ValueError("Invald interal layer requested")

        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            if i == latent:
                return h
        return h


if __name__ == "__main__":
    from botdet.data.dataset_botnet import BotnetDataset
    from botdet.data.dataloader import GraphDataLoader

    BASE_DATA_DIR = "/p/adversarialml/as9rw/datasets/botnet"
    dataset_name = "chord"

    botnet_dataset_test = BotnetDataset(
        name=dataset_name, root=BASE_DATA_DIR,
        add_nfeat_ones=True, in_memory=True,
        split='test', graph_format='dgl')
    print("Loaded test data")

    test_loader = GraphDataLoader(
        botnet_dataset_test, batch_size=1, shuffle=False, num_workers=0)

    # model = GCN()
    loss_func = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # model.train()

    epoch_losses = []
    for epoch in range(80):
        epoch_loss = 0
        for iter, (bg, label) in enumerate(test_loader):
            print(label)
            exit(0)
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
        epoch_losses.append(epoch_loss)