import dgl
import torch.nn as nn
import torch as ch
import dgl.nn.pytorch as dglnn
import torch.nn.functional as F
from data_utils import Reddit5KDataset
from dgl.dataloading import GraphDataLoader


ds = Reddit5KDataset()
data = ds.data

embed = nn.Embedding(data.number_of_nodes(), 32)
data.ndata['feat'] = embed.weight

print(data)
exit(0)

dataloader = ds.get_loader()


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


# Only an example, 7 is the input feature size
model = Classifier(7, 20, 5)
opt = ch.optim.Adam(model.parameters())
for epoch in range(20):
    for batched_graph, labels in dataloader:
        feats = batched_graph.ndata['attr']
        logits = model(batched_graph, feats)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
