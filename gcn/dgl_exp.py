# import torch as ch
import data_utils


# def evaluate(model, features, labels, mask):
#     model.eval()
#     with ch.no_grad():
#         logits = model(features)
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = ch.max(logits, dim=1)
#         correct = ch.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)

DATABASE = '/p/adversarialml/as9rw/datasets/'
ds = data_utils.PubMedData(DATABASE)
model = data_utils.get_model(ds)
model.cuda()

acc = data_utils.train_model(ds, 0.5, model)
print(acc)
# features = g.ndata['feat']
# labels = g.ndata['label']
# train_mask = g.ndata['train_mask']
# val_mask = g.ndata['val_mask']
# test_mask = g.ndata['test_mask']
# n_classes = data.num_labels
# n_edges = data.graph.number_of_edges()

# for epoch in range(1000):
#     model.train()
#     # forward
#     logits = model(features)
#     loss = loss_fcn(logits[train_mask], labels[train_mask])

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     acc = evaluate(model, features, labels, val_mask)
#     print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} | ". format(
#         epoch, loss.item(), acc))

# print()
# acc = evaluate(model, features, labels, test_mask)
# print("Test accuracy {:.2%}".format(acc))
