import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import utils


def train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn, epochs=40):
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

	for _ in range(epochs):
		# Train
		running_loss, running_acc = 0.0, 0.0
		num_samples = 0
		model.train()
		iterator = tqdm(trainloader)
		for (x, y) in iterator:
			x, y = x.cuda(), y.cuda()
			# Smile prediction
			y = y[:, 31:32]

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(x)
			loss = loss_fn(outputs, y.float())
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			running_acc  += acc_fn(outputs, y)
			num_samples += x.shape[0]

			iterator.set_description("[Train] Loss: %.3f Accuacy: %.2f" % (running_loss / num_samples, 100 * running_acc / num_samples))

		# Validation
		model.eval()
		running_loss, running_acc = 0.0, 0.0
		num_samples = 0
		for (x, y) in testloader:
			x, y = x.cuda(), y.cuda()
			# Smile prediction
			y = y[:, 31:32]

			outputs = model(x)
			loss = loss_fn(outputs, y.float())
			running_loss += loss.item()
			running_acc  += acc_fn(outputs, y)
			num_samples += x.shape[0]

		print("[Val] Loss: %.3f Accuacy: %.2f" % (running_loss / num_samples, 100 * running_acc / num_samples))


if __name__ == "__main__":
	# Census Income dataset
	# ci = utils.CensusIncome("./census_data/")
	# (x_tr, y_tr), (x_te, y_te) = ci.load_data()
	# clf = RandomForestClassifier(max_depth=10, random_state=0)
	# clf = MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=40)
	# clf.fit(x_tr, y_tr)
	# print(clf.score(x_tr, y_tr))
	# print(clf.score(x_te, y_te))

	# model = utils.MLP(x_te.shape[1]).cuda()
	# trainloader = DataLoader(TensorDataset(ch.Tensor(x_tr), ch.Tensor(y_tr)), batch_size=4096)
	# testloader  = DataLoader(TensorDataset(ch.Tensor(x_te), ch.Tensor(y_te)), batch_size=4096)

	# loss_fn = nn.BCELoss()
	# acc_fn = lambda outputs, y: ch.sum((y == (outputs >= 0.5)))
	# train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn)

	# CelebA dataset
	model = utils.FaceModel(512).cuda()
	ds = utils.Celeb().get_dataset()
	trainloader, testloader = ds.make_loaders(batch_size=1024, workers=8)

	loss_fn = nn.BCELoss()
	acc_fn = lambda outputs, y: ch.sum((y == (outputs >= 0.5)))
	train_as_they_said(model, trainloader, testloader, loss_fn, acc_fn)
