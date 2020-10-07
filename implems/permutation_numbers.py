import torch as ch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

import utils


def train_as_they_said(model, trainloader, testloader, loss_fn, epochs=40):
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

	for _ in range(epochs):
		# Train
		running_loss = 0.0
		num_samples = 0
		model.train()
		iterator = tqdm(trainloader)
		for (x, y) in iterator:

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(x)
			loss = loss_fn(outputs, y)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

			num_samples += y.shape[0]
			iterator.set_description("[Train] Loss: %.3f" % running_loss / num_samples)

		# Validation
		model.eval()
		running_loss = 0.0
		num_samples = 0
		for (x, y) in testloader:

			outputs = model(x)
			loss = loss_fn(outputs, y)
			running_loss += loss.item()
			num_samples += y.shape[0]

		print("[Val] Loss: %.3f" % running_loss / num_samples)


if __name__ == "__main__":
	ci = utils.CensusIncome("./census_data/")
	(x_tr, y_tr), (x_te, y_te) = ci.load_data()

	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(max_depth=10, random_state=0)
	clf.fit(x_tr, y_tr)
	print(clf.score(x_tr, y_tr))
	print(clf.score(x_te, y_te))

	model = utils.MLP(x_te.shape[1])
	trainloader = DataLoader(TensorDataset(ch.Tensor(x_tr), ch.Tensor(y_tr)), batch_size=128)
	testloader  = DataLoader(TensorDataset(ch.Tensor(x_te), ch.Tensor(y_te)), batch_size=128)

	loss_fn = nn.CrossEntropyLoss()
	train_as_they_said(model, trainloader, testloader, loss_fn)


	# constants = utils.Celeb()
	# ds = constants.get_dataset()
	# train_loader, val_loader = ds.make_loaders(batch_size=64, workers=8, shuffle_val=False, only_val=True)