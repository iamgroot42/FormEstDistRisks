import utils
import torch as ch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from robustness.tools.misc import log_string
from tqdm import tqdm


# Set random seed for reproducibility
SEED = 2020
np.random.seed(SEED)
ch.manual_seed(SEED)
if ch.cuda.is_available():
	ch.cuda.manual_seed(SEED)


def ready_data(data_loader, model):
	X, Y = [], []
	with ch.no_grad():
		for (x, _) in data_loader:
			latent, _ = model(x.cuda(), with_latent=True, just_latent=True)
			X.append(latent.cpu())
			Y.append(x)
	return ch.cat(X), ch.cat(Y)


def main(n_epochs):
	# Load actual CIFAR-10 dataset
	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	# Get original model
	model = constants.get_model("linf" , "vgg19")

	# Get data loaders
	train_loader, test_loader = ds.make_loaders(batch_size=1000, workers=8, data_aug=False)

	# Accumulate model features
	print(log_string('==> Processing Data'))
	X_train, Y_train = ready_data(train_loader, model)
	X_test, Y_test   = ready_data(test_loader, model)

	# Create model
	decoder = utils.Decoder()
	# Wrap with data parallelism and params on GPU
	decoder = ch.nn.DataParallel(decoder).cuda()

	# Load data
	batch_size = 100
	trainset = utils.BasicDataset(X_train, Y_train)
	testset  = utils.BasicDataset(X_test, Y_test)
	trainloader = ch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
	testloader  = ch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)

	# autoencoder.load_state_dict(torch.load("./weights/autoencoder.pkl"))

	# Define an optimizer and criterion
	# criterion = nn.BCELoss()
	criterion = nn.MSELoss()
	optimizer = optim.Adam(decoder.parameters())

	best_val_loss = np.inf
	for epoch in range(n_epochs):
		# Train
		decoder.train()
		running_loss = 0.0
		iterator = tqdm(enumerate(trainloader, 0), total=len(X_train) // batch_size)
		for i, (x, y) in iterator:
			x, y = x.cuda(), y.cuda()

			# ============ Forward ============
			y_ = decoder(x)
			loss = criterion(y_, y)
			# ============ Backward ============
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# ============ Logging ============
			running_loss += loss.data

			print_string = log_string("[Train] Epoch : %d/%d Loss : %.4f" % (epoch + 1, n_epochs, running_loss / (i+1)), "train")
			iterator.set_description(print_string)

		# Save most recent version of model
		ch.save(decoder.state_dict(), "./decoder_weights/checkpoint.pt")

		# Validation statistics every now and then
		if epoch % 5 == 0 and epoch > 0:
			running_loss = 0.0
			decoder.eval()
			iterator = tqdm(enumerate(testloader, 0), total=len(X_test) // batch_size)
			for i, (x, y) in iterator:
				x, y = x.cuda(), y.cuda()

				# ============ Forward ============
				with ch.no_grad():
					y_ = decoder(x)
					loss = criterion(y_, y)
					running_loss += loss.data

				print_string = log_string("[Val]   Epoch : %d/%d Loss : %.4f" % (epoch + 1, n_epochs, running_loss / (i+1)), "val")
				iterator.set_description(print_string)

			# Save model if validation loss got better
			if best_val_loss > running_loss / i:
				ch.save(decoder.state_dict(), "./decoder_weights/checkpoint.best.pt")
				best_val_loss = running_loss / i


	print(log_string('Finished Training'))


if __name__ == '__main__':
	main(n_epochs = 200)
