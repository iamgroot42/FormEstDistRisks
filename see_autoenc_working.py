import utils
import torch as ch
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from robustness.tools.misc import log_string
from robustness.tools.vis_tools import show_image_row


def ready_data(data_loader, model):
	X, Y = [], []
	with ch.no_grad():
		for (x, _) in data_loader:
			latent, _ = model(x.cuda(), with_latent=True, just_latent=True)
			X.append(latent.cpu())
			Y.append(x)
	return ch.cat(X), ch.cat(Y)


if __name__ == '__main__':
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
	# Load weights
	decoder.load_state_dict(ch.load("./decoder_weights/checkpoint.best.pt"))
	decoder.eval()

	with ch.no_grad():
		x, y = X_train[:32], Y_train[:32]
		recon = decoder(x.cuda()).cpu()

	show_image_row([recon, y], ["Reconstructions", "Original Images"], filename="./visualize/decoder_working.png")
	print(log_string('Reconstruction visualization complete'))
