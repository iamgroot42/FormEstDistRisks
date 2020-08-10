import torch as ch
import utils
import numpy as np
import os
from tqdm import tqdm


if __name__ == "__main__":
	import sys

	model_arch   = sys.argv[1]
	model_type   = sys.argv[2]
	dataset      = sys.argv[3]

	if dataset == 'cifar':
		dx = utils.CIFAR10()
	elif dataset == 'imagenet':
		dx = utils.ImageNet1000()
	elif dataset == 'svhn':
		dx = utils.SVHN10()
	elif dataset == 'binary':
		dx = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary/")
		# dx = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/cifar_binary_nodog/")
	else:
		raise ValueError("Dataset not supported")

	ds = dx.get_dataset()
	model = dx.get_model(model_type, model_arch, parallel=True)

	batch_size = 1024
	all_reps = []
	train_loader = None
	_, val_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True)
	preds = []
	truth = []
	
	for (im, label) in tqdm(val_loader):
		with ch.no_grad():
			logits, _ = model(im.cuda())
			label_ = ch.argmax(logits, 1).cpu().numpy()
			for i, l_ in enumerate(label_):
				preds.append(l_)
				truth.append(label[i].numpy())

	preds, truth = np.array(preds), np.array(truth)
	for i in range(10):
		preds_ = (preds == i)
		truth_ = (truth == i)
		print("Class %d : %.2f Accuracy" % (i, 100 * np.mean(preds_ == truth_)))
