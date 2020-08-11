import utils
import torch as ch
import numpy as np


class Ensemble:
	def __init__(self, paths, constant, parallel=True):
		self.paths = paths
		self.constant = constant
		self.models = []
		self.load_models(parallel)

	def load_models(self, parallel):
		for arch, path in self.paths:
			self.models.append(constants.get_model(path , arch, parallel=parallel))

	def get_logits(self, x):
		logits = []
		with ch.no_grad():
			for m in self.models:
				logit, _ = m(x)
				logits.append(logit.cpu().numpy())
		logits = np.array(logits)
		return logits.transpose(1, 0, 2)

	def agreement(self, x, num_agree):
		# If agreement over predictions in ensemble, store prediction over which they agree
		# Otherwise  None
		logits = self.get_logits(x)
		preds = np.argmax(logits, 2)
		agreement = []
		for i, pred in enumerate(preds):
			unique, counts = np.unique(pred, return_counts=True)
			best_count = np.argmax(counts)
			if counts[best_count] >= num_agree:
				agreement.append(unique[best_count])
			else:
				agreement.append(None)
		return np.array(agreement)


if __name__ == "__main__":
	# List of tuples (model arch, model type) to load to form ensemble
	model_configs = [("vgg16", "/u/as9rw/work/fnb/vgg16_cifar/nat/checkpoint.pt.best"),
			  ("resnet18", "/u/as9rw/work/fnb/resnet18_cifar/nat/checkpoint.pt.best"),
			  ("vgg19", "nat"),
			  ("resnet50", "nat"),
			  ("densenet169", "nat")]


	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	ensemble = Ensemble(model_configs, constants)
	batch_size = 8

	_, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
	for (im, lab) in data_loader:
		logits = ensemble.get_logits(im.cuda())
		preds = ensemble.agreement(im.cuda(), 2)
		print(preds)
		break
