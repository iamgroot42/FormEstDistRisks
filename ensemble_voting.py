import utils
import torch as ch
import os
from tqdm import tqdm
import numpy as np
from PIL import Image 


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


def read_and_collect_images(path):
	images = []
	for x in tqdm(os.listdir(path)):
		im = np.asarray(Image.open(os.path.join(path, x))).astype(np.float32) / 255.
		images.append(im)
	return np.array(images).transpose(0, 3, 1, 2)


if __name__ == "__main__":
	# List of tuples (model arch, model type) to load to form ensemble
	model_configs = [
			("vgg16", "/u/as9rw/work/fnb/vgg16_cifar/nat/checkpoint.pt.best"),
			("resnet18", "/u/as9rw/work/fnb/resnet18_cifar/nat/checkpoint.pt.best"),
			("vgg19", "nat"),
			("resnet50", "nat"),
			("densenet169", "nat")]

	# get_images = read_and_collect_images("./cifar_linf_one_gray_0_5/")
	# get_images = read_and_collect_images("./cifar_l2_one_gray_0_5/")
	# get_images = read_and_collect_images("./cifar_nat_one_gray_0_5/")
	# get_images = read_and_collect_images("./cifar_nat_one_gray_0_1/")
	get_images = read_and_collect_images("./cifar_binary_10p_dog_linf/")
	# get_images = read_and_collect_images("./cifar_binary_50p_dog_linf/")
	images_ch = ch.from_numpy(get_images).cuda()

	constants = utils.CIFAR10()
	# ds = constants.get_dataset()
	ensemble = Ensemble(model_configs, constants)
	batch_size = 8

	preds = ensemble.agreement(images_ch, 3)
	agreement = np.mean(preds != None)
	print("Agreement %.2f" % agreement)

	agreed_labels = preds[preds != None]
	unique, counts = np.unique(agreed_labels, return_counts=True)
	mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	for (u, c) in zip(unique, counts):
		print("%s : %.2f" % (mappinf[u], 100 * c/len(agreed_labels)))

	# _, data_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, shuffle_val=False)
	# for (im, lab) in data_loader:
		# logits = ensemble.get_logits(im.cuda())
		# preds = ensemble.agreement(im.cuda(), 2)
		# print(preds)
		# break
