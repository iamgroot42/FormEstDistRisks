import numpy as np
import torch as ch
from tqdm import tqdm
import utils
import os
from PIL import Image


if __name__ == "__main__":
	import sys
	folder_to_read = sys.argv[1]
	model_type     = sys.argv[2]
	
	cifar_constant = utils.CIFAR10()
	model = cifar_constant.get_model(model_type , "vgg19", parallel=True)
	# model = cifar_constant.get_model("nat" , "vgg19", parallel=True)
	# model = constants.get_model("nat", "vgg19", parallel=True)
	# model = constants.get_model("nat" , "resnet50", parallel=True)
	

	images = []
	# Read images in folder
	for impath in os.listdir(folder_to_read):
		loaded_image = np.asarray(Image.open(os.path.join(folder_to_read, impath))).astype('float32') / 255
		loaded_image = np.transpose(loaded_image, (2, 0, 1))
		images.append(loaded_image)

	images_cuda = ch.from_numpy(np.array(images)).cuda()

	# Get dataset
	classwise_distr = np.zeros((10,))
	logits, _ = model(images_cuda)
	preds = ch.argmax(logits, 1)
	for p in preds:
		classwise_distr[p] += 1
	
	mappinf = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	classwise_distr /= classwise_distr.sum()
	
	for i, x in enumerate(mappinf):
		print("%s : %.3f" % (x, classwise_distr[i] * 100))
