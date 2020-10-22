import numpy as np
import utils
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import os

from facenet_pytorch import InceptionResnetV1, MTCNN


def get_labels(dataloader):
	labels = []
	for (_, y) in tqdm(dataloader, total=len(dataloader)):
		labels.append(y)
	return np.concatenate(labels)


def filter(indices, value, ratio):
	qi    = np.nonzero(indices == value)[0]
	notqualify = np.nonzero(indices != value)[0]
	np.random.shuffle(notqualify)
	nqi = notqualify[:int(((1-ratio) * len(qi))/ratio)]
	return np.sort(np.concatenate((qi, nqi)))


def dump_files(path, model, dataloader, indices, target_prop, save_raw=False):
	# Make directories for classes (binary)
	os.mkdir(os.path.join(path, "0"))
	os.mkdir(os.path.join(path, "1"))
	trace, start = 0, 0
	for i, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
		y_ = y[:, target_prop]
		if save_raw:
			x_ = x.numpy()
		else:
			x_ = model(x.cuda()).cpu().detach().numpy()
		for j in range(y_.shape[0]):
			if start + j == indices[trace]:
				if y_[j] == 0:
					if save_raw:
						image = Image.fromarray((255 * np.transpose(x_[j], (1, 2, 0))).astype('uint8'))
						try:
							model(image, save_path=os.path.join(path, "0", str(trace)) + ".png")
						except:
							# Side view, no face to look at
							print("Class 0: Problematic image!")
					else:
						np.save(os.path.join(path, "0", str(trace)), x_[j])
				else:
					if save_raw:
						image = Image.fromarray((255 * np.transpose(x_[j], (1, 2, 0))).astype('uint8'))
						try:
							model(image, save_path=os.path.join(path, "1", str(trace)) + ".png")
						except:
							# Side view, no face to look at
							print("Class 1: Problematic image!")
						# image.save(os.path.join(path, "1", str(trace)) + ".png")
					else:
						np.save(os.path.join(path, "1", str(trace)), x_[j])
				trace += 1
				# If run through all indices there are in data, stop processing
				if trace == len(indices): return
		start += y.shape[0]


if __name__ == "__main__":

	constants = utils.Celeb()
	ds = constants.get_dataset()

	trainloader, testloader = ds.make_loaders(batch_size=256, workers=8, shuffle_train=False, shuffle_val=False)
	attrs = constants.attr_names
	print(attrs)
	# prop = attrs.index("Attractive")
	# prop = attrs.index("Male")
	# prop = attrs.index("Young")

	# target_prop = attrs.index("Smiling")
	target_prop = attrs.index("Male")
	labels_tr = get_labels(trainloader)
	labels_te = get_labels(testloader)
	# tags_tr = labels_tr[:, prop]
	# tags_te = labels_te[:, prop]
	
	# print("Original property ratio:", np.mean(tags_tr))
	print("Original label balance:", np.mean(labels_tr[:, target_prop]))
	# No filter (all data)
	picked_indices_tr = np.arange(labels_tr.shape[0])
	picked_indices_te = np.arange(labels_te.shape[0])
	# Attractive
	# picked_indices_tr = filter(tags_tr, 1, 0.68)
	# picked_indices_te = filter(tags_te, 1, 0.68)
	# Male
	# picked_indices_tr = filter(tags_tr, 1, 0.59)
	# picked_indices_te = filter(tags_te, 1, 0.59)
	# Old
	# picked_indices_tr = filter(tags_tr, 0, 0.37)
	# picked_indices_te = filter(tags_te, 0, 0.37)
	# print("Filtered property ratio:", np.mean(tags_tr[picked_indices_tr]))
	print("Filtered data label balance:", np.mean(labels_tr[picked_indices_tr, target_prop]))

	# CelebA dataset
	# model = utils.FaceModel(512, only_prep=True).cuda()
	# model = nn.DataParallel(model)

	model = MTCNN(device='cuda')

	# Get loaders again
	trainloader, testloader = ds.make_loaders(batch_size=512, workers=8, shuffle_train=False, shuffle_val=False, data_aug=False)
	# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_all"
	path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/male_all"
	# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_attractive"
	# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_male"
	# path = "/p/adversarialml/as9rw/datasets/celeba_raw_crop/smile_old"
	# path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_attractive_vggface"
	# path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_male"
	# path = "/p/adversarialml/as9rw/datasets/celeba_process_vggface1/smile_old"
	# Save test data
	os.mkdir(os.path.join(path, "test"))
	dump_files(os.path.join(path, "test"), model, testloader, picked_indices_te, target_prop, save_raw=True)
	# Save train data
	os.mkdir(os.path.join(path, "train"))
	dump_files(os.path.join(path, "train"), model, trainloader, picked_indices_tr, target_prop, save_raw=True)
