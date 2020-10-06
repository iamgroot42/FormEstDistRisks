import torch as ch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

# Custom module imports
import utils


def get_sampled_data(sample_ratio):
	# ds = utils.BinaryCIFAR("/p/adversarialml/as9rw/datasets/new_exp/small/100p_dog/").get_dataset()
	# n_classes = 2

	ds = utils.CIFAR10().get_dataset()
	ds = utils.RobustCIFAR10("/p/adversarialml/as9rw/datasets/cifar10_split2/", None).get_dataset()
	n_classes = 10

	# data_loader, _ = ds.make_loaders(batch_size=500, workers=8, shuffle_val=False)
	_, data_loader = ds.make_loaders(batch_size=500, workers=8, shuffle_val=False, only_val=True)
	dog_class = 5

	images, labels = utils.load_all_loader_data(data_loader)
	images_, labels_ = [], []
	
	# Focus on dog images
	# eligible_indices = np.nonzero(labels != dog_class)[:,0]
	look_at_these = [2,3,4,5,6,7]
	for i in look_at_these:
		eligible_indices = np.nonzero(labels == i)[:,0]
		np.random.shuffle(eligible_indices) 
		# Pick according to ratio
		picked_indices = eligible_indices[:int(len(eligible_indices) * sample_ratio)]
		images_.append(images[picked_indices])
		labels_.append(labels[picked_indices])
	
	images = ch.cat(images_)
	labels = ch.cat(labels_)

	use_ds = utils.BasicDataset(images, labels)
	return use_ds


def get_acc_on_data(model, dl):
	cor, tot        = 0, 0
	corr_no, tot_no = 0, 0
	mapping = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
	all_of_em, all_of_em_2 = [], []
	for x, y in dl:
		rep, _ = model(x.cuda(), fake_relu=False)
		# rep = ch.nn.functional.softmax(rep)
		# print(rep)
		# exit(0)
		rep = rep.detach()[:,0] # Focus on dog logits
		# rep, _ = model(x.cuda(), with_latent=True, fake_relu=False, just_latent=True)
		# rep = ch.sum(1.*(rep > 0), 1)
		
		dog   = (y == 5)
		nodog = (y != 5)
		
		for i in range(y.shape[0]): y[i] = mapping[y[i]]
		
		# Focus on dog, no_dog separately
		# cor += ch.sum(1*(preds == y.cuda()) * 1.0)
		# tot += y.shape[0]
		# cor     += ch.sum(rep[dog])
		# corr_no += ch.sum(rep[nodog])
		# tot += y[dog].shape[0]
		# tot_no += y[nodog].shape[0]
		all_of_em.append(rep[dog].cpu().numpy())
		all_of_em_2.append(rep[nodog].cpu().numpy())

	all_of_em = np.concatenate(all_of_em, 0)
	all_of_em_2 = np.concatenate(all_of_em_2, 0)
	return (np.mean(all_of_em), np.std(all_of_em)), (np.mean(all_of_em_2), np.std(all_of_em_2))
	# return cor/tot, corr_no/tot_no


if __name__ == "__main__":

	# Look at model's performance on f()=1 , then f()=0, and see if there is a distinction between the two

	# Test classifier on trained models (unseen)

	prefix = "/p/adversarialml/as9rw/new_exp_models/"
	suffix = "checkpoint.pt.best"
	constants = utils.BinaryCIFAR(None)
	paths_test = ["10p_linf", "10p_linf", "50p_linf", "50p_linf"]
	x = [0.1, 0.5]

	prefix = "/p/adversarialml/as9rw/new_exp_models/small/"
	# paths_test = [
	# 	"0p_linf", "0p_linf_2",
	# 	"10p_linf", "10p_linf_2",
	# 	"20p_linf", "20p_linf_2",
	# 	"30p_linf", "30p_linf_2",
	# 	"40p_linf", "40p_linf_2",
	# 	"50p_linf", "50p_linf_2",
	# 	"60p_linf", "60p_linf_2",
	# 	"70p_linf", "70p_linf_2",
	# 	"80p_linf", "80p_linf_2",
	# 	# "90p_linf", "90p_linf_2",
	# 	"100p_linf", "100p_linf_2"
	# ]
	paths_test = [
		"0p", "0p",
		"10p", "10p",
		"20p", "20p",
		"30p", "30p",
		"40p", "40p",
		"50p", "50p",
		"60p", "60p",
		"70p", "70p",
		"80p", "80p",
		"90p", "90p",
		"100p", "100p"
	]
	x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	# x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
	
	sample_ratio = 1.0
	use_ds = get_sampled_data(sample_ratio)

	y_1, y_2 = [], []
	z_1, z_2 = [], []

	ye_1, ye_2 = [], []
	ze_1, ze_2 = [], []

	for i, path in enumerate(paths_test):
		data_loader = DataLoader(use_ds, batch_size=1024, shuffle=False, num_workers=8)
		model = constants.get_model(os.path.join(prefix, path, suffix) , "vgg19", parallel=True)

		acc = get_acc_on_data(model, data_loader)
		if i % 2:
			# y_1.append(acc[0].item())
			# y_2.append(acc[1].item())
			y_1.append(acc[0][0])
			ye_1.append(acc[0][1])
			y_2.append(acc[1][0])
			ye_2.append(acc[1][1])
		else:
			# z_1.append(acc[0].item())
			# z_2.append(acc[1].item())
			z_1.append(acc[0][0])
			ze_1.append(acc[0][1])
			z_2.append(acc[1][0])
			ze_2.append(acc[1][1])

	import matplotlib.pyplot as plt
	import matplotlib as mpl
	mpl.rcParams['figure.dpi'] = 200

	# plt.plot(x, y_1, 'ro', label='on dog')
	# plt.plot(x, y_2, 'bo', label='on no-dog')
	# plt.plot(x, z_1, 'r+', label='on dog 2.0')
	# plt.plot(x, z_2, 'b+', label='on no-dog 2.0')

	plt.errorbar(x, y_1, yerr=ye_1, fmt='ro', label='on dog')
	plt.errorbar(x, y_2, yerr=ye_2, fmt='bo', label='on no-dog')
	plt.errorbar(x, z_1, yerr=ze_1, fmt='r+', label='on dog 2.0')
	plt.errorbar(x, z_2, yerr=ze_2, fmt='b+', label='on no-dog 2.0')


	plt.legend()
	plt.savefig("./visualize/vdirect_points.png")
