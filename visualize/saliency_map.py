import torch as ch
import sys
import numpy as np
from robustness.tools.vis_tools import show_image_row


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import utils

paths = [
"/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt",
"/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt",
"/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt",
"/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_10000.000000_16_0.010000_1/checkpoint.pt.best",
"/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_10000.000000_16_0.010000_1_fast_1/checkpoint.pt.best"
]

constants = utils.CIFAR10()
ds = constants.get_dataset()

def norm_grad(grad):
	grad_normed = grad
	for i, g in enumerate(grad_normed): 
		grad_normed[i] = (g - g.min())
		grad_normed[i] /= grad_normed[i].max()
		# grad_normed[i] = (g - g.min()) / (g.max() - g.min())
	return grad_normed


def get_grad(model):
	_, test_loader = ds.make_loaders(batch_size=8, workers=8, only_val=True, fixed_test_order=True)
	for i, (image, label) in enumerate(test_loader):
		image = image.cuda().requires_grad_(True)
		label = label.cuda()
		loss = ch.nn.CrossEntropyLoss()
		scores, _ = model(image)
		loss_value = loss(scores, label)
		grad = ch.autograd.grad(loss_value.mean(), [image])[0].cpu()
		return (image, norm_grad(grad))


(_, grad_norm) = get_grad(constants.get_model(paths[0], "vgg19"))
(_, grad_l2) = get_grad(constants.get_model(paths[1], "vgg19"))
(_, grad_linf) = get_grad(constants.get_model(paths[2], "vgg19"))
(image, grad_my) = get_grad(constants.get_model(paths[3], "vgg19"))
# (_, grad_my_fast) = get_grad(constants.get_model(paths[4], "vgg19"))

# show_image_row([image.detach().cpu(), grad_norm, grad_l2, grad_linf, grad_my, grad_my_fast],
show_image_row([image.detach().cpu(), grad_norm, grad_l2, grad_linf, grad_my],
				# ["Images", "Normal", "L2", "Linf", "Custom", "Custom Proper"],
				["Images", "Normal", "L2", "Linf", "Custom"],
				fontsize=22,
				filename="./saliency_maps.png")
