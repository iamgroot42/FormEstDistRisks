import torch as ch
import sys
import numpy as np
from robustness.tools.vis_tools import show_image_row


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
import utils

constants = utils.CIFAR10()
data_constants = utils.RobustCIFAR10("/p/adversarialml/as9rw/generated_images_binary_dog50p/", None)
ds = data_constants.get_dataset()

def norm_grad(grad):
	grad_normed = grad
	for i, g in enumerate(grad_normed): 
		grad_normed[i] = (g - g.min())
		grad_normed[i] /= grad_normed[i].max()
		# grad_normed[i] = (g - g.min()) / (g.max() - g.min())
	return grad_normed


def get_grad(model):
	_, test_loader = ds.make_loaders(batch_size=8, workers=8, only_val=True, shuffle_val=False)
	for i, (image, label) in enumerate(test_loader):
		if i < 15: continue
		image = image.cuda().requires_grad_(True)
		label = label.cuda()
		loss = ch.nn.CrossEntropyLoss()
		scores, _ = model(image)
		loss_value = loss(scores, label)
		grad = ch.autograd.grad(loss_value.mean(), [image])[0].cpu()
		return (image, norm_grad(grad))


(_, grad_norm) = get_grad(constants.get_model("nat", "vgg19"))
(_, grad_l2) = get_grad(constants.get_model("l2", "vgg19"))
(image, grad_linf) = get_grad(constants.get_model("linf", "vgg19"))

# show_image_row([image.detach().cpu(), grad_norm, grad_l2, grad_linf, grad_my, grad_my_fast],
show_image_row([image.detach().cpu(), grad_norm, grad_l2, grad_linf],
				["Images", "Normal", "L2", "Linf"],
				fontsize=22,
				filename="./saliency_maps.png")
