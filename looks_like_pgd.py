import os
import torch as ch
from robustness.tools.vis_tools import show_image_row
import numpy as np
import sys
from tqdm import tqdm
from torch.autograd import Variable
import imageio
import utils


def fast_gradient_method(model_fn, x, y, targeted=False):
	x = x.clone().detach().to(ch.float).requires_grad_(True)
  
	loss_fn = ch.nn.CrossEntropyLoss()
	loss = loss_fn(model_fn(x)[0], y)
	if targeted:
		loss = -loss

	loss.backward()
	attack_lr = 5e1#2e1
	optimal_perturbation = attack_lr * x.grad

	adv_x = x + optimal_perturbation
	adv_x = ch.clamp(adv_x, 0, )

	return adv_x


def projected_gradient_descent(model_fn, x, y, nb_iter, targeted):
	eta = ch.zeros_like(x)

	adv_x = x + eta
	adv_x = ch.clamp(adv_x, 0, 1)

	i = 0
	in_process_images = []
	while i < nb_iter:
		adv_x = fast_gradient_method(model_fn, adv_x, y, targeted)

		eta = adv_x - x
		adv_x = x + eta

		adv_x = ch.clamp(adv_x, 0, 1)
		i += 1

		# print(adv_x.shape)
		adv_x_ = adv_x.clone().detach().cpu().numpy().transpose(0, 2, 3, 1)
		adv_x_ = np.concatenate(adv_x_, 0).transpose(2, 0, 1)
		in_process_images.append(adv_x_)    

		# np_image = inp.data.detach().cpu().numpy()
		# np_image = np_image.transpose(0, 2, 3, 1)
		# retained.append(np.concatenate(np_image, 0))

		# in_process_images.append(adv_x.clone().detach().cpu())

	# return ch.cat(in_process_images, 0)
	return in_process_images



if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--model_type', type=str, default='linf', help='type of model (nat/l2/linf)')
	parser.add_argument('--iters', type=int, default=500, help='number of iterations')
	parser.add_argument('--bs', type=int, default=32, help='batch size while performing attack')
	parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: one of [binarycifar10, cifar10, imagenet]')
	
	args = parser.parse_args()
	for arg in vars(args):
		print(arg, " : ", getattr(args, arg))
	
	model_arch      = args.model_arch
	model_type      = args.model_type
	batch_size      = args.bs
	iters           = args.iters
	fake_relu       = (model_arch != 'vgg19')

	# Load model
	if args.dataset == 'cifar10':
		constants = utils.CIFAR10()
	elif args.dataset == 'imagenet':
		constants = utils.ImageNet1000()
	else:
		print("Invalid Dataset Specified")
	ds = constants.get_dataset()

	# Load model
	model = constants.get_model(model_type , model_arch)
	
	index_focus  = 29#2,6
	
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

	index_base, asr = 0, 0
	norm_count = 0

	i_want_this = 0
	
	iterator = tqdm(enumerate(test_loader))
	for num, (image, label) in iterator:
		if num < i_want_this:
			continue

		for i in range(10):
			image[i] = image[index_focus]

		y = ch.arange(0, 10, dtype=ch.long).cuda()
		
		wanted_images = projected_gradient_descent(model, image[:10].cuda(), y, iters, targeted=True)
		wanted_images = np.array(wanted_images).transpose(0, 2, 3, 1)

		# Save GIF
		print("==> Generating GIF")
		# basic_image = np.concatenate(image.cpu().numpy().transpose(0, 2, 3, 1), 0)
		# retained = [np.concatenate([x, basic_image], 1) for x in retained]
		# Scale to [0, 255] with uint8
		# print(wanted_images.shape)
		# exit(0)
		retained = [(255 * x).astype(np.uint8) for x in wanted_images]
		imageio.mimsave('./visualize/pgd_basic_deltas.gif', retained)

		exit(0)
