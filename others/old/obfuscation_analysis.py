import numpy as np
from tqdm import tqdm
import utils
import torch as ch
from robustness.tools.vis_tools import show_image_row
# from cleverhans.future.torch.attacks import spsa, fast_gradient_method
# from cleverhans.future.torch.attacks import hop_skip_jump_attack
# from cleverhans.future.torch.attacks import carlini_wagner_l2
from cleverhans.future.torch.attacks import sparse_l1_descent, projected_gradient_descent
# from spatial_attack import spatial_transformation_method
from robustness.tools.vis_tools import show_image_row


import sys

model_path = sys.argv[1]
model_arch = "vgg19"

dx = utils.CIFAR10()
ds = dx.get_dataset()
model = dx.get_model(model_path, model_arch)

print("Running attack on %s" % model_path)

class MadryToNormal:
	def __init__(self, model):
		self.model = model

	def __call__(self, x):
		logits, _ = self.model(x)
		return logits

batch_size = 800
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

misclass, total = 0, 0
iterator = tqdm(test_loader)
misclass_2 = 0

m = MadryToNormal(model)
for (im, label) in iterator:
	im, label = im.cuda(), label.cuda()
	# advs = fast_gradient_method(m, im.cuda(), eps=4/255, norm=np.inf, clip_min=0, clip_max=1)
	# advs = fast_gradient_method(m, im.cuda(), eps=0.5, norm=2, clip_min=0, clip_max=1)
	# advs = spsa(m, im.cuda(), 8/255, clip_min=0, clip_max=1, is_debug=False, nb_iter=100, early_stop_loss_threshold=-5, spsa_iters=2048)
	# advs = hop_skip_jump_attack(m, im.cuda(), 2, verbose=False)
	# advs = spatial_transformation_method(m, im.cuda())
	# gs = ch.Tensor([95,99,92,90]).long()
	# advs = sparse_l1_descent(m, im.cuda(), grad_sparsity=gs)
	eps = 8 / 255
	nb_iter = 500
	eps_iter = 2.5 * eps / nb_iter
	model_preds = ch.argmax(m(im), 1)
	# advs = projected_gradient_descent(m, im, eps, eps_iter, nb_iter, np.inf, clip_min=0, clip_max=1)

	attack_arg = {
		'constraint': 'inf',
		'eps':eps,
		'step_size': eps_iter,
		'iterations': nb_iter,
		'do_tqdm': False,
		'use_best': True,
		'targeted': False,
		'random_restarts': 20
	}
	adv_logits, advs = model(im, model_preds, make_adv=True, **attack_arg)
	
	# Visualize attack image
	# show_image_row([im.cpu(), advs.cpu()],
	# 				["Original Images", "Attack Images"],
	# 				fontsize=22,
	# 				filename="./visualize/obfuscation_attack.png")
	# exit(0)
	pert_labels = ch.argmax(adv_logits, 1)

	misclass   += (pert_labels != label).sum().item()
	misclass_2 += (pert_labels != model_preds).sum().item()
	total    += len(label)

	iterator.set_description('Attack Success Rate : %.2f | Misclassification Rate : %.2f' % (100 * misclass_2 / total, 100 * misclass / total))

	# show_image_row([im, advs.cpu()],
	# 				["Real Images", "Attack Images"],
	# 				fontsize=22,
	# 				filename="./visualize/spsa_attack.png")
