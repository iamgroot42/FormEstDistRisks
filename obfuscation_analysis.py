import numpy as np
from tqdm import tqdm
import utils
import torch as ch
from robustness.tools.vis_tools import show_image_row
# from cleverhans.future.torch.attacks import spsa, fast_gradient_method
from cleverhans.future.torch.attacks import hop_skip_jump_attack
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

batch_size = 2
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

misclass, total = 0, 0
iterator = tqdm(test_loader)

m = MadryToNormal(model)
for (im, label) in iterator:
	# advs = fast_gradient_method(m, im.cuda(), eps=4/255, norm=np.inf, clip_min=0, clip_max=1)
	# advs = fast_gradient_method(m, im.cuda(), eps=0.5, norm=2, clip_min=0, clip_max=1)
	# advs = spsa(m, im.cuda(), 8/255, clip_min=0, clip_max=1, is_debug=False, nb_iter=100, early_stop_loss_threshold=-5, spsa_iters=2048)
	advs = hop_skip_jump_attack(m, im.cuda(), 2, verbose=False)
	pert_labels = ch.argmax(m(advs), 1).cpu()

	misclass += (pert_labels != label).sum().item()
	total    += len(label)

	iterator.set_description('Attack success rate : %f' % (100 * misclass / total))

	# show_image_row([im, advs.cpu()],
	# 				["Real Images", "Attack Images"],
	# 				fontsize=22,
	# 				filename="./visualize/spsa_attack.png")
