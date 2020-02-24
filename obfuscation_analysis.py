import numpy as np
from tqdm import tqdm
import utils
import torch as ch
from robustness.tools.vis_tools import show_image_row
from cleverhans.future.torch.attacks import spsa
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

batch_size = 256
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

misclass, total = 0, 0
iterator = tqdm(test_loader)

m = MadryToNormal(model)
for (im, label) in iterator:
	spsa_advs = spsa(m, im.cuda(), 8/255, clip_min=0, clip_max=1, is_debug=False, nb_iter=100, early_stop_loss_threshold=-5, spsa_iters=2048)
	pert_labels = ch.argmax(m(spsa_advs), 1).cpu()

	misclass += (pert_labels != label).sum().item()
	total    += len(label)

	iterator.set_description('Attack success rate : %f' % (misclass / total))

	# show_image_row([im, spsa_advs.cpu()],
	# 				["Real Images", "Attack Images"],
	# 				fontsize=22,
	# 				filename="./visualize/spsa_attack.png")
