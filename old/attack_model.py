import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
from tqdm import tqdm

model_path = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt"

ds = CIFAR()

# Load model to attack
model_kwargs = {
	'arch': 'vgg19',
	'dataset': ds,
	'resume_path': model_path
}
model, _ = make_and_restore_model(**model_kwargs)
model.eval()

eps        = 0.65/255
nb_iters   = 20
constraint = 'inf'

attack_arg = {
	'constraint': constraint,
	'eps':eps,
	'step_size': (eps * 2.5) / nb_iters,
	'iterations': nb_iters,
	'do_tqdm': False,
	'use_best': True,
	'targeted': False,
	'random_restarts': 20
}

batch_size = 2000
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

save_data = False

attack_x, attack_y = [], []
num_examples, asr = 0, 0
iterator = tqdm(test_loader)
for (im, label) in iterator:
	im, label = im.cuda(), label.cuda()
	adv_logits, adv_im = model(im, label, make_adv=True, **attack_arg)
	num_examples += adv_im.shape[0]
	# Attack success rate (ASR) statistics
	adv_labels = ch.argmax(adv_logits, 1)
	asr       += ch.sum(adv_labels != label).cpu().numpy()
	iterator.set_description("Attack success rate : %.2f" % (100 * asr / num_examples))
	if save_data:
		attack_x.append(adv_im.cpu())
		attack_y.append(label.cpu())

if save_data:
	attack_x = ch.cat(attack_x, 0).numpy()
	attack_y = ch.cat(attack_y, 0).numpy()

	np.save("vgg_linf_on_custom_images", attack_x)
	np.save("vgg_linf_on_custom_labels", attack_y)
