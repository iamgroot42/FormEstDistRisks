import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
from tqdm import tqdm


# model_path = "/p/adversarialml/as9rw/models_correct/normal/checkpoint.pt.latest"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_10000.000000_16_0.010000_1/checkpoint.pt.best"
model_path = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt"

# ds = GenericBinary(ds_path)
ds = CIFAR()

# Load model to attack
model_kwargs = {
	'arch': 'vgg19',
	'dataset': ds,
	'resume_path': model_path
}
model, _ = make_and_restore_model(**model_kwargs)
model.eval()

# attack_arg = {
# 	'constraint': '2',
# 	'eps':0.5,
# 	'step_size': (0.5 * 2.5) / 100,
# 	'iterations': 100,
# 	'do_tqdm': False,
# 	'use_best': True,
# 	'targeted': False,
# 	'random_restarts': 20
# }
attack_arg = {
	'constraint':'inf',
	'eps': 8/255,
	'step_size': ((8 /255) * 2.5) / 100,
	'iterations': 100, 
	'do_tqdm': False,
	'use_best': True,
	'targeted': False,
	'random_restarts': 20
}

batch_size = 512
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

attack_x, attack_y = [], []
for (im, label) in tqdm(test_loader):
	_, adv_im = model(im, label, make_adv=True, **attack_arg)
	attack_x.append(adv_im.cpu())
	attack_y.append(label.cpu())

attack_x = ch.cat(attack_x, 0).numpy()
attack_y = ch.cat(attack_y, 0).numpy()

np.save("vgg_linf_on_custom_images", attack_x)
np.save("vgg_linf_on_custom_labels", attack_y)
