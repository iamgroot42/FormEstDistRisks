import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
from tqdm import tqdm

# model_path   = "/p/adversarialml/as9rw/models_cifar10_vgg/delta_model.pt"
# model_path  = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt"
# model_path   = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt"
# model_path   = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt"
# model_path   = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_10000.000000_16_0.010000_1/checkpoint.pt.best"
model_path = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"

ds = CIFAR()

# Load model to attack
model_kwargs = {
	'arch': 'vgg19',
	'dataset': ds,
	'resume_path': model_path
}
model, _ = make_and_restore_model(**model_kwargs)
model.eval()

batch_size = 128

attack_x = np.load("vgg_linf_on_custom_images.npy")
attack_y = np.load("vgg_linf_on_custom_labels.npy")

accuracies = 0

for i in range(0, attack_y.shape[0], batch_size):
	im = attack_x[i:i + batch_size]
	im = ch.from_numpy(im).cuda()
	logits, _ = model(im)
	preds = ch.argmax(logits, 1).cpu().numpy()
	accuracies += np.sum(attack_y[i:i + batch_size] == preds)

print("Attack success rate : %f" % (100 * (1 - accuracies / attack_y.shape[0])))
