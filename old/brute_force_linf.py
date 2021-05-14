import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
from tqdm import tqdm

# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"
model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_linf_8.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_l2_0_5.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10_vgg/cifar_nat.pt"

ds = CIFAR("/p/adversarialml/as9rw/datasets/cifar10")

# Load model to attack
model_kwargs = {
	'arch': 'vgg19',
	'dataset': ds,
	'resume_path': model_path
}
model, _ = make_and_restore_model(**model_kwargs)
model.eval()

batch_size = 1
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

def random_search_eps_range(x, eps, batch_size=1024):
	n_c, w, h = x.shape[1:]
	empty_images = ch.zeros((batch_size, n_c, w, h))
	counter = 0
	for ep in range(-eps, eps + 1):
		for i in range(n_c):
			for j in range(w):
				for k in range(h):
					empty_images[counter][i][j][k] = ep / 255
					counter += 1
					if counter == batch_size:
						send_over = empty_images.clone().cuda()
						empty_images = ch.zeros((batch_size, n_c, w, h))
						counter = 0 
						yield send_over
	yield empty_images.cuda()


eps       = 8
steps     = []
iterator  = tqdm(test_loader)
succeeded, tried = 0, 0
for (x, y) in iterator:
	x, y = x.cuda(), y.cuda()
	found_adv = False
	n_queries = 0
	tried += 1
	generator = random_search_eps_range(x, eps, 512)
	x_ = x.repeat(512, 1, 1, 1)
	while not found_adv:
		try:
			perts = next(generator)
		except StopIteration:
			break
		n_queries += perts.shape[0]
		adv_x = ch.clamp(x_ + perts, 0, 1)
		adv_logits, _ = model(adv_x)
		adv_preds = ch.argmax(adv_logits, 1)
		if ch.sum(adv_preds != y) > 0:
			found_adv = True
	if found_adv:
		steps.append(n_queries)
		succeeded += 1
	iterator.set_description("Average Trials per Image : %.2f | ASR : %.2f" % (sum(steps) / len(steps), 100 * succeeded /tried))
