import torch as ch
import utils
from tqdm import tqdm

# model_type = "/p/adversarialml/as9rw/models_cifar10_vgg19/custom_adv_train_try_10.000000_100.000000_16_0.010000_3_fast_1/checkpoint.pt.best"
model_type = "/p/adversarialml/as9rw/models_cifar10_vgg/linf_16/checkpoint.pt.best"
# model_type = "nat"
dataset_name = 'cifar'
# dataset_name = 'imagenet'

if dataset_name == 'cifar':
	constants = utils.CIFAR10()
elif dataset_name == 'imagenet':
	constants = utils.ImageNet1000()

ds = constants.get_dataset()
# model = constants.get_model(model_type , "resnet50")
model = constants.get_model(model_type , "vgg19")

# eps        = 4/255
eps        = 2/255
# nb_iters   = 100
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
	# 'random_restarts': 20
}

batch_size = 256 #100#100#75
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

print("Model %s on Dataset %s with Constraint %s" % (model_type, dataset_name, constraint))

attack_x, attack_y = [], []
num_examples, asr = 0, 0
iterator = tqdm(test_loader)
for (im, _) in iterator:
	im    = im.cuda()
	label = ch.argmax(model(im)[0], 1)

	adv_logits, adv_im = model(im, label, make_adv=True, **attack_arg)
	num_examples += adv_im.shape[0]
	# Attack success rate (ASR) statistics
	adv_labels = ch.argmax(adv_logits, 1)
	asr       += ch.sum(adv_labels != label).cpu().numpy()
	iterator.set_description("Attack Success Rate : %.2f" % (100 * asr / num_examples))
