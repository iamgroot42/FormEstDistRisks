import numpy as np
import utils
import matplotlib as mpl
import torch as ch
import seaborn as sns
mpl.rcParams['figure.dpi'] = 200
from tqdm import tqdm

paths = []
paths.append(("/p/adversarialml/as9rw/cifar10_vgg_stats/nat/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/nat/deltas.txt"))
paths.append(("/p/adversarialml/as9rw/cifar10_vgg_stats/linf/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/linf/deltas.txt"))
paths.append(("/p/adversarialml/as9rw/cifar10_vgg_stats/l2/stats/", "/p/adversarialml/as9rw/cifar10_vgg_stats/l2/deltas.txt"))
paths.append(("/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3/", "/u/as9rw/work/fnb/1e1_1e2_1e-2_16_3.txt"))

labels = ["standard", "$L_\infty$ Robust", "$L_2$ Robust", "Sensitivity"]

# Load model
constants = utils.CIFAR10()
models_to_try = ["nat", "linf", "l2"]

# Load up dataset, perform attack, note which examples seem to be the easiest to fool (small epsilon works)
ds = constants.get_dataset()
_, test_loader = ds.make_loaders(batch_size=64, workers=8, only_val=True, fixed_test_order=True)
# eps = 1 / 255
# steps = 1
# attack_arg = {
# 	'constraint':'inf',
# 	'eps': eps,
# 	'step_size': (eps * 2.5) / steps,
# 	'iterations': 100, 
# 	'do_tqdm': False,
# 	'use_best': True,
# 	'targeted': False
# }
# example_success = np.zeros((10000, ))
# tries = 10
# Look at these examples and their delta distributions for various models
# for mtype in models_to_try:
# 	base  = 0
# 	model = constants.get_model(mtype , "vgg19")
# 	for im, label in tqdm(test_loader):
# 		for i in range(tries):
# 			im, label = im.cuda(), label.cuda()
# 			# Get prediction on clean data
# 			clean_logits, _ = model(im)
# 			logits, _       = model(im, label, make_adv=True, **attack_arg)
# 			clean_preds = ch.argmax(clean_logits, dim=1)
# 			label_pred  = ch.argmax(logits, dim=1)
# 			for j in range(label.shape[0]):
# 				# Only consider examples where clean prediction is correct:
# 				if clean_preds[j] == label[j]:
# 					example_success[base + j] += 1 * (label_pred[j] != label[j].cpu().item())
# 		base += label.shape[0]
# 		if base > 300:
# 			break

# Get ordering of neurons (seemingly easiest to attack -> hardest to attack)
# fool_ratings = np.argsort(example_success)
# hardest_to_fool = fool_ratings[:10]
# easiest_to_fool = fool_ratings[-10:]
# print(hardest_to_fool, example_success[hardest_to_fool]) # 4999 6663 6664 6665 6666 6667 6668 6669 6670 6671
# print(easiest_to_fool, example_success[easiest_to_fool]) # 20  32 210 100 293 284 213  12   9 125
specific_example = 6671

for j, path in enumerate(paths):
	(mean, std) = utils.get_stats(path[0])

	# Only plot non-inf delta values
	senses = utils.get_sensitivities(path[1])[:, specific_example]

	# Look at specific example
	picked_senses = []
	for i, sense in enumerate(senses):
		if sense != np.inf: picked_senses.append((sense - mean[i]) / std[i])
	picked_senses = np.array(picked_senses)

	# Filter out np.inf values
	picked_senses[picked_senses == np.inf] = np.max(picked_senses[picked_senses != np.inf])

	# Plot delta values for this model
	ax = sns.lineplot(x=np.arange(picked_senses.shape[0]), y=np.log(np.sort(picked_senses)), label=labels[j])

# ax.set(xlim=(0,15))
ax.set_xlabel("Neurons", fontsize=15)
ax.set_ylabel("Scaled $\log(\Delta)$ Values", fontsize=15)
ax.set_title("For Example %d (hard to fool)" % (specific_example))

plt = ax.get_figure()
plt.savefig("delta_values_specific_examples.png")
