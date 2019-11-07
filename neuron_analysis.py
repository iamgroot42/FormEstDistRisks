import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys

ds = CIFAR()

model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': sys.argv[1]
}

model, _ = make_and_restore_model(**model_kwargs)

attack_args = []
# L-1 (SLIDE) attack
attack_args.append({
	'constraint':'1',
	'eps':10,
	'step_size': 1,
	'iterations': 20, 
	'do_tqdm': False,
	'targeted': True,
	'use_best': False
})
# L-2 attack
attack_args.append({
	'constraint':'2',
	'eps':0.5,
	'step_size': 0.5 * 2.5 / 20,
	'iterations': 20, 
	'do_tqdm': False,
	'targeted': True,
	'use_best': False
})
# L-inf attack
attack_args.append({
	'constraint':'inf',
	'eps':8/255,
	'step_size': 8/255 * 2.5 / 20,
	'iterations': 20, 
	'do_tqdm': False,
	'targeted': True,
	'use_best': False
})

def make_labels_binary(labels, target_class, reps):
	return (2 * (labels == target_class) - 1).unsqueeze_(-1).repeat([1, reps])

classwise_p_useful = {}
classwise_gamma_useful = {i:{} for i in range(len(attack_args))}
num_samples = 0

precompute_bs = 128
_, test_loader = ds.make_loaders(batch_size=precompute_bs, workers=8, only_val=True)
all_means, all_vars = [], []
num_batches = 0
# Calculate emperical mean, variance to normalize representations (before calculating correlations)
for (im, label) in test_loader:
	with ch.no_grad():
		(_, rep), _ = model(im.cuda(), with_latent=True)
	rep_cpu = rep.cpu().numpy()
	all_means.append(np.mean(rep_cpu, axis=0))
	all_vars.append(np.var(rep_cpu, axis=0))
	num_batches += 1


# Re-define test loader
_, test_loader = ds.make_loaders(batch_size=128, workers=8, only_val=True)

# Combine their variations together using this formula : https://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
all_var  = np.sum(all_means, axis=0)
all_var *= ((num_batches - 1) * precompute_bs)/(precompute_bs - 1)
all_var += np.sum(all_vars, axis=0)
all_var *= (precompute_bs - 1) / (precompute_bs * num_batches - 1)
all_mean = np.mean(all_means, axis=0)
all_std  = np.sqrt(all_var)
# Convert to Pytorch tensors
ch_mean = ch.from_numpy(all_mean).cuda()
ch_std  = ch.from_numpy(all_std).cuda()

for (im, label) in test_loader:
	num_samples += im.shape[0]
	target_label = (label + ch.randint_like(label, high=ds.num_classes - 1)) % ds.num_classes
	# Take note of representation on unperturbed image
	# Use this to identify p-useful features
	with ch.no_grad():
		(_, rep), _ = model(im.cuda(), with_latent=True)
		# Normalize to zero mean, unit variance
		rep = (rep - ch_mean) / ch_std

	# Consider N binary classification tasks (1-vs-all scenarios)
	for i in range(ds.num_classes):
		binary_label = make_labels_binary(label, i, rep.shape[1])
		classwise_p_useful[i] = ch.sum(rep * binary_label.cuda(), axis=0) + classwise_p_useful.get(i, ch.zeros_like(rep[0]).cuda())
	for j, attack_arg in enumerate(attack_args):
		# Get perturbed images
		adv_out, adv_im = model(im, target_label, make_adv=True, **attack_arg)
		with ch.no_grad():
			# Get representation for these images
			# Use these to identify gamma-robustly useful features
			(_, rep), _ = model(adv_im, with_latent=True)
			# Normalize to zero mean, unit variance
			rep = (rep - ch_mean) / ch_std

			# Consider N binary classification tasks (1-vs-all scenarios)
			for i in range(ds.num_classes):
				binary_target_label = make_labels_binary(target_label, i, rep.shape[1])
				classwise_gamma_useful[j][i] = ch.sum(rep * binary_target_label.cuda(), axis=0) + classwise_gamma_useful[j].get(i, ch.zeros_like(rep[0]).cuda())

for i in range(ds.num_classes):
	classwise_p_useful[i]     /= num_samples
	for j in range(len(attack_args)):
		classwise_gamma_useful[j][i] /= num_samples

array_to_line = lambda x: " ".join(str(y) for y in x)

# Dump correlation values for clean examples
with open("clean.txt", 'w') as f:
	for i in range(ds.num_classes):
		write_this = classwise_p_useful[i].cpu().numpy()
		f.write(array_to_line(write_this) + "\n")

# Dump correlation values per attack
for i, a in enumerate(attack_args):
	with open("l_%s.txt" % (a['constraint']), 'w') as f:
		for j in range(ds.num_classes):
			write_this = classwise_gamma_useful[i][j].cpu().numpy()
			f.write(array_to_line(write_this) + "\n")
