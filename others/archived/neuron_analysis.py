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
model.eval()

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
	'step_size': (0.5 * 2.5) / 20,
	'iterations': 20, 
	'do_tqdm': False,
	'targeted': True,
	'use_best': False
})
# L-inf attack
attack_args.append({
	'constraint':'inf',
	'eps':8/255,
	'step_size': ((8/255) * 2.5) / 20,
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
n_times = 10

batch_size = 32
all_reps = []
train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)
# Calculate emperical mean, variance to normalize representations
# Sample from train set
for (im, label) in train_loader:
	with ch.no_grad():
		(_, rep), _ = model(im.cuda(), with_latent=True)
	all_reps.append(rep)
# Sample from test set
for (im, label) in val_loader:
	with ch.no_grad():
		(_, rep), _ = model(im.cuda(), with_latent=True)
	all_reps.append(rep)

all_reps = ch.cat(all_reps)
ch_mean = ch.mean(all_reps, dim=0)
ch_std = ch.std(all_reps, dim=0)

# Re-define test loader
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True)

print("Mean:", ch_mean)
print("Std: ",   ch_std)

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
			reps = []
			for _ in range(n_times):
				# Get representation for these images
				# Use these to identify gamma-robustly useful features
				(_, rep), _ = model(adv_im, with_latent=True)
				# Normalize to zero mean, unit variance
				rep = (rep - ch_mean) / ch_std
				reps.append(rep)

			reps = ch.stack(reps).cuda()
			# Consider N binary classification tasks (1-vs-all scenarios)
			for i in range(ds.num_classes):
				binary_target_label = make_labels_binary(target_label, i, rep.shape[1])
				min_correlation, _ = ch.min(reps * binary_target_label.cuda().unsqueeze_(0).repeat(n_times, 1, 1), axis=0)
				classwise_gamma_useful[j][i] = ch.sum(min_correlation, axis=0) + classwise_gamma_useful[j].get(i, ch.zeros_like(rep[0]).cuda())

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
