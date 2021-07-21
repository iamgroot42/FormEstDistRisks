import torch as ch
from robustness.datasets import GenericBinary
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys

ds_path    = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
model_path = sys.argv[1]

ds = GenericBinary(ds_path)

model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': model_path
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

def make_labels_pos_neg(labels, reps):
	return (2 * (labels == 0) - 1).unsqueeze_(-1).repeat([1, reps])

num_samples = 0
n_times = 20

batch_size = 64
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
ch_mean  = ch.mean(all_reps, dim=0)
ch_std   = ch.std(all_reps, dim=0)

p_useful = ch.zeros_like(ch_mean).cuda()
gamma_useful = {i:ch.zeros_like(ch_mean).cuda() for i in range(len(attack_args))}

# Re-define test loader
_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)

print("Mean:", ch_mean)
print("Std: ",   ch_std)

# Dump mean, std vectors for later use:
np_mean = ch_mean.cpu().numpy()
np_std  = ch_std.cpu().numpy()
np.save("feature_mean", np_mean)
np.save("feature_std",   np_std)

exit()

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
	binary_label = make_labels_pos_neg(label, rep.shape[1])
	p_useful += ch.sum(rep * binary_label.cuda(), axis=0)
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
			
			binary_target_label = make_labels_pos_neg(target_label, rep.shape[1])
			min_correlation, _  = ch.min(reps * binary_target_label.cuda().unsqueeze_(0).repeat(n_times, 1, 1), axis=0)
			gamma_useful[j]     = ch.sum(min_correlation, axis=0)

p_useful /= num_samples
for i in range(len(attack_args)):
	gamma_useful[i] /= num_samples

array_to_line = lambda x: " ".join(str(y) for y in x)

# Dump correlation values for clean examples
with open("clean.txt", 'w') as f:
	write_this = p_useful.cpu().numpy()
	f.write(array_to_line(write_this) + "\n")

# Dump correlation values per attack
for i, a in enumerate(attack_args):
	with open("l_%s.txt" % (a['constraint']), 'w') as f:
		write_this = gamma_useful[i].cpu().numpy()
		f.write(array_to_line(write_this) + "\n")
