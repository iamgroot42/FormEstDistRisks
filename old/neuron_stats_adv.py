import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

ds_path      = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
model_path   = sys.argv[1]
prefix       = sys.argv[2]
which_attack = int(sys.argv[3])

# ds = GenericBinary(ds_path)
ds = CIFAR()

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
	'use_best': False
})
# L-2 attack
attack_args.append({
	'constraint':'2',
	'eps':0.5,
	'step_size': (0.5 * 2.5) / 20,
	'iterations': 20, 
	'do_tqdm': False,
	'use_best': False
})
# L-inf attack
attack_args.append({
	'constraint':'inf',
	'eps':8/255,
	'step_size': ((8/255) * 2.5) / 20,
	'iterations': 20, 
	'do_tqdm': False,
	'use_best': False
})

n_times = 10

batch_size = 128
all_reps = []
train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)

# Pick an attack
attack_arg = attack_args[which_attack]
print(attack_arg)

# Calculate emperical mean, variance to normalize representations
def get_reps(data_loader, attack_arg, reps):
	for (im, label) in tqdm(data_loader):
		# Repeat N times (to account for randomness in attack)
		for _ in range(n_times):
			# Perform attack, get attack representation
			_, adv_im = model(im, label, make_adv=True, **attack_arg)
			with ch.no_grad():
				(_, rep), _ = model(adv_im, with_latent=True)
			reps.append(rep.cpu())

# Get reps for train ,val sets
get_reps(train_loader, attack_arg, all_reps)
print("Done with training data")
get_reps(val_loader, attack_arg, all_reps)
print("Done with testing data")

all_reps = ch.cat(all_reps)
ch_mean  = ch.mean(all_reps, dim=0)
ch_std   = ch.std(all_reps, dim=0)

print("Mean:", ch_mean)
print("Std: ",   ch_std)

# Dump mean, std vectors for later use:
np_mean = ch_mean.cpu().numpy()
np_std  = ch_std.cpu().numpy()
np.save(prefix + "feature_mean", np_mean)
np.save(prefix + "feature_std",   np_std)

# CUDA_VISIBLE_DEVICES=0 python neuron_stats_adv.py /p/adversarialml/as9rw/models_correct/normal/checkpoint.pt.latest linf_for_normal_ 0