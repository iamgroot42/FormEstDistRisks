import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

ds_path      = "/p/adversarialml/as9rw/datasets/cifar_binary/animal_vehicle_correct"
model_path   = sys.argv[1]
prefix       = sys.argv[2]

# ds = GenericBinary(ds_path)
ds = CIFAR()

model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': model_path
}

model, _ = make_and_restore_model(**model_kwargs)
model.eval()

batch_size = 128
all_reps = []
train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)

# Calculate emperical mean, variance to normalize representations
def get_reps(data_loader):
	for (im, label) in tqdm(data_loader):
		with ch.no_grad():
			(_, rep), _ = model(im, with_latent=True)
			all_reps.append(rep.cpu())

# Get reps for train ,val sets
get_reps(train_loader)
print("Done with training data")
get_reps(val_loader)
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

# CUDA_VISIBLE_DEVICES=0 python neuron_stats_adv.py /p/adversarialml/as9rw/models_correct/normal/checkpoint.pt.latest linf_for_normal_