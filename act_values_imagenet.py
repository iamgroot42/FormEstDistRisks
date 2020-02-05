import torch as ch
import utils
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

model_arch   = sys.argv[1]
model_type   = sys.argv[2]
prefix       = sys.argv[3]

dx = utils.ImageNet1000()
ds = dx.get_dataset()

model = dx.get_model(model_type, model_arch)

batch_size = 256
all_reps = []
_, val_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True)

# Calculate emperical mean, variance to normalize representations
def get_reps(data_loader):
	for (im, label) in tqdm(data_loader):
		with ch.no_grad():
			(_, rep), _ = model(im, with_latent=True)
			all_reps.append(rep.cpu())

# Get reps for val set
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
