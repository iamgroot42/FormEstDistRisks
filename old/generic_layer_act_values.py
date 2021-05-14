import torch as ch
import utils
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

model_arch   = "vgg19"
model_type   = sys.argv[1]
prefix       = sys.argv[2]
feature      = int(sys.argv[3])

dx = utils.CIFAR10()
ds = dx.get_dataset()

model = dx.get_model(model_type, model_arch)

batch_size = 512
all_reps = []
train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)

# Calculate emperical mean, variance to normalize representations
def get_reps(data_loader):
	for (im, label) in tqdm(data_loader):
		with ch.no_grad():
			rep, _ = model(im, this_layer_output=feature, just_latent=True)
			# Flatten out rep
			all_reps.append(rep.cpu())

# Get reps for train ,val sets
get_reps(train_loader)
print("Done with training data")
get_reps(val_loader)
print("Done with testing data")

all_reps = ch.cat(all_reps)
ch_mean  = ch.mean(all_reps, dim=0)
ch_std   = ch.std(all_reps, dim=0)

print("Statistics shape:", ch_mean.shape)

# Dump mean, std vectors for later use:
np_mean = ch_mean.cpu().numpy()
np_std  = ch_std.cpu().numpy()
np.save(prefix + "feature_mean", np_mean)
np.save(prefix + "feature_std",   np_std)
