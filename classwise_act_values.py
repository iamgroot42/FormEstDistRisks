import torch as ch
import utils
from robustness.model_utils import make_and_restore_model
import numpy as np
import sys
from tqdm import tqdm

model_arch   = sys.argv[1]
model_type   = sys.argv[2]
prefix       = sys.argv[3]

dx = utils.CIFAR10()
ds = dx.get_dataset()

model = dx.get_model(model_type, model_arch)

batch_size = 256
classwise_reps = [[] for _ in range(10)]
train_loader, val_loader = ds.make_loaders(batch_size=batch_size, workers=8)

# Calculate emperical mean, variance to normalize representations
def get_reps(data_loader):
	for (im, label) in tqdm(data_loader):
		with ch.no_grad():
			(_, rep), _ = model(im, with_latent=True)
			rep = rep.cpu()
			for i, y in enumerate(label):
				classwise_reps[y].append(rep[i].numpy())

# Get reps for train ,val sets
get_reps(train_loader)
print("Done with training data")
get_reps(val_loader)
print("Done with testing data")

for i, c in enumerate(classwise_reps):
	classwise_reps[i] = np.array(c)
	classwise_reps[i] = (np.mean(classwise_reps[i], 0), np.std(classwise_reps[i], 0))

classwise_reps = np.array(classwise_reps)
print(classwise_reps.shape)
np.save(prefix + "feature_all", classwise_reps)
