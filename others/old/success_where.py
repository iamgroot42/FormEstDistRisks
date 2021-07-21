import numpy as np
import torch as ch
from robustness.model_utils import make_and_restore_model
from robustness.datasets import GenericBinary, CIFAR
import sys
import utils
from tqdm import tqdm

binary = False
# Load model
if binary:
	constants = utils.BinaryCIFAR()
else:
	constants = utils.CIFAR10()
ds = constants.get_dataset()

# model_path = "/p/adversarialml/as9rw/models_cifar10/cifar_linf_8.pt"
model_path = "/p/adversarialml/as9rw/models_cifar10/cifar_l2_0_5.pt"
# model_path = "/p/adversarialml/as9rw/models_cifar10/cifar_nat.pt"

# Load model
model_kwargs = {
	'arch': 'resnet50',
	'dataset': ds,
	'resume_path': model_path
}
model, _ = make_and_restore_model(**model_kwargs)
model.eval()

first_attack    = np.load(sys.argv[1])
second_attack   = np.load(sys.argv[2])
original_labels = np.load("/u/as9rw/work/fnb/attack_labels.npy")
batch_size =  128
first_success  = []
second_success = []


for i in tqdm(range(0, first_attack.shape[0], batch_size)):
	batch_x = first_attack[i: i + batch_size]
	batch_x = ch.from_numpy(batch_x).cuda()
	with ch.no_grad():
		pred_x, _ = model(batch_x)
	pred_x = ch.argmax(pred_x, dim=1).cpu().numpy()
	first_success.append(pred_x)

for i in tqdm(range(0, second_attack.shape[0], batch_size)):
	batch_y = second_attack[i: i + batch_size]
	batch_y = ch.from_numpy(batch_y).cuda()
	with ch.no_grad():
		pred_y, _ = model(batch_y)
	pred_y = ch.argmax(pred_y, dim=1).cpu().numpy()
	second_success.append(pred_y)

first_success  = np.concatenate(first_success, 0)
second_success = np.concatenate(second_success, 0)

n = second_success.shape[0] // first_success.shape[0]
actual_second_success = []
j = 0
for i in range(0, second_success.shape[0], n):
	actual_second_success.append(np.any(second_success[i:i+n] != original_labels[j]))
	j += 1

actual_second_success = np.array(actual_second_success)
first_success = (first_success != original_labels)

combined_success = np.logical_or(first_success, actual_second_success)

print("Attack success rate for first model : %f" % (np.mean(first_success)))
print("Attack success rate for second model : %f" % (np.mean(actual_second_success)))
print("Attack success rate for both models (union) : %f" % (np.mean(combined_success)))
