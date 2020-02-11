from robustness.train import train_model
from robustness.tools import helpers
from robustness import defaults
from robustness.defaults import check_and_fill_args
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS

import cox
import torch as ch
import sys
import utils

arch = "vgg19"

delta = float(sys.argv[1])


def custom_train_loss(model, inp, targets):
	(logits, features), _ = model(inp, with_latent=True)
	w = model.module.model.classifier.weight
	min_values = ch.ones((logits.shape[0], w.shape[0])).cuda() * float('inf')
	# When working with means, set inf to some other large value
	# min_values = ch.ones((logits.shape[0], w.shape[0])).cuda() * 1e3

	indices = ch.arange(logits.shape[0])
	for j in range(w.shape[0]):
		numerator = logits[indices, targets] - logits[:, j]
		denominator = w[j, :].unsqueeze(0) - w[targets, :]
		these_indices = (targets != j)
		deltas_care_about = ch.abs(numerator[these_indices].unsqueeze(1) / denominator[these_indices])
		# Only consider delta values which are achievable (ReLU), if not set to inf
		deltas_care_about[features[these_indices] + deltas_care_about < 0] = float('inf')
		# When working with means, set inf to some other large value
		# deltas_care_about[deltas_care_about == float('inf')] = 1e3
		min_values[these_indices, j], _ = ch.min(deltas_care_about, dim=1) # minimum across neurons
		# min_values[these_indices, j] = ch.mean(deltas_care_about, dim=1) # average across neurons

	# Minimum perturbation across classes
	extra_loss, _ = ch.min(min_values, dim=1) # minimum across classes
	# extra_loss = ch.mean(min_values, dim=1) # average across classes
	# Average min-perturbation across batch
	extra_loss = extra_loss.mean().cuda()
	return delta / extra_loss


train_kwargs = {
    'out_dir': "/p/adversarialml/as9rw/models_cifar10_vgg_new/",
    'adv_train': 0,
    'exp_name': 'custom_adv_train_try_' + str(delta),
    'dataset': 'cifar',
    'arch': arch,
    'regularizer': custom_train_loss
}

ds_class = DATASETS[train_kwargs['dataset']]

train_args = cox.utils.Parameters(train_kwargs)

dx = utils.CIFAR10()
dataset = dx.get_dataset()

args = check_and_fill_args(train_args, defaults.TRAINING_ARGS, ds_class)
args = check_and_fill_args(train_args, defaults.MODEL_LOADER_ARGS, ds_class)

model, _ = make_and_restore_model(arch=arch, dataset=dataset)

# Make the data loaders
train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug))

# Prefetches data to improve performance
train_loader = helpers.DataPrefetcher(train_loader)
val_loader = helpers.DataPrefetcher(val_loader)

store = cox.store.Store(args.out_dir, args.exp_name)
args_dict = args.as_dict() if isinstance(args, cox.utils.Parameters) else vars(args)
schema = cox.store.schema_from_dict(args_dict)
store.add_table('metadata', schema)
store['metadata'].append_row(args_dict)

model = train_model(args, model, (train_loader, val_loader), store=store)
