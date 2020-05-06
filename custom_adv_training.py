import torch as ch
from robustness.train import train_model
from robustness.tools import helpers
from robustness import defaults
from robustness.defaults import check_and_fill_args
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS
import cox
import utils
import argparse

import custom_losses


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str,   default='vgg19', help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--top_k',      type=int,   default=64,      help='top-k (neurons) considered while calculating loss terms')
	parser.add_argument('--start_lr',   type=float, default=0.1,     help='starting LR for optimizer')
	parser.add_argument('--delta_1',    type=float, default=1e0,     help='loss coefficient for first term')
	parser.add_argument('--delta_2',    type=float, default=1e0,     help='loss coefficient for second term')
	parser.add_argument('--reg_type',   type=int,   default=3,       help='Regularization type [0, 12]')
	parser.add_argument('--batch_size', type=int,   default=128,     help='Batch Size')

	# Print arguments
	parsed_args = parser.parse_args()
	for arg in vars(parsed_args):
		print(arg, " : ", getattr(parsed_args, arg))

	# Get loss function
	loss_fn = custom_losses.get_custom_loss(parsed_args.reg_type)
	if loss_fn is None:
		raise ValueError("Invalid regularization requested. Exiting")
		exit(0)

	# Define running statistics for feature mean, std
	feature_normalizer = ch.nn.BatchNorm1d(512, affine=False).cuda()

	def regularizer(model, inp, targets, train_criterion, adv, attack_kwargs):
			return loss_fn(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, feature_normalizer, attack_kwargs)

	train_kwargs = {
	    'out_dir': "/p/adversarialml/as9rw/models_cifar10_%s/" % (parsed_args.model_arch),
	    'adv_train': 0,
	    'exp_name': 'sensitive_train_%f_%f_%d_%f_%s_new' % (parsed_args.delta_1, parsed_args.delta_2, parsed_args.top_k, parsed_args.start_lr, parsed_args.reg_type),
	    'dataset': 'cifar',
    	'arch': parsed_args.model_arch,
    	'adv_eval': True,
    	'batch_size': parsed_args.batch_size,
    	'attack_lr': (2.5 * 0.5) / 10,
    	'constraint': '2',
    	'eps': 0.5,
	    'attack_steps': 20,
    	'use_best': True,
    	'eps_fadein_epochs': 0,
    	'random_restarts': 0,
    	'lr': parsed_args.start_lr,
    	'use_adv_eval_criteria': 1,
    	'regularizer': regularizer,
    	'let_reg_handle_loss': True
	}

	ds_class = DATASETS[train_kwargs['dataset']]
	train_args = cox.utils.Parameters(train_kwargs)

	dx      = utils.CIFAR10()
	dataset = dx.get_dataset()

	args = check_and_fill_args(train_args, defaults.TRAINING_ARGS, ds_class)
	args = check_and_fill_args(train_args, defaults.MODEL_LOADER_ARGS, ds_class)

	model, _ = make_and_restore_model(arch=parsed_args.model_arch, dataset=dataset)
	
	# Make the data loaders
	train_loader, val_loader = dataset.make_loaders(args.workers, args.batch_size, data_aug=bool(args.data_aug))

	# Prefetches data to improve performance
	train_loader = helpers.DataPrefetcher(train_loader)
	val_loader   = helpers.DataPrefetcher(val_loader)

	store = cox.store.Store(args.out_dir, args.exp_name)
	args_dict = args.as_dict() if isinstance(args, cox.utils.Parameters) else vars(args)
	schema = cox.store.schema_from_dict(args_dict)
	store.add_table('metadata', schema)
	store['metadata'].append_row(args_dict)

	model = train_model(args, model, (train_loader, val_loader), store=store)
