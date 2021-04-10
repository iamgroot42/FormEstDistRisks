import torch as ch
from robustness.datasets import GenericBinary, CIFAR
from robustness.model_utils import make_and_restore_model
import numpy as np
from tqdm import tqdm

import utils


if __name__ == "__main__":
	import sys
	model_type = sys.argv[1]
	constants = utils.CIFAR10()
	ds = constants.get_dataset()
	model = constants.get_model(model_type , "vgg19")

	attack_args = []
	attack_args.append({
		'constraint': '2',
		'eps':0.5,
		'do_tqdm': False,
		'use_best': True,
		'targeted': False
	})
	attack_args.append({
		'constraint':'inf',
		'eps': 8/255, 
		'do_tqdm': False,
		'use_best': True,
		'targeted': False
	})

	batch_size = 256
	_, test_loader = ds.make_loaders(batch_size=batch_size, workers=8, only_val=True, fixed_test_order=True)
	n_tries = 20
	observations = []
	for (im, label) in tqdm(test_loader):
		for i in range(n_tries):
			for attack_arg in attack_args:
				attack_arg['iterations'] = i + 1
				attack_arg['step_size'] = 2.5 * attack_arg['eps']  / attack_arg['iterations']
				(_, features), _ = model(im.cuda(), label, with_latent=True, make_adv=True, **attack_arg)
				observations.append(features.cpu().detach())

	observations = ch.cat(observations, 0).numpy()
	coeffs = np.corrcoef(observations, rowvar=False)
	np.save("sense_correlation", coeffs)
