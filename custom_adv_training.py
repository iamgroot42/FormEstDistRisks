import torch as ch
import numpy as np
from robustness.train import train_model
from robustness.tools import helpers
from robustness import defaults
from robustness.defaults import check_and_fill_args
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS


from itertools import combinations 
import cox
import utils
import argparse


def custom_train_loss(model, inp, targets, delta):
	with ch.no_grad():
		(logits, features), _ = model(inp, with_latent=True)
	w = model.module.model.classifier.weight
	
	# Define highest delta achievable
	max_val = float('inf') #1e10

	min_values = ch.ones((logits.shape[0], w.shape[0])).cuda() * max_val
	# When working with means, set inf to some other large value

	# Calculate feature statistics across batch (for relative scoring of delta values)
	features_mean, features_std = ch.mean(features, dim=0).unsqueeze(0), ch.std(features, dim=0).unsqueeze(0)
	features_mean = features_mean.detach()
	features_std  = features_std.detach()

	indices = ch.arange(logits.shape[0])
	for j in range(w.shape[0]):
		numerator = logits[indices, targets] - logits[:, j]
		denominator = w[j, :].unsqueeze(0) - w[targets, :]
		these_indices = (targets != j)
		deltas_care_about = numerator[these_indices].unsqueeze(1) / denominator[these_indices]
		# Only consider delta values which are achievable (ReLU), if not set to inf
		deltas_care_about[features[these_indices] + deltas_care_about < 0] = max_val
		# Scale delta values
		deltas_care_about = ch.abs(ch.div(deltas_care_about - features_mean, features_std + 1e-8))
		# deltas_care_about = ch.abs(deltas_care_about)
		# Weed out NaNs (set to max_value)
		deltas_care_about[deltas_care_about != deltas_care_about] = max_val
		min_values[these_indices, j], _ = ch.min(deltas_care_about, dim=1) # minimum across neurons
	# extra_loss = ch.stack([ch.mean(min_values[i, :][min_values[i, :] != max_val]) for i in range(min_values.shape[0])], dim=0)
	extra_loss = ch.mean(min_values, dim=1)
	# Minimum perturbation across classes
	extra_loss, _ = ch.min(min_values, dim=1)
	# Weed out NaNs (set to max_value)
	extra_loss = extra_loss[extra_loss == extra_loss]
	# Ignore 'cannot be attacked' (corresponding to inf) inputs for loss
	extra_loss = extra_loss[extra_loss != max_val]
	# If empty (all examples cannot be perturbed), contribute nothing to loss
	if extra_loss.size()[0] == 0:
		return 0
	extra_loss = extra_loss.mean().cuda()
	return - delta * extra_loss


def custom_train_loss_better(model, inp, targets, top_k, delta_1, delta_2):
	with ch.no_grad():
		(logits, features), _ = model(inp, with_latent=True)
	# w = model.module.model.classifier.weight
	w = model.module.model.linear.weight
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		diffs.append(ch.max(ch.abs(diff[0] - diff[1])))
	# Consider this across all class pairs (pick maximum possible)
	first_term = ch.max(ch.stack(diffs, dim=0))

	# TODO : consider only topk (much like second_term)

	diffs_2 = []
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2)) / features_norm
		# use_these, _ = ch.max(normalized_drop_term, dim=1)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	# second_term, _ = ch.max(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	return delta_1 * first_term + delta_2 * second_term


def custom_train_loss_better_modified(model, inp, targets, top_k, delta_1, delta_2):
	with ch.no_grad():
		(logits, features), _ = model(inp, with_latent=True)
	w = model.module.model.classifier.weight
	# w = model.module.model.linear.weight
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		topk_diff, _ = ch.topk(ch.abs(diff[0] - diff[1]), top_k)
		diffs.append(ch.mean(topk_diff))
	# Consider this across all class pairs (pick maximum possible)
	first_term = ch.max(ch.stack(diffs, dim=0))

	# TODO : consider only topk (much like second_term)

	diffs_2 = []
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2)) / features_norm
		# use_these, _ = ch.max(normalized_drop_term, dim=1)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	# second_term, _ = ch.max(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	return delta_1 * first_term + delta_2 * second_term


def custom_train_loss_better_faster(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	# with ch.no_grad():
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight
	# w = model.module.model.linear.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		diffs.append(ch.max(ch.abs(diff[0] - diff[1])))
	# Consider this across all class pairs (pick maximum possible)
	first_term = ch.max(ch.stack(diffs, dim=0))

	diffs_2 = []
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2)) / features_norm
		# use_these, _ = ch.max(normalized_drop_term, dim=1)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	# second_term, _ = ch.max(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term)


def as_it_is_loss(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	# w = model.module.model.classifier.weight
	w = model.module.model.linear.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		topk_diff, _ = ch.topk(ch.abs(diff[0] - diff[1]), top_k)
		diffs.append(ch.mean(topk_diff))
	first_term = ch.max(ch.stack(diffs, dim=0))

	diffs_2 = []
	# Consider detaching this term for possibly less orthogonal gradients?
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2) / features_norm)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	# delta_3 = 1e3
	# third_term, _ =  ch.topk(ch.std(features, dim=0), top_k)
	# third_term = ch.mean(third_term)

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term)
	# return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term + delta_3 * third_term)


def as_it_is_loss_simpler(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		topk_diff, _ = ch.topk(ch.abs(diff[0] - diff[1]), top_k)
		diffs.append(ch.mean(topk_diff))
	first_term = ch.stack(diffs, dim=0)
	first_term, _ = ch.topk(first_term, 9) # n_classes - 1
	first_term = ch.mean(first_term)

	diffs_2 = []
	# Consider detaching this term for possibly less orthogonal gradients?
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2) / features_norm)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)
	
	delta_3 = 5e-1
	third_term, _ =  ch.topk(ch.std(features, dim=0), top_k)
	third_term = ch.mean(third_term)

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term + delta_3 * third_term)


def try_dave(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	eps = 1e-5
	classwise_scaled_deltas = []
	f_mean = ch.mean(features, dim=0).unsqueeze(0).detach()
	f_std  = ch.std(features, dim=0).unsqueeze(0).detach()
	f_w_norm = ch.sum(features, dim=1).unsqueeze(1)
	l_y = ch.stack([logits[i, y] for i,y in enumerate(targets)], dim=0)
	den_right = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		l_y_hat  = logits[:, i]
		den_left = w[i, :]
		denominator = den_left - den_right
		numerator   = (l_y - l_y_hat).unsqueeze(1)
		delta = ch.abs(numerator / denominator)
		scaled_delta = (delta - f_mean) / (f_std + eps)
		# Mask (do not consider same-class targets)
		scaled_delta *= (targets != i).unsqueeze(1).type(scaled_delta.type())
		# Filter our NANs
		scaled_delta[scaled_delta != scaled_delta] = eps
		# Square instead of abs (for cleaner gradients)
		scaled_delta = ch.pow(scaled_delta, 2)
		classwise_scaled_deltas.append(scaled_delta)
	classwise_scaled_deltas = ch.stack(classwise_scaled_deltas, dim=0)
	classwise_scaled_deltas, _ = ch.topk(classwise_scaled_deltas, top_k, dim=2)
	classwise_scaled_deltas = ch.mean(classwise_scaled_deltas)

	return ((logits, features), final_inp, loss, -delta_1 * classwise_scaled_deltas)


def try_dave_just_first(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	variance = ch.std(features, dim=0)
	reg_term, _ = ch.topk(variance, top_k)
	reg_term = ch.mean(reg_term)
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


def try_dave_just_second(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	mean = ch.mean(features, dim=0)
	_, indices = ch.topk(-mean, top_k)
	reg_term = -ch.mean(mean[indices])
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


def try_dave_just_third(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		max_w_diffs, _ = ch.topk(ch.pow(diff[0] - diff[1], 2), top_k)
		diffs.append(ch.mean(max_w_diffs))
	# Consider this across all class pairs (pick maximum possible)
	reg_term = ch.mean(ch.stack(diffs, dim=0))
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


def try_dave_just_fourth(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	# Consider detaching this term for possibly less orthogonal gradients?
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	first_logits = ch.stack([logits[i, y] for i, y in enumerate(targets)], dim=0)
	diffs = []
	# Iterate over classes
	for i in range(logits.shape[1]):
		second_logits = logits[:, i]
		these_indices = (targets == i)
		diff = ch.pow(first_logits - second_logits, 2)
		diff[these_indices] = np.inf
		diffs.append(diff)
	diffs = ch.stack(diffs, dim=0)
	diffs, _ = ch.topk(-diffs, 5, dim=0)
	reg_term = ch.mean(diffs)
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


def try_dave_just_fifth(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		max_w_diffs, _ = ch.topk(ch.pow(diff[0] - diff[1], 2), top_k)
		diffs.append(ch.mean(max_w_diffs))
	# Consider this across all class pairs (pick maximum possible)
	reg_term = ch.mean(ch.stack(diffs, dim=0))

	# reg_term_2,  = ch.topk(ch.mean(features), top_k, dim=0)
	# reg_term_2  =  -ch.mean(reg_term_2)
	reg_term_3, _ = ch.topk(ch.std(features, dim=0), top_k)
	reg_term_3 = ch.pow(reg_term_3, 2) # Get rid of square-root
	reg_term_3 = ch.mean(reg_term_3)
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term + delta_2 * reg_term_3)


def try_dave_just_sixth(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	diffs = []
	# Consider detaching this term for possibly less orthogonal gradients?
	features_norm = ch.sum(features, dim=1).unsqueeze(1)
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = ch.abs(features * (diff_2_1 - diff_2_2) / features_norm)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs.append(use_these)
	reg_term = ch.mean(ch.stack(diffs, dim=0), dim=0)
	reg_term = ch.mean(reg_term)
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


def custom_train_loss_best(model, inp, targets, top_k, delta_1, delta_2):
	(logits, features), _ = model(inp, with_latent=True)
	w = model.module.model.classifier.weight
	b = model.module.model.classifier.bias
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		max_w_diffs, _ = ch.topk(ch.abs(diff[0] - diff[1]), top_k)
		diffs.append(ch.mean(max_w_diffs))
	# Consider this across all class pairs (pick maximum possible)
	first_term = ch.mean(ch.stack(diffs, dim=0))

	# return delta_1 * first_term

	diffs_2 = []
	diff_2_1 = ch.stack([w[y, :] for y in targets], dim=0)
	# Iterate over classes
	eps = 1e-10
	for i in range(logits.shape[1]):
		diff_2_2 = w[i, :].unsqueeze(0)
		normalized_drop_term = features * (diff_2_1 - diff_2_2)
		denominator = ch.sum(normalized_drop_term, dim=1).unsqueeze(1) + eps
		normalized_drop_term /= denominator.repeat(1, normalized_drop_term.shape[1])
		normalized_drop_term = ch.abs(normalized_drop_term)
		use_these, _ = ch.topk(normalized_drop_term, top_k, dim=1)
		use_these = ch.mean(use_these, dim=1)
		diffs_2.append(use_these)
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	return delta_1 * first_term + delta_2 * second_term


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_arch', type=str, help='arch of model (resnet50/vgg19/desnetnet169)')
	parser.add_argument('--top_k', type=int, default=64, help='top-k (neurons) considered while calculating loss terms')
	parser.add_argument('--start_lr', type=float, default=0.1, help='starting LR for optimizer')
	parser.add_argument('--delta_1', type=float, default=1e0, help='loss coefficient for first term')
	parser.add_argument('--delta_2', type=float, default=1e0, help='loss coefficient for second term')
	parser.add_argument('--reg_type', type=int, default=1, help='Regularization type [0, 12]')
	parser.add_argument('--batch_size', type=int, default=128, help='Batch Size')
	parser.add_argument('--fast', type=bool, default=False, help='Use optimized method (single forward pass) with reg_type=1')

	parsed_args = parser.parse_args()
	for arg in vars(parsed_args):
		print(arg, " : ", getattr(parsed_args, arg))

	if parsed_args.reg_type == 0:
		def regularizer(model, inp, targets):
			return custom_train_loss(model, inp, targets, parsed_args.delta_1)
	elif parsed_args.reg_type == 1:
		def regularizer(model, inp, targets):
			return custom_train_loss_better(model, inp, targets, parsed_args.top_k, parsed_args.delta_1, parsed_args.delta_2)
	elif parsed_args.reg_type == 2:
		def regularizer(model, inp, targets):
			return custom_train_loss_best(model, inp, targets, parsed_args.top_k, parsed_args.delta_1, parsed_args.delta_2)
	elif parsed_args.reg_type == 3:
		pass
	elif parsed_args.reg_type == 4:
		def regularizer(model, inp, targets):
			return custom_train_loss_better_modified(model, inp, targets, parsed_args.top_k, parsed_args.delta_1, parsed_args.delta_2)
	elif parsed_args.reg_type == 5:
		pass
	elif parsed_args.reg_type == 6:
		pass
	elif parsed_args.reg_type == 7:
		pass
	elif parsed_args.reg_type == 8:
		pass
	elif parsed_args.reg_type == 9:
		pass
	elif parsed_args.reg_type == 10:
		pass
	elif parsed_args.reg_type == 11:
		pass
	elif parsed_args.reg_type == 12:
		pass
	else:
		print("Invalid regularization requested. Exiting")
		exit(0)

	if parsed_args.fast:
		def regularizer(model, inp, targets, train_criterion, adv, attack_kwargs):
			if parsed_args.reg_type == 1:
				return custom_train_loss_better_faster(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 3:
				return as_it_is_loss(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 5:
				return as_it_is_loss_simpler(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 6:
				return try_dave(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 7:
				return try_dave_just_first(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 8:
				return try_dave_just_second(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 9:
				return try_dave_just_third(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 10:
				return try_dave_just_fourth(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 11:
				return try_dave_just_fifth(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)
			elif parsed_args.reg_type == 12:
				return try_dave_just_sixth(model, inp, targets, parsed_args.top_k, parsed_args.delta_1,
					parsed_args.delta_2, train_criterion, adv, attack_kwargs)

	train_kwargs = {
	    'out_dir': "/p/adversarialml/as9rw/models_cifar10_%s/" % (parsed_args.model_arch),
	    'adv_train': 0,
	    'exp_name': 'custom_adv_train_try_%f_%f_%d_%f_%s_fast_%d' % (parsed_args.delta_1, parsed_args.delta_2, parsed_args.top_k, parsed_args.start_lr, parsed_args.reg_type, parsed_args.fast),
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
    	'let_reg_handle_loss': parsed_args.fast
	}

	ds_class = DATASETS[train_kwargs['dataset']]

	train_args = cox.utils.Parameters(train_kwargs)

	dx = utils.CIFAR10()
	dataset = dx.get_dataset()

	args = check_and_fill_args(train_args, defaults.TRAINING_ARGS, ds_class)
	args = check_and_fill_args(train_args, defaults.MODEL_LOADER_ARGS, ds_class)

	model, _ = make_and_restore_model(arch=parsed_args.model_arch, dataset=dataset)
	
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
