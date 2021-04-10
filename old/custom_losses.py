from itertools import combinations
import numpy as np
import torch as ch


FUNC_MAPPING = {}

class function_mapping(object):

	def __init__(self, index):
		self.index = index


	def __call__(self, f):
		def wrapped_f(*args):
			f(*args)
		
		FUNC_MAPPING[self.index] = f
		return wrapped_f


def get_custom_loss(loss_id):
	return FUNC_MAPPING.get(loss_id, None)


@function_mapping(1)
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
	second_term = ch.mean(ch.stack(diffs_2, dim=0), dim=0)
	second_term = ch.mean(second_term)

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term)


@function_mapping(3)
def as_it_is_loss(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, feature_normalizer, attack_kwargs):
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

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term)


@function_mapping(5)
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


@function_mapping(6)
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


@function_mapping(7)
def try_dave_just_first(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	variance = ch.std(features, dim=0)
	reg_term, _ = ch.topk(variance, top_k)
	reg_term = ch.mean(reg_term)
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


@function_mapping(8)
def try_dave_just_second(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	mean = ch.mean(features, dim=0)
	_, indices = ch.topk(-mean, top_k)
	reg_term = -ch.mean(mean[indices])
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term)


@function_mapping(9)
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


@function_mapping(10)
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


@function_mapping(11)
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

	reg_term_3, _ = ch.topk(ch.std(features, dim=0), top_k)
	reg_term_3 = ch.pow(reg_term_3, 2) # Get rid of square-root
	reg_term_3 = ch.mean(reg_term_3)
	
	return ((logits, features), final_inp, loss, delta_1 * reg_term + delta_2 * reg_term_3)


@function_mapping(12)
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


@function_mapping(13)
def directly_with_gm(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, feature_normalizer, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Pass features through BN layer to get updated mean, std
	feature_normalizer(features)

	# Calculate normal loss
	loss = train_criterion(logits, targets)
	
	n_features = w.shape[1]
	n_classes  = logits.shape[1]

	# Large constant
	large_const = 1e3
	small_const = 1e-5

	# Aggregate across all classes
	delta_values = []
	for i in range(n_classes):
		numerator   = logits.gather(1, targets.view(-1,1)) - logits[:, i].unsqueeze(1)
		denominator = w[i].unsqueeze(0) - w[targets]
		# Do not count cases where label = target
		mask = (targets != i).unsqueeze(1)
		term = mask * (numerator / denominator)
		# Set values with NANs to a small consant
		term[term != term] = 1 + small_const
		delta_values.append(term)
	delta_values = ch.stack(delta_values)

	# Normalize delta values with feature-wise mean, std
	# Do not back-propagate beyond these terms (make life easier for the model)
	with ch.no_grad():
		mean, var =  feature_normalizer.running_mean, feature_normalizer.running_var
		delta_values = (delta_values - mean) / ch.sqrt(var + small_const)

	# Work in log-space (to implciitly minimize product)
	# Square delta values for 1) bette gradients, and 2) positive domain for log
	# delta_values = ch.log(ch.pow(delta_values, 2))
	delta_values = ch.pow(delta_values, 2)

	# Focus on only the top-k values (to not overwhelm model)
	delta_values = delta_values.transpose(0, 1)
	delta_values.resize_((delta_values.shape[0], delta_values.shape[1] * delta_values.shape[2]))
	delta_values, _ = ch.topk(delta_values, top_k, dim=1, largest=False)

	# Do not count -inf (0 value to log as input)
	# delta_values[delta_values == -np.inf] = 0
	extra_term   = delta_values.mean()
	extra_term   = large_const - extra_term
	# print(extra_term)
	# exit(0)

	return ((logits, features), final_inp, loss, delta_1 * extra_term)


@function_mapping(14)
def directly_with_gm(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, feature_normalizer, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Pass features through BN layer to get updated mean, std
	feature_normalizer(features)

	# Calculate normal loss
	loss = train_criterion(logits, targets)
	
	n_features = w.shape[1]
	n_classes  = logits.shape[1]

	# Large constant
	large_const = 1e1
	small_const = 1e-5

	# Aggregate across all classes
	delta_values = []
	for i in range(n_classes):
		numerator   = logits.gather(1, targets.view(-1,1)) - logits[:, i].unsqueeze(1)
		denominator = w[i].unsqueeze(0) - w[targets]
		# Do not count cases where label = target
		mask = (targets != i).unsqueeze(1)
		term = mask * (numerator / denominator)
		# Set values with NANs to a small consant
		term[term != term] = 1 + small_const
		delta_values.append(term)
	delta_values = ch.stack(delta_values)

	# Normalize delta values with feature-wise mean, std
	# Do not back-propagate beyond these terms (make life easier for the model)
	with ch.no_grad():
		var = feature_normalizer.running_var
		delta_values /= ch.sqrt(var + small_const)

	# Work in log-space (to implciitly minimize product)
	# Square delta values for 1) bette gradients, and 2) positive domain for log
	delta_values = ch.log(ch.pow(delta_values, 2))
	# delta_values = ch.pow(delta_values, 2)

	# Focus on only the top-k values (to not overwhelm model)
	# delta_values = delta_values.transpose(0, 1)
	# delta_values.resize_((delta_values.shape[0], delta_values.shape[1] * delta_values.shape[2]))
	# delta_values, _ = ch.topk(delta_values, top_k, dim=1, largest=False)

	# DO not count NaN/INF values
	delta_values[ch.abs(delta_values) == np.inf] = 0
	delta_values[delta_values != delta_values] = 0
	extra_term   = delta_values.mean()
	extra_term   = large_const - extra_term

	return ((logits, features), final_inp, loss, delta_1 * extra_term)


@function_mapping(15)
def focus_on_weights(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, feature_normalizer, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Pass features through BN layer to get updated mean, std
	# feature_normalizer(features)

	# Calculate normal loss
	loss = train_criterion(logits, targets)
	
	n_features = w.shape[1]
	n_classes  = logits.shape[1]

	var = ch.pow(ch.std(features, 0), 2)
	sorted_weights, _ = ch.sort(w, dim=0)
	diffs = []
	for i in range(n_classes - 1):
		diffs.append(ch.pow(sorted_weights[i+1] - sorted_weights[i], 2) * var)

	diffs    = ch.stack(diffs)
	diffs, _ = ch.topk(diffs, top_k)
	aux_loss = diffs.mean()

	return ((logits, features), final_inp, loss, delta_1 * aux_loss)

@function_mapping(16)
def i_am_mad_loss(model, inp, targets, top_k, delta_1, delta_2, train_criterion, adv, feature_normalizer, attack_kwargs):
	(logits, features), final_inp = model(inp, target=targets, make_adv=adv, with_latent=True, **attack_kwargs)
	w = model.module.model.classifier.weight

	# Calculate normal loss
	loss = train_criterion(logits, targets)

	# Pass features through BN layer to get updated mean, std
	feature_normalizer(features)

	var = feature_normalizer.running_var
	
	# First term : minimize weight values for same feature across any two different classes (nC2)
	diffs = []
	for c in combinations(range(logits.shape[1]), 2):
		# Across all possible (i, j) class pairs
		diff = w[c, :]
		# Note differences in weight values for same feature, different classes
		all_diffs = ch.pow(diff[0] - diff[1])
		print(var.shape)
		print(all_diffs.shape)
		exit(0)
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

	return ((logits, features), final_inp, loss, delta_1 * first_term + delta_2 * second_term)