import torch as ch
from torch.autograd import Variable
import sys
from tqdm import tqdm

def l2_project(orig_input, x, eps):
	diff = x - orig_input
	diff = diff.renorm(p=2, dim=0, maxnorm=eps)
	return ch.clamp(orig_input + diff, 0, 1)


def linf_project(orig_input, x, eps):
	diff = x - orig_input
	diff = ch.clamp(diff, -eps, eps)
	return ch.clamp(diff + orig_input, 0, 1)


def l1_project(orig_input, x, eps):
	diff = x - orig_input
	diff = diff.renorm(p=1, dim=0, maxnorm=eps)
	return ch.clamp(orig_input + diff, 0, 1)


def basic_project(orig_input, x, eps):
	return ch.clamp(x, 0, 1)


def project_pertb(p):
	if p == '2':
		return l2_project
	elif p == '1':
		return l1_project
	elif p == 'inf':
		return linf_project
	else:
		return basic_project


def custom_optimization(model, inp_og, target_rep, indices_mask, eps, p='2', iters=200, reg_weight=1e0, verbose=True):
	inp = Variable(inp_og.clone(), requires_grad=True)
	optimizer = ch.optim.Adam([inp], lr=0.001)
	iterator = range(iters)
	# use_best behavior
	best_loss, best_x = float('inf'), None
	if verbose:
		iterator = tqdm(iterator)
	for i in iterator:
		optimizer.zero_grad()
		# Get image rep
		(_, rep), _ = model(inp, with_latent=True, fake_relu=True)
		# Get loss
		loss = ch.div(ch.norm(rep - target_rep, dim=1), ch.norm(target_rep, dim=1))
		aux_loss  = ch.sum(ch.abs((rep - target_rep) * indices_mask), dim=1)
		aux_loss  = ch.div(aux_loss, ch.norm(target_rep * indices_mask, dim=1))
		opt_loss  = loss + reg_weight * aux_loss
		this_loss = opt_loss.mean().item()
		# Store best loss and x so far
		if best_loss > this_loss:
			best_loss = this_loss
			best_x    = inp.data.clone()
		if verbose:
			# Print loss
			iterator.set_description('Loss : %f' % this_loss)
		# Back-prop loss
		opt_loss.backward(ch.ones_like(opt_loss), retain_graph=True)
		optimizer.step()
		# Project data : constain ro eps p-norm ball
		inp.data = project_pertb(p)(inp_og, inp.data, eps)
	return best_x
	

def madry_optimization(model, inp_og, target_rep, indices_mask, eps, random_restart_targets, iters=100,
	reg_weight=1e0, p='2', verbose=True, custom_best=False, fake_relu=True, random_restarts=0, inject=None,
	only_latent=False):

	# Modified inversion loss that puts emphasis on non-matching neurons to have similar activations
	def custom_inversion_loss(m, inp, targ):
		output, rep = m(inp, with_latent=True, fake_relu=fake_relu, this_layer_output=inject)
		# Normalized L2 error w.r.t. the target representation
		if inject is None:
			# loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
			loss = ch.norm(rep - targ, dim=1)
			# Extra loss term (normalized)
			aux_loss = ch.sum(ch.abs((rep - targ) * indices_mask), dim=1)
			aux_loss = ch.div(aux_loss, ch.norm(targ * indices_mask, dim=1))
			# Lagrangian formulation:
			return loss + reg_weight * aux_loss, output
		else:
			rep_  = rep.view(rep.shape[0], -1)
			targ_ = targ.view(rep.shape[0], -1)
			# print(ch.sum(rep_  - targ_))
			# print(ch.norm(rep_ - targ_, dim=1))
			# print(ch.norm(targ_, dim=1))
			# print(ch.sum(ch.norm(rep_ - targ_, dim=1) - ch.norm(targ_, dim=1)))
			# exit(0)
			# print(ch.norm(rep_ - targ_, dim=1))
			# print(ch.norm(targ_, dim=1))
			loss  = ch.mean(ch.pow(rep_ - targ_, 2), dim=1) # Do not normalize
			# print(ch.mean(loss))
			# loss  = ch.div(ch.norm(rep_ - targ_, dim=1), ch.norm(targ_, dim=1))
			# Extra loss term (normalized)
			# flattened_term = (ch.abs((rep - targ) * indices_mask)).view(targ.shape[0], -1)
			# aux_loss       = ch.sum(flattened_term, dim=1)
			# masked_targ    = (targ * indices_mask).view(targ.shape[0], -1)
			# Do not normalize
			# aux_loss       = ch.div(aux_loss, ch.norm(masked_targ, dim=1))
			return loss, output

	if custom_best:
		# If True, use the 'only neuron i' based 'best' evaluation
		if custom_best is True:
			def custom_loss_fn(loss, x):
				# Check how much beyond minimum delta the  perturbation on i^th index is
				# Negative sign, since we want higher delta-diff to score better
				(_, rep), _ = model(x, with_latent=True, fake_relu=fake_relu, this_layer_output=inject)
				return - ch.sum((rep - target_rep) * indices_mask, dim=1)
			custom_best = custom_loss_fn
		# Else, expect custom_best function to be passed along
	else:
		# If nothing passed along, use simple comparison
		custom_best = None

	kwargs = {
		'custom_loss': custom_inversion_loss,
		'constraint': p,
		'eps': eps,
		'step_size': 2.5 * eps / iters,
		'iterations': iters,
		'targeted': True,
		'do_tqdm': verbose,
		'custom_best': custom_best,
		'random_restarts': random_restarts,
		'random_restart_targets': random_restart_targets
	}
	_, im_matched = model(inp_og, target_rep, make_adv=True, **kwargs)
	return im_matched


def n_grad(model, X, target):    
	# Calculate Fischer Information Matrix (expected)
	target_rep = model.predict(X)
	(_, rep), _ = model(inp, with_latent=True, fake_relu=True)
	cov = np.zeros((X.T.shape))
	for i in range(X.shape[0]):
		cov[:, i] = X[i].T * (Y[i] - Y_[i])
	cov = np.cov(cov)
	f_inv = np.linalg.pinv(cov)

	nat_grad =  np.matmul(f_inv, s_grad(model, X, Y))
	return nat_grad


def natural_gradient_optimization(model, inp_og, target_rep, indices_mask, eps, iters=100, reg_weight=1e0, p=None, verbose=True):
	# This technique ignores 'p'
	inp = Variable(inp_og.clone(), requires_grad=True)
	for i in range(iters):
		(_, rep), _ = model(inp, with_latent=True, fake_relu=True)
		# Get loss
		loss = ch.div(ch.norm(rep - target_rep, dim=1), ch.norm(target_rep, dim=1))
		aux_loss = ch.sum(ch.abs((rep - target_rep) * indices_mask), dim=1)
		aux_loss = ch.div(aux_loss, ch.norm(target_rep * indices_mask, dim=1))
		loss += reg_weight * aux_loss 
		# Calculate gradients
		loss.backward(ch.ones_like(loss), retain_graph=True)
		# Calculate covariance of loss gradient (flatten out to n * features shape)
		inp_grad_flat =  inp.grad.view(inp.shape[0], -1)
		inp_grad_flat -= ch.sum(inp_grad_flat, dim=0)
		loss_cov = ch.t(inp_grad_flat)
		loss_cov = ch.mm(loss_cov, inp_grad_flat) / loss_cov.shape[0]
		F = loss_cov.pinverse()
		# Compute modified gradient (flattened version)
		grad_inp = inp.grad.view(inp.shape[0], -1) @ F
		# Back-prop loss
		inp.data -= 1e-5 * grad_inp.view(inp.grad.shape)
		if verbose:
			print(loss.mean().item())
		# Project difference back to Lp norm ball
		inp.data = project_pertb(p)(inp_og, inp.data, eps)
		# Clip image to [0,1] range
		inp.data = ch.clamp(inp.data, 0, 1)
		# Zero gradient
		inp.grad.data.zero_()
	return inp.data


def n_free_optimization(model, inp_og, target_rep, eps, p='2', iters=200, verbose=True):
	inp = Variable(inp_og.clone(), requires_grad=True)
	optimizer = ch.optim.Adam([inp], lr=0.001)
	iterator = range(iters)
	if verbose:
		iterator = tqdm(iterator)
	for i in iterator:
		optimizer.zero_grad()
		# Get image rep
		(_, rep), _ = model(inp, with_latent=True, fake_relu=True)
		# Get loss
		opt_loss = ch.div(ch.norm(rep - target_rep, dim=1), ch.norm(target_rep, dim=1))
		if verbose:
			# Print loss
			iterator.set_description('Loss : %f' % opt_loss.mean().item())
		# Back-prop loss
		opt_loss.backward(ch.ones_like(opt_loss), retain_graph=True)
		optimizer.step()
		# Project data : constain ro eps p-norm ball
		inp.data = project_pertb(p)(inp_og, inp.data, eps)
	return inp.data
