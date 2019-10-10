from cleverhans import attacks
import datasets
import copy
import json
import numpy as np


class Attack:
	def __init__(self, attack, dataset, wrap, session, params_path):
		self.attack = attack
		self.dataset = dataset
		self.params_path = params_path
		self.read_params()
		self.wrap = wrap
		self.session = session
		self.attack_object = self.attack(self.wrap, sess=self.session)
		if self.dataset.name not in self.attack_params.keys():
			print("[Attack] Params for %s dataset not found, default values will be used" % self.dataset.name)
		else:
			if self.name not in self.attack_params[self.dataset.name].keys():
				print("[Attack] Params for %s attack for %s dataset not found, default values will be used" % (self.name, self.dataset.name))

	def read_params(self):
		with open(self.params_path, 'r') as f:
			print("[Attack] Loading params from %s" % self.params_path)
			self.attack_params  = json.load(f)

	def attack_data(self, data, custom_params=None):
		params = copy.deepcopy(self.attack_params[self.dataset.name][self.name])
		if custom_params:
			# Override params with given list of params
			for k, v in custom_params.items():
				params[k] = v
		return self.attack_object.generate_np(data, **params)

	def batch_attack(self, data, custom_params=None, batch_size=32):
		perturbed_X = np.array([])
		for i in range(0, data.shape[0], batch_size):
			mini_batch = data[i: i + batch_size,:]
			if mini_batch.shape[0] == 0:
				break
			adv_x_mini = self.attack_data(mini_batch, custom_params)
			if perturbed_X.shape[0] != 0:
				perturbed_X = np.append(perturbed_X, adv_x_mini, axis=0)
			else:
				perturbed_X = adv_x_mini
		return perturbed_X


class MadryEtAl(Attack):
	def __init__(self, dataset, wrap, session, params_path="./params/robust_train.params"):
		self.name = "MadryEtAl"
		super().__init__(attack=attacks.MadryEtAl, dataset=dataset, wrap=wrap, session=session, params_path=params_path)


class FGSM(Attack):
	def __init__(self, dataset, wrap, session, params_path="./params/robust_train.params"):
		self.name = "FGSM"
		super().__init__(attack=attacks.FGSM, dataset=dataset, wrap=wrap, session=session, params_path=params_path)


# Update every time you add a new attack
attack_list = [MadryEtAl, FGSM]
