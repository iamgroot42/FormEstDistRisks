from cleverhans import attacks
import datasets
import copy
import json

# Update every time you add a new attack
attack_list = [MadryEtAl, FGSM]


class Attack:
	def __init__(self, attack, dataset, wrap, session, params_path="./params/robust_train.params"):
		self.attack = attack
		self.dataset = dataset
		self.params_path = params_path
		self.read_params()
		self.dataset_not_in_check(self.dataset.__name__)
		self.attack_not_in_check(self.attack.__name__)
		self.wrap = wrap
		self.session = session
		self.attack_object = self.attack(self.wrap, sess=self.session)

	def dataset_not_in_check(self, y):
		if y not in [x.__name__ for x in datasets.dataset_list]:
			raise NotImplementedError("Dataset %s not implemented in datasets module" % y)

	def attack_not_in_check(self, y):
		if y not in [x.__name__ for x in attack_list]:
			raise NotImplementedError("Attack %s not implemented in attacks module" % y)

	def read_params(self):
		with open(self.params_path, 'r') as f:
			self.attack_params  = json.load(f)
			for k, _ in self.attack_params.keys():
				self.dataset_not_in_check(k)

	def attack_data(self, data, custom_params=None):
		params = copy.deepcopy(self.attack_params[self.dataset.__name__][self.attack._name__])
		if custom_params:
			# Override params with given list of params
			for k, v in custom_params.items():
				params[k] = v
		return self.attack_object.generate_np(data, **params)


class MadryEtAl(Attack):
	def __init__(self, dataset, wrap, session):
		super().__init__(attack=attacks.MadryEtAl, dataset=dataset, wrap=wrap, session=session)


class FGSM(Attack):
	def __init__(self, dataset, wrap, session):
		super().__init__(attack=attacks.FGSM, dataset=dataset, wrap=wrap, session=session)
