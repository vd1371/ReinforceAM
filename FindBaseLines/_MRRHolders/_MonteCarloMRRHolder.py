import numpy as np
import pprint
import joblib
from copy import deepcopy

class MCMRRHolder:

	def __init__(self, **params):

		self.n_elements = params.get('n_elements')
		self.n_steps = params.get("settings").n_steps
		self.dim_actions = params.get("dim_actions")
		self.IDs = params.get("env").asset_IDs
		self.logger = params.get("logger")
		self.base_direc = params.get("base_direc")
		self.n_samples = 0

		self.reset()

	def reset(self):
		self.all_mrrs = {}
		for id_ in self.IDs:
			self.all_mrrs[id_] = np.zeros((self.n_elements, self.n_steps, self.dim_actions))

	def update(self, network_mrrs):

		self.n_samples += 1
		for id_ in self.IDs:
			for ne in range(self.n_elements):
				for step in range(self.n_steps):
					action = int(network_mrrs[id_][ne][step])
					self.all_mrrs[id_][ne][step][action] += 1

	def _convert_mrrs_to_list(self):
		for id_ in self.IDs:
			self.all_mrrs[id_] = np.argmax(self.all_mrrs[id_], axis = 2).tolist()

	def get_baseline(self):
		self._convert_mrrs_to_list()
		joblib.dump(self.all_mrrs, self.base_direc + "FixedPlans.json")
		self.logger.info(pprint.pformat(self.all_mrrs))

		return self.all_mrrs
