import numpy as np

from ._save_actions_histogram import _save_actions_histogram
from ._save_costs_histogram import _save_costs_histogram
from ._save_averages_holders import _save_averages_holders


class SimResultsHolder:

	def __init__(self, **params):
		
		self.n_elements = params.pop("n_elements", 3)
		self.n_steps = params.pop('n_steps', 10)
		self.dim_actions = params.pop('dim_actions', 4)
		self.n_sim = params.pop('n_sim', 1000)
		self.dt = params.pop('dt', 2)
		self.logger = params.pop('logger', None)
		self.IDs = params.pop("env").asset_IDs
		self.base_direc = params.pop("base_direc")

		self.refresh()

	def refresh(self):

		self.R_avg = 0
		self.ac_avg = 0
		self.uc_avg = 0

		self.ac_avg_holder = []
		self.uc_avg_holder = []

		self.i = 0
		self.A_histogram = {id_: np.zeros((self.n_elements, self.n_steps, self.dim_actions)) for id_ in self.IDs}
		self.ac_histogram = {id_: np.zeros((self.n_elements, self.n_steps)) for id_ in self.IDs}
		self.uc_histogram = {id_: np.zeros((self.n_elements, self.n_steps)) for id_ in self.IDs}

	def update_histogram(self, actions, agency_costs, user_costs):

		for id_ in self.IDs:
			for ne in range(self.n_elements):
				for step in range(self.n_steps):

					action = actions[id_][ne][step]
					self.A_histogram[id_][ne][step][action] += 1
					self.ac_histogram[id_][ne][step] += agency_costs[id_][ne][step]
					self.uc_histogram[id_][ne][step] += user_costs[id_][ne][step]

	def update_averages(self, R_avg, ac_avg, uc_avg):

		self.R_avg = (self.R_avg * self.i + R_avg)/(self.i+1)
		self.ac_avg = (self.ac_avg * self.i + ac_avg)/(self.i+1)
		self.uc_avg = (self.uc_avg * self.i + uc_avg)/(self.i+1)

		self.ac_avg_holder.append(ac_avg)
		self.uc_avg_holder.append(uc_avg)

		self.i += 1

	def get_avgs(self):
		return self.R_avg, self.ac_avg, self.uc_avg

	def get_histogram(self, for_):

		for id_ in self.A_histogram:
			self.A_histogram[id_] /= self.n_sim
			self.ac_histogram[id_] /= self.n_sim
			self.uc_histogram[id_] /= self.n_sim

			if not self.logger is None:
				for ne in range(self.n_elements):
					self.logger.info(f"Asset{id_}-Elem{ne} Actions:\n{self.A_histogram[id_][ne]}")

		_save_actions_histogram(**self.__dict__)
		_save_costs_histogram(**self.__dict__)
		_save_averages_holders(for_, **self.__dict__)



	

		